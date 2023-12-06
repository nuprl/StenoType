from concurrent import futures
from functools import lru_cache
from pathlib import Path
from subprocess import DEVNULL, PIPE
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Any, Optional
import argparse
import evaluate
import Levenshtein
import numpy as np
import re
import shutil
import subprocess
import tarfile
import tempfile

from model import Tokenizer
from inference import Config
from util import transform
import util

ACCURACY_METRIC = evaluate.load("accuracy")

FILE_RE = re.compile("^// FILE: (\S+)$")
TSC_ERROR_RE = re.compile("^(.*\.ts)\((\d+),\d+\):")

LEVENSHTEIN_THRESHOLD = 0.99

TSC_PATH = shutil.which("tsc")
if not TSC_PATH:
    print("Could not find tsc")
    exit(1)


@lru_cache(maxsize=32)
def _accuracy(tokenizer: PreTrainedTokenizer, original: str, output: str) -> float:
    # Tokenize the original and output, and pad them to the same length
    # NumPy tensors may be more memory efficient than Python lists
    original_tokens, output_tokens = tokenizer(
        [original, output],
        padding=True,
        return_attention_mask=False,
        return_tensors="np",
    )["input_ids"]

    return ACCURACY_METRIC.compute(
        references=original_tokens, predictions=output_tokens
    )["accuracy"]


@lru_cache(maxsize=32)
def _levenshtein(original: str, output: str) -> float:
    return Levenshtein.ratio(original, output)


@lru_cache(maxsize=32)
def _tsc(content: str, tmpdir: Optional[str] = None) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ts", dir=tmpdir, encoding="utf-8"
    ) as f:
        # Save content to temp file
        print(content, file=f, end="", flush=True)
        tmpfile = Path(f.name)

        # Run tsc on temp file
        args = [
            str(TSC_PATH),
            "--noEmit",                     # don't emit JavaScript
            "--esModuleInterop",            # better handling of CJS/ES6 modules
            "--moduleResolution", "node",   # use node's module resolution
            "--target", "es2022",           # enable support for es2022 features
            "--lib", "es2022,dom",          # enable additional library APIs
            str(tmpfile),
        ]  # fmt: skip
        result = subprocess.run(
            args, stdout=PIPE, stderr=DEVNULL, encoding="utf-8", cwd=tmpfile.parent
        )

        return result.returncode == 0, result.stdout


@lru_cache(maxsize=32)
def _split_files(name: str, content: str) -> dict[str, dict[str, Any]]:
    # Given the content of a bundled project, split it up by files
    res: dict[str, list[str]] = {}
    curr_file = name
    for line in content.splitlines():
        # What file are we now processing?
        if match := FILE_RE.match(line):
            curr_file = match[1]

        # Add the line to the dictionary
        if curr_file in res:
            res[curr_file].append(line)
        else:
            res[curr_file] = [line]

    # File contents are lists of strings, need to join them to get strings
    return {k: {"content": "\n".join(v)} for k, v in res.items()}


@lru_cache(maxsize=32)
def _lines_to_file(content: str) -> list[str]:
    # Initialize index/line 0 to the empty string
    res: list[str] = [""]
    curr_file = ""
    for line in content.splitlines():
        if match := FILE_RE.match(line):
            curr_file = match[1]
        res.append(curr_file)
    # Add another line; sometimes errors will point one past the last line
    res.append(curr_file)
    return res


@lru_cache(maxsize=32)
def _split_tsc_logs(logs: str) -> list[str]:
    # Takes a string containing tsc output and returns a list of errors
    errors_list = []
    for line in logs.splitlines():
        if TSC_ERROR_RE.match(line):
            # This line is an error
            errors_list.append(line)
        else:
            # This line is part of the previous error
            errors_list[-1] += f"\n{line}"
    return errors_list


@lru_cache(maxsize=32)
def _split_errors(name: str, output: str, tsc_logs: str) -> dict[str, dict[str, Any]]:
    # lines is a list of file names, such that
    # line[i] is the name of the file that line i of output was originally from
    lines = _lines_to_file(output)

    # Get the set of files in this bundle, but remove the empty string
    files = set(lines) - {""}

    # Split the tsc logs into a list, one item per error (some errors span multiple lines)
    errors_list = _split_tsc_logs(tsc_logs)

    # If files is empty, then this is not a bundle, so all errors map to the single file
    if not files:
        return {name: {"errors": errors_list}}

    res: dict[str, dict[str, Any]] = {f: {"errors": []} for f in files}
    for e in errors_list:
        # Skip library files
        if (match := TSC_ERROR_RE.match(e)) and "node_modules" not in e:
            file = lines[int(match[2])]
            if file in res:
                res[file]["errors"].append(e)
            else:
                # Error refers to a line that is not from a file in the bundle
                # This could be code added by the bundler or the model
                # Either way, we add a dummy file
                res[name] = {"errors": [e]}

    return res


def _unzip_files_results(mapping: dict[str, dict]) -> dict[str, list]:
    # mapping is a dictionary that maps filenames to a results dictionary
    # However, saving this in a dataset causes all filenames of all entries to be merged
    # So we restructure this by splitting the keys/values of the dictionary into two lists
    ret: dict[str, list] = {"files": [], "results": []}
    for f, r in mapping.items():
        ret["files"].append(f)
        ret["results"].append(r)
    return ret


def _zip_files_results(mapping: dict[str, list]) -> dict[str, dict]:
    # We have a dict {"files": files_list, "results": results_list}
    # and we want to zip the two lists together to make a dict
    return dict(zip(mapping["files"], mapping["results"]))


def _evaluate_files(
    name: str, original: str, output: str, tsc_logs: str
) -> dict[str, dict[str, Any]]:
    # Split the original file; later we need to extract type definitions
    split_original = _split_files(name, original)

    # Merge the output and errors maps
    split_output = _split_files(name, output)
    split_errors = _split_errors(name, output, tsc_logs)
    file_results = {
        k: split_output.get(k, {"content": ""}) | split_errors.get(k, {"errors": []})
        for k in (split_output.keys() | split_errors.keys())
    }

    # Now iterate over each file
    for file, res in file_results.items():
        original = split_original.get(file, {"content": ""})["content"]
        original_untyped = transform.delete_types(original)
        file_content = res["content"]
        file_untyped = transform.delete_types(file_content)

        # Count errors
        file_results[file]["num_errors"] = len(res["errors"])

        # Compute type annotation stats
        annotation_text = [
            transform.node_to_str(n.named_children[0])
            for n in transform.extract_type_annotation_nodes(file_content)
        ]
        annotations_added = len(annotation_text)
        file_results[file]["num_annotation_sites"] = transform.count_annotation_sites(
            file_content, exclude_child_annotations=True
        )
        file_results[file]["num_annotations_added"] = annotations_added
        file_results[file]["num_annotations_trivial"] = len(
            [n for n in annotation_text if "any" in n or "Function" in n]
        )
        file_results[file]["type_annotations"] = set(annotation_text)

        # Compute type definition stats
        original_types = set(transform.get_type_definition_names(original_untyped))
        output_types = set(transform.get_type_definition_names(file_content))
        definitions_added = len(output_types - original_types)
        definitions_used = transform.get_used_type_definitions(file_content)
        annotations_undefined = transform.get_undefined_type_names(file_content)
        file_results[file]["num_definitions_added"] = definitions_added
        file_results[file]["num_definitions_used"] = len(definitions_used)
        file_results[file]["num_definitions_undefined"] = len(annotations_undefined)
        file_results[file]["type_definitions"] = output_types
        file_results[file]["type_definitions_used"] = definitions_used
        file_results[file]["type_annotations_undefined"] = annotations_undefined

        # For now, "correct" means no errors, original_untyped is the same as
        # output_untyped, and at least one annotation or definition was added
        file_results[file]["correct"] = (
            len(res["errors"]) == 0
            and original_untyped == file_untyped
            and (annotations_added + definitions_added) > 0
        )

    return file_results


def _evaluate_completion(
    p_idx: int,
    c_idx: int,
    name: str,
    original: str,
    original_untyped: str,
    completion: dict[str, Any],
    tokenizer: Tokenizer,
    tmpdir: Optional[str] = None,
) -> tuple[int, int, dict[str, Any]]:
    try:
        if completion["error"]:
            return p_idx, c_idx, completion

        output = completion["output"]
        output_untyped = transform.delete_types(output)

        completion["token_count"] = len(tokenizer(output))
        completion["accuracy"] = _accuracy(tokenizer.tokenizer, original, output)
        completion["levenshtein"] = _levenshtein(original, output)
        completion["untyped_levenshtein"] = _levenshtein(
            original_untyped, output_untyped
        )

        completion["parses"] = transform.is_valid_syntax(output)
        type_checks, tsc_logs = _tsc(output, tmpdir)
        completion["type_checks"] = type_checks
        completion["tsc_logs"] = tsc_logs

        # Get per-file results
        files_results = _evaluate_files(name, original, output, tsc_logs)
        completion["files_results"] = _unzip_files_results(files_results)

        return p_idx, c_idx, completion
    except:
        print(f"!!! ERROR with problem {p_idx}, completion {c_idx} !!!")
        raise


def run_evaluation(config: Config, args: argparse.Namespace):
    results_path = config.infer_output_path(args.results_directory)
    dataset = util.load_dataset(results_path)

    model_path = util.get_model_path(config.model_name, args.models_directory)
    tokenizer = Tokenizer(model_path)

    # If we already processed this, early return
    eval_output = config.eval_output_path(args.results_directory)
    if Path(eval_output).exists():
        print(f"Skipping because evaluation output {eval_output} already exists\n")
        return

    # We can't update the dataset directly, so save the results in a map
    results: list[dict[int, Any]] = [{} for _ in range(len(dataset))]

    # Type checking may require type declarations, so set up a temporary directory
    # and extract the type declarations (if they exist)
    with tempfile.TemporaryDirectory() as tmpdir:
        if config.dataset_config.type_decls:
            with tarfile.open(config.dataset_config.type_decls) as tar:
                tar.extractall(tmpdir)

        # Set up a process pool so we can give each completion to a process
        with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            fs = [
                executor.submit(
                    _evaluate_completion,
                    p_idx,
                    c_idx,
                    d["name"],
                    d["content"],
                    d["content_without_types"],
                    c,
                    tokenizer,
                    tmpdir,
                )
                for p_idx, d in enumerate(dataset)
                for c_idx, c in enumerate(d["results"])
            ]

            for f in tqdm(
                futures.as_completed(fs),
                desc="Evaluating results",
                total=len(fs),
                miniters=1,
            ):
                p_idx, c_idx, result = f.result()
                results[p_idx][c_idx] = result

    # Now write the results to the dataset
    results_list = [[r[k] for k in sorted(r.keys())] for r in results]
    dataset = dataset.remove_columns("results").add_column(
        name="results", column=results_list
    )

    # Save dataset
    util.save_dataset(dataset, eval_output, args.workers)
    print()


def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Parameters:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    A numerically stable script for calculating an unbiased estimate of pass@k.
    Taken from Fig. 3 of the OpenAI Codex paper, "Evaluating Large Language
    Models Trained on Code," https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _summarize_example(example: dict[str, Any]) -> dict[str, Any]:
    results = example["results"]

    num_completions = len(results)
    num_type_checks = len([r for r in results if r["type_checks"] if not r["error"]])
    pct_type_checks = 0 if num_completions == 0 else num_type_checks / num_completions
    num_correct = len([r for r in results if r["correct"]])
    avg_accuracy = util.mean_or_default(
        [r["accuracy"] for r in results if not r["error"]], default=0
    )
    avg_levenshtein = util.mean_or_default(
        [r["levenshtein"] for r in results if not r["error"]], default=0
    )
    avg_untyped_levenshtein = util.mean_or_default(
        [
            r["untyped_levenshtein"]
            for r in results
            if not r["error"] and r["untyped_levenshtein"]
        ],
        default=0.0,
    )
    avg_type_errors = util.mean_or_default(
        [r["type_errors"] for r in results if not r["error"]]
    )
    avg_parse_errors = util.mean_or_default(
        [r["parse_errors"] for r in results if not r["error"]]
    )
    pass_1 = _pass_at_k(num_completions, num_correct, 1)

    example["num_completions"] = num_completions
    example["num_type_checks"] = num_type_checks
    example["pct_type_checks"] = pct_type_checks
    example["avg_accuracy"] = avg_accuracy
    example["avg_levenshtein"] = avg_levenshtein
    example["avg_untyped_levenshtein"] = avg_untyped_levenshtein
    example["avg_type_errors"] = avg_type_errors
    example["avg_parse_errors"] = avg_parse_errors
    example["pass@1"] = pass_1

    return example


def _summarize_dataset(
    config: Config, args: argparse.Namespace
) -> Optional[dict[str, Any]]:
    results_path = config.eval_output_path(args.results_directory)
    dataset = util.load_dataset(results_path)

    # If we haven't processed this, print an error
    if all(len(c.keys()) <= 2 for r in dataset["results"] for c in r):
        print(f"Skipping {results_path} because it has not been evaluated!\n")
        return None

    # Summarize each example
    dataset = dataset.map(
        _summarize_example, num_proc=args.workers, desc="Summarizing results"
    )
    summary_output = config.summary_output_path(args.results_directory)
    util.save_dataset(dataset, summary_output, args.workers)

    # Summarize dataset
    total_completions = len([r for d in dataset for r in d["results"]])
    total_errors = len([r for d in dataset for r in d["results"] if r["error"]])
    total_type_checks = np.sum(dataset["num_type_checks"])
    pct_errors = 0 if total_completions == 0 else total_errors / total_completions
    pct_type_checks = (
        0 if total_completions == 0 else total_type_checks / total_completions
    )
    avg_accuracy = util.mean_or_default(
        [r["accuracy"] for d in dataset for r in d["results"] if not r["error"]]
    )
    avg_levenshtein = util.mean_or_default(
        [r["levenshtein"] for d in dataset for r in d["results"] if not r["error"]]
    )
    avg_untyped_levenshtein = util.mean_or_default(
        [
            r["untyped_levenshtein"]
            for d in dataset
            for r in d["results"]
            if not r["error"] and r["untyped_levenshtein"]
        ]
    )
    avg_type_errors = util.mean_or_default(
        [r["type_errors"] for d in dataset for r in d["results"] if not r["error"]]
    )
    avg_parse_errors = util.mean_or_default(
        [r["parse_errors"] for d in dataset for r in d["results"] if not r["error"]]
    )
    pass_1 = util.mean_or_default(dataset["pass@1"])

    return {
        "model": config.model_name,
        "num_problems": len(dataset),
        "total_completions": total_completions,
        "total_errors": total_errors,
        "total_type_checks": total_type_checks,
        "pct_errors": pct_errors,
        "pct_type_checks": pct_type_checks,
        "avg_accuracy": avg_accuracy,
        "avg_levenshtein": avg_levenshtein,
        "avg_untyped_levenshtein": avg_untyped_levenshtein,
        "avg_type_errors": avg_type_errors,
        "avg_parse_errors": avg_parse_errors,
        "pass@1": pass_1,
    }


def summarize_results(configs: list[Config], args: argparse.Namespace):
    summaries = []
    for config in configs:
        # Note: _summarize_dataset adds columns to the dataset and writes to disk
        summary = _summarize_dataset(config, args)
        if not summary:
            continue
        summaries.append(summary)

        # Print the summary
        print(f"===Stats for model {config.model_name}===")
        print(f"Number of problems: {summary['num_problems']}")
        print(f"Total completions: {summary['total_completions']}")
        print(f"Errors: {summary['total_errors']} " f"({summary['pct_errors']:.1%})")
        print(
            f"Type checks: {summary['total_type_checks']} "
            f"({summary['pct_type_checks']:.1%})"
        )
        print(f"Accuracy: {summary['avg_accuracy']:.1%}")
        print(f"Levenshtein: {summary['avg_levenshtein']:.1%}")
        print(
            "Untyped Levenshtein (for files that type check): "
            f"{summary['avg_untyped_levenshtein']:.1%}"
        )
        print(f"Type errors: {summary['avg_type_errors']:.1f}")
        print(f"Parse errors: {summary['avg_parse_errors']:.1f}")
        print(f"pass@1 (type checking): {summary['pass@1']:.1%}")
        print()

    # Write to jsonl file
    path = Path(args.results_directory, "summary.jsonl")
    util.write_jsonl(path, summaries, util.NpEncoder)
