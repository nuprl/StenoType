from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from subprocess import DEVNULL, PIPE
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Any, Optional
import argparse
import evaluate
import json
import Levenshtein
import numpy as np
import subprocess

from model import Tokenizer
from experiment import ExperimentConfig
from util import ROOT_DIR, transform
import util

ACCURACY_METRIC = evaluate.load("accuracy")


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


def _levenshtein(original: str, output: str) -> float:
    return Levenshtein.ratio(original, output)


def _typescript(contents: str) -> Optional[tuple[int, int]]:
    args = ["node", str(Path(ROOT_DIR, "ts", "main.js").resolve())]
    result = subprocess.run(
        args, input=contents, stdout=PIPE, stderr=DEVNULL, encoding="utf-8"
    )
    if len(result.stdout) > 0:
        data = json.loads(result.stdout)
        if data:
            return data["type_errors"], data["parse_errors"]

    return None


def _evaluate_completion(
    p_idx: int,
    c_idx: int,
    original: str,
    completion: dict[str, Any],
    tokenizer: Tokenizer,
) -> tuple[int, int, dict[str, Any]]:
    if completion["error"]:
        return p_idx, c_idx, completion

    output = completion["output"]

    completion["token_count"] = len(tokenizer(output))
    completion["accuracy"] = _accuracy(tokenizer.tokenizer, original, output)
    completion["levenshtein"] = _levenshtein(original, output)

    res = _typescript(output)
    if res:
        te, pe = res
        completion["type_errors"] = te
        completion["parse_errors"] = pe
        completion["type_checks"] = te == 0 and pe == 0

        # Only compute untyped Levenshtein if the output type checks
        if completion["type_checks"]:
            original_untyped = transform.delete_types(original)
            output_untyped = transform.delete_types(output)
            completion["untyped_levenshtein"] = _levenshtein(
                original_untyped, output_untyped
            )
        else:
            completion["untyped_levenshtein"] = None
    else:
        completion["error"] = True

    return p_idx, c_idx, completion


def run_evaluation(config: ExperimentConfig, args: argparse.Namespace):
    # For now, the output name is {model_name}.parquet. Later we might have
    # different experiments for a model, so we will need different names.
    results_path = util.get_results_name(config.model_name, args.results_directory)
    dataset = util.load_dataset(results_path)

    model_path = util.get_model_path(config.model_name, args.models_directory)
    tokenizer = Tokenizer(model_path)

    # If we already processed this, early return
    if all(len(c.keys()) > 2 for r in dataset["results"] for c in r):
        print(f"Skipping {results_path} because it was already processed\n")
        return

    # Set up a process pool so we can give each completion to a process
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        fs = [
            executor.submit(
                _evaluate_completion, p_idx, c_idx, d["content"], c, tokenizer
            )
            for p_idx, d in enumerate(dataset)
            for c_idx, c in enumerate(d["results"])
        ]

        # We can't update the dataset directly, so save the results in a map
        results: list[dict[int, Any]] = [{} for _ in range(len(dataset))]
        for i, f in enumerate(tqdm(fs, desc="Evaluating results", miniters=1)):
            p_idx, c_idx, result = f.result()
            results[p_idx][c_idx] = result

        # Now write the results to the dataset
        results_list = [[r[k] for k in sorted(r.keys())] for r in results]
        dataset = dataset.remove_columns("results").add_column(
            name="results", column=results_list
        )

    # Save dataset
    util.save_dataset(dataset, results_path, args.workers)
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
    avg_accuracy = np.mean([r["accuracy"] for r in results if not r["error"]])
    avg_levenshtein = np.mean([r["levenshtein"] for r in results if not r["error"]])
    avg_untyped_levenshtein = np.mean(
        [
            r["untyped_levenshtein"]
            for r in results
            if not r["error"] and r["untyped_levenshtein"]
        ]
    )
    avg_type_errors = np.mean([r["type_errors"] for r in results if not r["error"]])
    avg_parse_errors = np.mean([r["parse_errors"] for r in results if not r["error"]])
    pass_1 = _pass_at_k(num_completions, num_type_checks, 1)

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
    config: ExperimentConfig, args: argparse.Namespace
) -> Optional[dict[str, Any]]:
    results_path = util.get_results_name(config.model_name, args.results_directory)
    dataset = util.load_dataset(results_path)

    # If we haven't processed this, print an error
    if all(len(c.keys()) <= 2 for r in dataset["results"] for c in r):
        print(f"Skipping {results_path} because it has not been evaluated!\n")
        return None

    # Summarize each example
    dataset = dataset.map(
        _summarize_example, num_proc=args.workers, desc="Summarizing results"
    )
    util.save_dataset(dataset, results_path, args.workers)

    # Summarize dataset
    total_completions = len([r for d in dataset for r in d["results"]])
    total_errors = len([r for d in dataset for r in d["results"] if r["error"]])
    total_type_checks = np.sum(dataset["num_type_checks"])
    pct_errors = 0 if total_completions == 0 else total_errors / total_completions
    pct_type_checks = (
        0 if total_completions == 0 else total_type_checks / total_completions
    )
    avg_accuracy = np.mean(
        [r["accuracy"] for d in dataset for r in d["results"] if not r["error"]]
    )
    avg_levenshtein = np.mean(
        [r["levenshtein"] for d in dataset for r in d["results"] if not r["error"]]
    )
    avg_untyped_levenshtein = np.mean(
        [
            r["untyped_levenshtein"]
            for d in dataset
            for r in d["results"]
            if not r["error"] and r["untyped_levenshtein"]
        ]
    )
    avg_type_errors = np.mean(
        [r["type_errors"] for d in dataset for r in d["results"] if not r["error"]]
    )
    avg_parse_errors = np.mean(
        [r["parse_errors"] for d in dataset for r in d["results"] if not r["error"]]
    )
    pass_1 = np.mean(dataset["pass@1"])

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


def summarize_results(configs: list[ExperimentConfig], args: argparse.Namespace):
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
