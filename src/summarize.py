from concurrent import futures
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional
import argparse
import numpy as np

from inference import Config
from util import transform
import util


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


def _mean_or_default(
    results: list[dict[str, Any]], field: str, default: Optional[float] = None
) -> Optional[float]:
    if values := [r[field] for r in results if not r["error"]]:
        # Non-empty list
        return np.mean(values)
    elif default is not None:
        # Empty list and we have a default value
        return default
    else:
        return None


def _count_and_pct(
    results: list[dict[str, Any]], field: str, total: int, skip_error: bool = True
) -> tuple[int, float]:
    if skip_error:
        count = len([r for r in results if r[field] if not r["error"]])
    else:
        count = len([r for r in results if r[field]])
    pct = 0 if total == 0 else count / total

    return count, pct


def _sum(results: list[dict[str, Any]], field: str) -> int:
    return np.sum([r[field] for r in results])


def _summarize_completion(
    completion: dict[str, Any], original_untyped: str
) -> dict[str, Any]:
    file_results = completion["files_results"]["results"]

    number_fields = [
        "num_annotation_sites",
        "num_annotations_added",
        "num_annotations_trivial",
        "num_definitions_added",
        "num_definitions_used",
        "num_errors",
        "num_types_undefined",
    ]
    for f in number_fields:
        completion[f] = np.sum([file[f] for file in file_results])

    set_fields = [
        "type_annotations",
        "type_definitions",
        "type_definitions_used",
        "types_undefined",
    ]
    for f in set_fields:
        completion[f] = set([e for file in file_results for e in file[f]])

    completion["num_files"] = len(file_results)
    completion["num_errorfree_files"] = len(
        [None for file in file_results if not file["errors"]]
    )

    # TODO: pct_annotations_trivial_in_errorfree_files

    completion["errors_per_file"] = completion["num_errors"] / completion["num_files"]

    # For now, "correct" means type checks, original_untyped is the same as
    # output_untyped, and at least one annotation or definition was added
    output_untyped = transform.delete_types(completion["output"], delete_comments=True)
    completion["correct"] = (
        completion["type_checks"]
        and original_untyped == output_untyped
        and (completion["num_annotations_added"] + completion["num_definitions_added"])
        > 0
    )

    return completion


def _summarize_example(
    idx: int, example: dict[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    results = example["results"]
    for i, completion in enumerate(results):
        results[i] = _summarize_completion(completion, example["content_without_types"])

    num_completions = len(results)
    num_parses, pct_parses = _count_and_pct(results, "parses", num_completions)
    num_type_checks, pct_type_checks = _count_and_pct(
        results, "type_checks", num_completions
    )
    num_correct, pct_correct = _count_and_pct(results, "correct", num_completions)
    num_failed, pct_failed = _count_and_pct(
        results, "error", num_completions, skip_error=False
    )
    pass_1_correct = _pass_at_k(num_completions, num_correct, 1)

    total_files = _sum(results, "num_files")
    total_errorfree_files = _sum(results, "num_errorfree_files")

    errors_per_file = _sum(results, "num_errors") / total_files
    pct_errorfree_files = total_errorfree_files / total_files
    pass_1_errorfree = _pass_at_k(total_files, total_errorfree_files, 1)

    avg_accuracy = _mean_or_default(results, "accuracy", default=0.0)
    avg_levenshtein = _mean_or_default(results, "levenshtein", default=0.0)
    avg_untyped_levenshtein = _mean_or_default(
        results, "untyped_levenshtein", default=0.0
    )
    avg_token_count = _mean_or_default(results, "token_count", default=0.0)

    avg_definitions_added = _mean_or_default(
        results, "num_definitions_added", default=0.0
    )
    avg_definitions_used = _mean_or_default(
        results, "num_definitions_used", default=0.0
    )
    avg_types_undefined = _mean_or_default(results, "num_types_undefined", default=0.0)

    pct_annotations_trivial = _sum(results, "num_annotations_trivial") / _sum(
        results, "num_annotation_sites"
    )

    # TODO: pct_annotations_trivial_in_errorfree_files

    summary = {
        "num_completions": num_completions,
        "num_parses": num_parses,
        "pct_parses": pct_parses,
        "num_type_checks": num_type_checks,
        "pct_type_checks": pct_type_checks,
        "num_correct": num_correct,
        "pct_correct": pct_correct,
        "num_failed": num_failed,
        "pct_failed": pct_failed,
        "pass@1_correct": pass_1_correct,
        "errors_per_file": errors_per_file,
        "pct_errorfree_files": pct_errorfree_files,
        "pass@1_errorfree": pass_1_errorfree,
        "avg_accuracy": avg_accuracy,
        "avg_levenshtein": avg_levenshtein,
        "avg_untyped_levenshtein": avg_untyped_levenshtein,
        "avg_token_count": avg_token_count,
        "avg_definitions_added": avg_definitions_added,
        "avg_definitions_used": avg_definitions_used,
        "avg_types_undefined": avg_types_undefined,
        "pct_annotations_trivial": pct_annotations_trivial,
    }

    return idx, results, summary


def _summarize_dataset(config: Config, args: argparse.Namespace) -> dict[str, Any]:
    eval_output = config.eval_output_path(args.results_directory)
    if not Path(eval_output).exists():
        print(f"Error: results file does not exist: {eval_output}")
        exit(1)
    dataset = util.load_dataset(eval_output)

    # We can't update the dataset directly, so save the results/summaries in a list
    results: list[dict[str, Any]] = [{} for _ in range(len(dataset))]
    summaries: list[dict[str, Any]] = [{} for _ in range(len(dataset))]

    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        fs = [
            executor.submit(_summarize_example, idx, d) for idx, d in enumerate(dataset)
        ]

        for f in tqdm(
            futures.as_completed(fs),
            desc="Summarizing results",
            total=len(fs),
            miniters=1,
        ):
            idx, result, summary = f.result()
            results[idx], summaries[idx] = result, summary

    # Now write the results/summaries to the dataset
    dataset = (
        dataset.remove_columns("results")
        .add_column(name="results", column=results)
        .add_column(name="summaries", column=summaries)
    )

    # Don't skip if it already exists: we need to return something and this is
    # relatively cheap to compute
    summary_output = config.summary_output_path(args.results_directory)
    util.save_dataset(dataset, summary_output, args.workers)
    print()

    # TODO: calculate dataset-level results (by aggregating problem stats)
    # Return dictionary of dataset-level results

    return {}


def summarize(configs: list[Config], args: argparse.Namespace):
    summaries = []
    for config in configs:
        summary = _summarize_dataset(config, args)
        summaries.append(summary)

    # TODO: aggregate datasets and summarize everything, output to jsonl
