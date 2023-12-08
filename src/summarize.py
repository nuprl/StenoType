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


def _count_pct(
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
        completion[f] = _sum(file_results, f)

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

    completion["pct_annotations_trivial"] = (
        0
        if completion["num_annotations_added"] == 0
        else (
            completion["num_annotations_trivial"] / completion["num_annotations_added"]
        )
    )

    tot_ann_trivial_errorfree = int(
        np.sum(
            [f["num_annotations_trivial"] for f in file_results if f["num_errors"] == 0]
        )
    )
    tot_ann_added_errorfree = int(
        np.sum(
            [f["num_annotations_added"] for f in file_results if f["num_errors"] == 0]
        )
    )
    completion["num_annotations_trivial_errorfree_files"] = tot_ann_trivial_errorfree
    completion["num_annotations_added_errorfree_files"] = tot_ann_added_errorfree
    completion["pct_annotations_trivial_errorfree_files"] = (
        0
        if tot_ann_added_errorfree == 0
        else (tot_ann_trivial_errorfree / tot_ann_added_errorfree)
    )

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

    summary: dict[str, Any] = {}
    num_completions = len(results)
    summary["num_completions"] = num_completions

    count_fields = ["parses", "type_checks", "correct"]
    for f in count_fields:
        summary[f"num_{f}"], summary[f"pct_{f}"] = _count_pct(
            results, f, num_completions
        )
    # Can't put this in the loop because we need to rename error -> failed
    summary["num_failed"], summary["pct_failed"] = _count_pct(
        results, "error", num_completions, skip_error=False
    )
    summary["pass@1_correct"] = _pass_at_k(num_completions, summary["num_correct"], 1)

    total_files = _sum(results, "num_files")
    total_errorfree_files = _sum(results, "num_errorfree_files")
    summary["avg_errors"] = _mean_or_default(results, "num_errors", default=0.0)
    summary["errors_per_file"] = _sum(results, "num_errors") / total_files
    summary["pct_errorfree_files"] = total_errorfree_files / total_files
    summary["pass@1_errorfree"] = _pass_at_k(total_files, total_errorfree_files, 1)

    avg_fields = [
        "accuracy",
        "levenshtein",
        "untyped_levenshtein",
        "token_count",
    ]
    for f in avg_fields:
        summary[f"avg_{f}"] = _mean_or_default(results, f, default=0.0)

    avg_fields_num = [
        "num_annotation_sites",
        "num_annotations_added",
        "num_annotations_trivial",
        "num_definitions_added",
        "num_definitions_used",
        "num_types_undefined",
    ]
    for f in avg_fields_num:
        summary[f.replace("num", "avg")] = _mean_or_default(results, f, default=0.0)

    total_annotations_trivial = _sum(results, "num_annotations_trivial")
    total_annotations_added = _sum(results, "num_annotations_added")
    summary["num_annotations_trivial"] = total_annotations_trivial
    summary["num_annotations_added"] = total_annotations_added
    summary["pct_annotations_trivial"] = (
        0
        if total_annotations_added == 0
        else (total_annotations_trivial / total_annotations_added)
    )

    tot_ann_trivial_errorfree = _sum(results, "num_annotations_trivial_errorfree_files")
    tot_ann_added_errorfree = _sum(results, "num_annotations_added_errorfree_files")
    summary["num_annotations_trivial_errorfree_files"] = tot_ann_trivial_errorfree
    summary["num_annotations_added_errorfree_files"] = tot_ann_added_errorfree
    summary["pct_annotations_trivial_errorfree_files"] = (
        0
        if tot_ann_added_errorfree == 0
        else (tot_ann_trivial_errorfree / tot_ann_added_errorfree)
    )

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
