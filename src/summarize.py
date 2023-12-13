from concurrent import futures
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional
import argparse
import numpy as np

from inference import Config
from util import transform
import util

# TODO: A better approach would have been to flatten the datasets, one result
# (note: a "result" is a single file of a completion of a problem of a dataset)
# per row of a data frame, and then aggregate using pandas/R.

# TODO: can maybe use some more helpers, but be careful about mixing
# completion level, problem level, and dataset level

# TODO: Should division by zero return 0 or None or nan? Need to handle these
# cases, and also cases where data is missing


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
    total_errors = _sum(results, "num_errors")
    summary["tot_files"] = total_files
    summary["tot_errorfree_files"] = total_errorfree_files
    summary["tot_errors"] = total_errors
    summary["errors_per_file"] = total_errors / total_files
    summary["pct_errorfree_files"] = total_errorfree_files / total_files

    # TODO: errorfree isn't enough, we want types to have been added
    # Probably want to distinguish project-level correct vs file-level correct
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
        "num_errors",
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
    summary["num_annotation_sites"] = _sum(results, "num_annotation_sites")
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

        for future in tqdm(
            futures.as_completed(fs),
            desc="Summarizing results",
            total=len(fs),
            miniters=1,
        ):
            idx, result, summary = future.result()
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

    dataset_summary: dict[str, Any] = {
        "model": config.filename,
        "num_problems": len(dataset),
    }

    tot_completions = np.sum([s["num_completions"] for s in dataset["summaries"]])
    dataset_summary["tot_completions"] = tot_completions

    count_fields = ["num_correct", "num_failed", "num_parses", "num_type_checks"]
    for f in count_fields:
        total = np.sum([s[f] for s in dataset["summaries"]])
        dataset_summary[f.replace("num", "tot")] = total
        dataset_summary[f.replace("num", "pct")] = total / tot_completions
    dataset_summary["pass@1_correct"] = _pass_at_k(
        tot_completions, dataset_summary["tot_correct"], 1
    )

    count_error_fields = ["tot_files", "tot_errorfree_files", "tot_errors"]
    for f in count_error_fields:
        dataset_summary[f] = np.sum([s[f] for s in dataset["summaries"]])

    total_files = np.sum([s["tot_files"] for s in dataset["summaries"]])
    total_errorfree_files = np.sum(
        [s["tot_errorfree_files"] for s in dataset["summaries"]]
    )
    total_errors = np.sum([s["tot_errors"] for s in dataset["summaries"]])
    dataset_summary["tot_files"] = total_files
    dataset_summary["tot_errorfree_files"] = total_errorfree_files
    dataset_summary["tot_errors"] = total_errors
    dataset_summary["errors_per_file"] = total_errors / total_files
    dataset_summary["pct_errorfree_files"] = total_errorfree_files / total_files
    dataset_summary["pass@1_errorfree"] = _pass_at_k(
        total_files, total_errorfree_files, 1
    )

    avg_fields = [
        "avg_annotation_sites",
        "avg_annotations_added",
        "avg_annotations_trivial",
        "avg_definitions_added",
        "avg_definitions_used",
        "avg_types_undefined",
        "avg_errors",
        "avg_accuracy",
        "avg_levenshtein",
        "avg_untyped_levenshtein",
        "avg_token_count",
    ]
    for f in avg_fields:
        dataset_summary[f] = np.average(
            [s[f] for s in dataset["summaries"]],
            weights=[s["num_completions"] for s in dataset["summaries"]],
        )

    total_annotations_trivial = np.sum(
        [s["num_annotations_trivial"] for s in dataset["summaries"]]
    )
    total_annotations_added = np.sum(
        [s["num_annotations_added"] for s in dataset["summaries"]]
    )
    dataset_summary["pct_annotations_trivial"] = (
        0
        if total_annotations_added == 0
        else (total_annotations_trivial / total_annotations_added)
    )

    tot_ann_trivial_errorfree = np.sum(
        [s["num_annotations_trivial_errorfree_files"] for s in dataset["summaries"]]
    )
    tot_ann_added_errorfree = np.sum(
        [s["num_annotations_added_errorfree_files"] for s in dataset["summaries"]]
    )
    dataset_summary["pct_annotations_trivial_errorfree_files"] = (
        0
        if tot_ann_added_errorfree == 0
        else (tot_ann_trivial_errorfree / tot_ann_added_errorfree)
    )

    total_annotation_sites = np.sum(
        [s["num_annotation_sites"] for s in dataset["summaries"]]
    )
    dataset_summary["pct_annotation_sites_filled"] = (
        0
        if total_annotation_sites == 0
        else (total_annotations_added / total_annotation_sites)
    )

    return dataset_summary


def summarize(configs: list[Config], args: argparse.Namespace):
    summaries = []
    for config in configs:
        summary = _summarize_dataset(config, args)
        summaries.append(summary)

        print(f"===Stats for configuration {config.filename}===")
        print(f"Number of problems: {summary['num_problems']}")
        print(f"Total completions: {summary['tot_completions']}")
        print(f"Failed: {summary['tot_failed']} ({summary['pct_failed']:.1%})")

        print(f"Parses: {summary['tot_parses']} ({summary['pct_parses']:.1%})")
        print(
            f"Type checks: {summary['tot_type_checks']} ({summary['pct_type_checks']:.1%})"
        )
        print(f"Errors per problem: {summary['avg_errors']:.1f}")
        print(f"Errors per file: {summary['errors_per_file']:.1f}")

        print(f"Levenshtein: {summary['avg_levenshtein']:.1%}")
        print(f"Untyped Levenshtein: {summary['avg_untyped_levenshtein']:.1%}")

        print(f"Annotations added: {summary['avg_annotations_added']:.1f}")
        print(f"Definitions added: {summary['avg_definitions_added']:.1f}")

        print(f"Annotation sites filled: {summary['pct_annotation_sites_filled']:.1%}")
        print(f"Trivial annotations: {summary['pct_annotations_trivial']:.1%}")
        print(f"Definitions used: {summary['avg_definitions_used']:.1f}")

        print(f"pass@1 (correct): {summary['pass@1_correct']:.1%}")
        print(f"pass@1 (errorfree files): {summary['pass@1_errorfree']:.1%}")
        print()

    jsonl_path = Path(args.results_directory, "summary.jsonl")
    util.write_jsonl(jsonl_path, summaries, util.NpEncoder)

    csv_path = Path(args.results_directory, "summary.csv")
    util.write_csv(csv_path, summaries)
