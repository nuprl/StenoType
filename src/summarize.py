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


def _mean(
    results: list[dict[str, Any]], field: str, default: float = float("nan")
) -> Optional[float]:
    if values := [r[field] for r in results if not np.isnan(r[field])]:
        # Non-empty list
        return np.mean(values)
    else:
        return default


def _sum(results: list[dict[str, Any]], field: str) -> int:
    return np.sum([r[field] for r in results if not np.isnan(r[field])])


def _count(results: list[dict[str, Any]], field: str) -> int:
    return len([0 for r in results if r[field]])


def _div(numerator: int, denominator: int, default: float = float("nan")) -> float:
    if denominator != 0:
        return numerator / denominator
    else:
        return default


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
    completion["num_correct_files"] = _count(file_results, "correct")
    completion["num_errorfree_files"] = len(
        [0 for file in file_results if not file["errors"]]
    )

    completion["pct_annotations_trivial"] = _div(
        completion["num_annotations_trivial"], completion["num_annotations_added"]
    )

    completion["num_annotations_trivial_correct_files"] = int(
        np.sum([f["num_annotations_trivial"] for f in file_results if f["correct"]])
    )
    completion["num_annotations_added_correct_files"] = int(
        np.sum([f["num_annotations_added"] for f in file_results if f["correct"]])
    )
    completion["pct_annotations_trivial_correct_files"] = _div(
        completion["num_annotations_trivial_correct_files"],
        completion["num_annotations_added_correct_files"],
    )

    completion["pct_annotation_sites_filled"] = _div(
        completion["num_annotations_added"], completion["num_annotation_sites"]
    )

    completion["errors_per_file"] = _div(
        completion["num_errors"], completion["num_files"]
    )

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
        count = _count(results, f)
        summary[f"num_{f}"] = count
        summary[f"pct_{f}"] = _div(count, num_completions)

    total_files = _sum(results, "num_files")
    total_correct_files = _sum(results, "num_correct_files")
    total_errorfree_files = _sum(results, "num_errorfree_files")
    total_errors = _sum(results, "num_errors")
    summary["tot_files"] = total_files
    summary["tot_correct_files"] = total_correct_files
    summary["tot_errorfree_files"] = total_errorfree_files
    summary["tot_errors"] = total_errors
    summary["errors_per_file"] = _div(total_errors, total_files)
    summary["pct_errorfree_files"] = _div(total_errorfree_files, total_files)

    summary["pass@1_project"] = _pass_at_k(num_completions, summary["num_correct"], 1)
    summary["pass@1_files"] = _pass_at_k(total_files, total_correct_files, 1)

    avg_fields = [
        "accuracy",
        "levenshtein",
        "untyped_levenshtein",
        "token_count",
    ]
    for f in avg_fields:
        summary[f"avg_{f}"] = _mean(results, f)

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
        summary[f.replace("num", "avg")] = _mean(results, f)

    summary["num_annotation_sites"] = _sum(results, "num_annotation_sites")
    summary["num_annotations_trivial"] = _sum(results, "num_annotations_trivial")
    summary["num_annotations_added"] = _sum(results, "num_annotations_added")
    summary["pct_annotations_trivial"] = _div(
        summary["num_annotations_trivial"], summary["num_annotations_added"]
    )

    summary["num_annotations_trivial_correct_files"] = _sum(
        results, "num_annotations_trivial_correct_files"
    )
    summary["num_annotations_added_correct_files"] = _sum(
        results, "num_annotations_added_correct_files"
    )
    summary["pct_annotations_trivial_correct_files"] = _div(
        summary["num_annotations_trivial_correct_files"],
        summary["num_annotations_added_correct_files"],
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
    summaries = dataset["summaries"]

    dataset_summary["tot_completions"] = _sum(summaries, "num_completions")

    count_fields = ["num_correct", "num_parses", "num_type_checks"]
    for f in count_fields:
        total = _sum(summaries, f)
        dataset_summary[f.replace("num", "tot")] = total
        dataset_summary[f.replace("num", "pct")] = (
            total / dataset_summary["tot_completions"]
        )
    dataset_summary["pass@1_project"] = _pass_at_k(
        dataset_summary["tot_completions"], dataset_summary["tot_correct"], 1
    )

    count_error_fields = ["tot_files", "tot_errorfree_files", "tot_errors"]
    for f in count_error_fields:
        dataset_summary[f] = _sum(summaries, f)

    total_files = _sum(summaries, "tot_files")
    total_correct_files = _sum(summaries, "tot_correct_files")
    total_errorfree_files = _sum(summaries, "tot_errorfree_files")
    total_errors = _sum(summaries, "tot_errors")
    dataset_summary["tot_files"] = total_files
    dataset_summary["tot_errorfree_files"] = total_errorfree_files
    dataset_summary["tot_errors"] = total_errors
    dataset_summary["errors_per_file"] = _div(total_errors, total_files)
    dataset_summary["pct_errorfree_files"] = _div(total_errorfree_files, total_files)
    dataset_summary["pass@1_files"] = _pass_at_k(total_files, total_correct_files, 1)

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
            [s[f] for s in summaries],
            weights=[s["num_completions"] for s in summaries],
        )

    total_annotations_trivial = _sum(summaries, "num_annotations_trivial")
    total_annotations_added = _sum(summaries, "num_annotations_added")
    dataset_summary["pct_annotations_trivial"] = _div(
        total_annotations_trivial, total_annotations_added
    )

    tot_ann_trivial_correct = _sum(summaries, "num_annotations_trivial_correct_files")
    tot_ann_added_correct = _sum(summaries, "num_annotations_added_correct_files")
    dataset_summary["pct_annotations_trivial_correct_files"] = _div(
        tot_ann_trivial_correct, tot_ann_added_correct
    )

    total_annotation_sites = _sum(summaries, "num_annotation_sites")
    dataset_summary["pct_annotation_sites_filled"] = _div(
        total_annotations_added,
        total_annotation_sites,
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

        print(f"pass@1 (project): {summary['pass@1_project']:.1%}")
        print(f"pass@1 (files): {summary['pass@1_files']:.1%}")
        print()

    jsonl_path = Path(args.results_directory, "summary.jsonl")
    util.write_jsonl(jsonl_path, summaries, util.NpEncoder)

    csv_path = Path(args.results_directory, "summary.csv")
    util.write_csv(csv_path, summaries)
