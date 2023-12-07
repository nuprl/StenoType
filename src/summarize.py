from concurrent import futures
from pathlib import Path
from tqdm import tqdm
from typing import Any
import argparse
import numpy as np

from inference import Config
from util import transform
import util


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


def _summarize_example(idx: int, example: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    results = example["results"]
    for i, completion in enumerate(results):
        results[i] = _summarize_completion(completion, example["content_without_types"])

    # TODO:
    # next iterate over problems
    # need to calculate problem-level results (by aggregating completion stats)

    return idx, results


def _summarize_dataset(config: Config, args: argparse.Namespace) -> dict[str, Any]:
    eval_output = config.eval_output_path(args.results_directory)
    if not Path(eval_output).exists():
        print(f"Error: results file does not exist: {eval_output}")
        exit(1)
    dataset = util.load_dataset(eval_output)

    # We can't update the dataset directly, so save the results in a list
    results: list[dict[str, Any]] = [{} for _ in range(len(dataset))]

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
            idx, result = f.result()
            results[idx] = result

    # Now write the results to the dataset
    dataset = dataset.remove_columns("results").add_column(
        name="results", column=results
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
