from datasets import Dataset, IterableDataset
from functools import partial
from pathlib import Path
from typing import Any
import argparse
import os

from evaluation import run_evaluation
from model import Model
from type_inference import TypeInference
from util import transform
import util

COLUMN_WITHOUT_TYPES = "content_without_types"
OUTPUT_COLUMN = "output"
ERROR_COLUMN = "error"

def parse_args() -> argparse.Namespace:
    # os.cpu_count() is the number of CPUs on the system,
    # not the number available to the current process
    cpu_count = len(os.sched_getaffinity(0))

    parser = argparse.ArgumentParser(
        description="Runs StarCoder to infer types for JavaScript")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="load the input dataset from a .parquet file, .jsonl file, or local "
             "Hugging Face dataset; otherwise tries to load from the Hugging Face Hub")
    parser.add_argument(
        "--output",
        type=str,
        help="path to output results, can be a .parquet file, .jsonl file, or local "
             "directory")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset revision, if loading from the Hub; defaults to 'main'")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split, if loading from the Hub; defaults to 'train'")
    parser.add_argument(
        "--content-column",
        type=str,
        default="content",
        help="Column with the file contents; defaults to 'content'")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model to load")
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port for the model server")
    parser.add_argument(
        "--devices",
        type=str,
        help="GPU devices to use")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use; defaults to {cpu_count}")
    parser.add_argument(
        "--skim",
        action="store_true",
        help="browse through the dataset, one example at a time")

    args = parser.parse_args()

    if not args.skim:
        if not args.model or not args.devices:
            print("error: the following arguments are required: --model, --devices")
            exit(2)

    output = args.output
    if output:
        if Path(output).exists():
            print(f"Output path {output} already exists, please delete, rename, or "
                  "choose a different output path!")
            exit(2)
        elif not (output.endswith(".parquet") or output.endswith(".jsonl")):
            Path(output).mkdir(parents=True, exist_ok=True)

    return args

def add_column_without_types(example: dict[str, Any], column: str) -> dict[str, Any]:
    # Delete type annotations and definitions
    content = example[column]
    stripped = transform.delete_types(content)
    example[COLUMN_WITHOUT_TYPES] = stripped
    return example

def prepare_dataset(
    dataset: Dataset | IterableDataset,
    model: Model,
    content_column: str,
    workers: int
) -> Dataset | IterableDataset:
    # Remove type annotations and definitions, add as new column
    dataset = dataset.map(
        partial(add_column_without_types, column=content_column),
        num_proc=workers,
        desc="Removing types"
    )

    # Remove empty rows (since removing types may end up removing everything)
    dataset = dataset.filter(
        lambda e: not transform.is_empty(e[COLUMN_WITHOUT_TYPES]),
        num_proc=workers,
        desc="Removing empty examples"
    )

    # Remove examples that are too long
    dataset = dataset.filter(
        lambda e: (len(model.tokenize(e[COLUMN_WITHOUT_TYPES])) < model.INPUT_SIZE),
        num_proc=workers,
        desc="Removing large examples"
    )

    return dataset

def infer_on_example(
    example: dict[str, Any],
    typeinf: TypeInference
) -> dict[str, Any]:
    # For now, we're assuming TypeScript with type annotations and definitions removed.
    stripped = example[COLUMN_WITHOUT_TYPES]

    # Run type inference
    output = typeinf.infer_with_definitions(stripped)
    if output:
        example[OUTPUT_COLUMN] = output
        example[ERROR_COLUMN] = False
    else:
        example[OUTPUT_COLUMN] = ""
        example[ERROR_COLUMN] = True

    return example

def run_inference(
    dataset: Dataset | IterableDataset,
    model: Model,
    typeinf: TypeInference,
    content_column: str,
    workers: int
) -> Dataset | IterableDataset:
    # Inference is the bottleneck, so too many workers will slow things down
    inference_workers = min(workers, 8)

    dataset = prepare_dataset(dataset, model, content_column, workers)
    with util.timer():
        dataset = dataset.map(
            partial(infer_on_example, typeinf=typeinf),
            num_proc=inference_workers, desc="Inferring types"
        )
    dataset = dataset.select_columns([
        "hexsha",
        "max_stars_repo_path",
        "max_stars_repo_name",
        "content",
        COLUMN_WITHOUT_TYPES,
        OUTPUT_COLUMN,
        ERROR_COLUMN
    ])

    return dataset

def main():
    args = parse_args()
    dataset = util.load_dataset(args.dataset, args.split, args.revision, args.workers)

    if args.skim:
        for i, d in enumerate(dataset):
            print("===REPO===")
            print(i, d["max_stars_repo_name"], d["max_stars_repo_path"])
            print("===ORIGINAL===")
            print(d["content"])
            print("===INPUT===")
            print(d[COLUMN_WITHOUT_TYPES])
            print("===OUTPUT===")
            print(d[OUTPUT_COLUMN])
            print("===RESULTS===")
            print(f"Accuracy {d['accuracy']:.2%}\n"
                  f"Levenshtein {d['levenshtein']:.2%}\n"
                  f"Type errors {d['type_errors']}\n"
                  f"Parse errors {d['parse_errors']}")
            input("===EOF===")
        return

    model = Model(args.model, args.port, args.devices)
    tokenizer = model.tokenizer
    typeinf = TypeInference(model)

    # Run inference
    num_examples = len(dataset)
    dataset = run_inference(
        dataset, model, typeinf, args.content_column, args.workers
    )
    num_removed = num_examples - len(dataset)

    # Run evaluation
    dataset = run_evaluation(
        dataset,
        tokenizer,
        num_examples,
        num_removed,
        args.content_column,
        OUTPUT_COLUMN,
        ERROR_COLUMN,
        args.workers
    )

    # Save results to disk, if output was provided
    util.save_dataset(dataset, args.output, args.workers)

    # TODO:
    # get rid of argument parsing, just run experiments by running script

if __name__ == "__main__":
    main()
