from datasets import Dataset, IterableDataset
from functools import partial
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import Any
import argparse
import pandas as pd
import os

from model import Model
from type_inference import TypeInference
import evaluation
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
    stripped = util.delete_types(content)
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
        lambda e: not util.is_empty(e[COLUMN_WITHOUT_TYPES]),
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
    typeinf: TypeInference,
    column: str
) -> dict[str, Any]:
    # For now, we're assuming TypeScript with type annotations and definitions removed.
    content = example[column]
    stripped = example[COLUMN_WITHOUT_TYPES]

    # Run type inference
    output = typeinf.infer_with_definitions(stripped)

    # TODO: wanted to drop unneeded columns, but this seems to keep old columns
    result = {
        "hexsha": example["hexsha"],
        "max_stars_repo_path": example["max_stars_repo_path"],
        "max_stars_repo_name": example["max_stars_repo_name"],
        "content": content,
        OUTPUT_COLUMN: "",
        ERROR_COLUMN: True,
    }

    if output:
        result[OUTPUT_COLUMN] = output
        result[ERROR_COLUMN] = False

    return result

def run_experiments(
    dataset: Dataset | IterableDataset,
    model: Model,
    typeinf: TypeInference,
    content_column: str,
    workers: int
) -> Dataset | IterableDataset:
    # Inference is the bottleneck, so too many workers will slow things down
    inference_workers = min(workers, 8)

    with util.timer():
        dataset = prepare_dataset(dataset, model, content_column, workers)
        dataset = dataset.map(
            partial(infer_on_example, typeinf=typeinf, column=COLUMN_WITHOUT_TYPES),
            num_proc=inference_workers, desc="Inferring types"
        )

    return dataset

def evaluate_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    original_column: str,
) -> dict[str, Any]:
    original = example[original_column]
    output = example[OUTPUT_COLUMN]

    example["accuracy"] = evaluation.accuracy(tokenizer, original, output)
    example["levenshtein"] = evaluation.levenshtein(original, output)

    type_errors, parse_errors = evaluation.typescript(output)
    example["type_errors"] = type_errors
    example["parse_errors"] = parse_errors

    return example

def run_evaluation(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizer,
    num_examples: int,
    num_removed: int,
    content_column: str,
    workers: int
) -> Dataset | IterableDataset:
    # Remove examples that had errors
    num_runs = len(dataset)
    dataset = dataset.filter(
        lambda e: not e[ERROR_COLUMN],
        num_proc=workers,
        desc="Removing failed runs"
    )
    num_errors = num_runs - len(dataset)

    dataset = dataset.map(
        partial(
            evaluate_example,
            tokenizer=tokenizer,
            original_column=content_column
        ),
        num_proc=workers,
        desc="Evaluating results"
    )

    num_typechecked = len([d for d in dataset["type_errors"] if d == 0])
    pct_typechecked = num_typechecked / len(dataset)

    # Print result statistics
    print("Number of examples in the original:", num_examples)
    print("Number of examples skipped:", num_removed)
    print("Number of examples failed:", num_errors)
    print("Number of examples that type checked: "
          f"{num_typechecked} ({pct_typechecked:.2%})")
    results = pd.DataFrame({
        "accuracy": dataset["accuracy"],
        "levenshtein": dataset["levenshtein"],
        "type_errors": dataset["type_errors"],
        "parse_errors": dataset["parse_errors"]
    })
    print(results.describe())

    return dataset

def main():
    args = parse_args()
    dataset = util.load_dataset(args.dataset, args.split, args.revision, args.workers)

    if args.skim:
        for d in dataset:
            print("===REPO===")
            print(d["max_stars_repo_name"], d["max_stars_repo_path"])
            print("===INPUT===")
            print(d["content_without_types"])
            print("===OUTPUT===")
            print(d["output"])
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

    # Run experiments
    num_examples = len(dataset)
    dataset = run_experiments(
        dataset, model, typeinf, args.content_column, args.workers
    )
    num_removed = num_examples - len(dataset)

    # Run evaluation
    dataset = run_evaluation(
        dataset, tokenizer, num_examples, num_removed, args.content_column, args.workers
    )

    # Save results to disk, if output was provided
    util.save_dataset(dataset, args.output, args.workers)

    # TODO:
    # get rid of argument parsing, just run experiments by running script

if __name__ == "__main__":
    main()
