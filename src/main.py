from datasets import Dataset, IterableDataset
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
    cpu_count = os.cpu_count()

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
        required=True,
        help="Path to model to load")
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port for the model server")
    parser.add_argument(
        "--devices",
        type=str,
        required=True,
        help="GPU devices to use")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use; defaults to {cpu_count}")

    args = parser.parse_args()
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
    args: argparse.Namespace,
    model: Model
) -> Dataset | IterableDataset:
    # Remove type annotations and definitions, add as new column
    dataset = dataset.map(
        lambda e: add_column_without_types(e, args.content_column),
        num_proc=args.workers,
        desc="Removing types"
    )

    # Remove empty rows (since removing types may end up removing everything)
    dataset = dataset.filter(
        lambda e: not util.is_empty(e[COLUMN_WITHOUT_TYPES]),
        num_proc=args.workers,
        desc="Removing empty examples"
    )

    # Remove examples that are too long
    dataset = dataset.filter(
        lambda e: (len(model.tokenize(e[COLUMN_WITHOUT_TYPES])) < model.INPUT_SIZE),
        num_proc=args.workers,
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

def evaluate_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    original_column: str,
) -> dict[str, Any]:
    original = example[original_column]
    output = example[OUTPUT_COLUMN]

    example["accuracy"] = evaluation.accuracy(tokenizer, original, output)
    example["levenshtein"] = evaluation.levenshtein(original, output)

    return example

def run_evaluation(
    dataset: Dataset | IterableDataset,
    args: argparse.Namespace,
    model: Model,
    num_examples: int,
    num_removed: int
) -> None:
    # Remove examples that had errors
    num_runs = len(dataset)
    dataset = dataset.filter(
        lambda e: not e[ERROR_COLUMN],
        num_proc=args.workers,
        desc="Removing failed runs"
    )
    num_errors = num_runs - len(dataset)

    # TODO: tsc server
    dataset = dataset.map(
        lambda e: evaluate_example(
            e,
            model.tokenizer,
            args.content_column
        ),
        num_proc=args.workers,
        desc="Evaluating results"
    )

    # Print results statistics
    print("Number of examples in the original:", num_examples)
    print("Number of examples skipped:", num_removed)
    print("Number of examples failed:", num_errors)
    results = pd.DataFrame({
        "accuracy": dataset["accuracy"],
        "levenshtein": dataset["levenshtein"]
    })
    print(results.describe())

def main():
    args = parse_args()

    model = Model(args.model, args.port, args.devices)
    typeinf = TypeInference(model)
    dataset = util.load_dataset(args.dataset, args.split, args.revision, args.workers)

    with util.timer():
        num_examples = len(dataset)
        dataset = prepare_dataset(dataset, args, model)
        num_removed = num_examples - len(dataset)

        # Run experiments
        dataset = dataset.map(
            lambda e: infer_on_example(e, typeinf, COLUMN_WITHOUT_TYPES),
            num_proc=args.workers, desc="Inferring types"
        )

        run_evaluation(dataset, args, model, num_examples, num_removed)

    # Save result dataset to disk
    util.save_dataset(dataset, args.output, args.workers)

    # TODO:
    # get rid of argument parsing, just run experiments by running script

if __name__ == "__main__":
    main()
