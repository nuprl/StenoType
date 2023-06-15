from datasets import (Dataset,              # type: ignore
                      DatasetDict,
                      IterableDataset,
                      IterableDatasetDict)
from pathlib import Path
from typing import Any
import argparse
import datasets
import pandas as pd
import os

from model import Model
from type_inference import TypeInference
import util

COLUMN_WITHOUT_TYPES = "content_without_types"

def parse_args():
    cpu_count = os.cpu_count()

    parser = argparse.ArgumentParser(
        description="Runs StarCoder to infer types for JavaScript")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="load the input dataset from a .parquet file, .jsonl file, or local" +
             "Hugging Face dataset; otherwise tries to load from the Hugging Face Hub")
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
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use; defaults to {cpu_count}")

    return parser.parse_args()

def load_dataset(
    dataset: str,
    split: str,
    revision: str,
    workers: int
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """
    Load a dataset. Tries to interpret dataset as a path and loads a local file
    (in Parquet or JSON Lines format) or directory. If the path does not exist,
    load the dataset from the Hugging Face Hub.
    """
    if Path(dataset).exists():
        print(f"Loading dataset from {dataset} from disk...", flush=True)
        if dataset.endswith(".parquet"):
            return Dataset.from_parquet(dataset)
        elif dataset.endswith(".jsonl"):
            return Dataset.from_json(dataset)
        else:
            return datasets.load_from_disk(dataset)
    else:
        print(f"Loading dataset {dataset} from the Hugging Face Hub...", flush=True)
        return datasets.load_dataset(dataset,
                                     split=split,
                                     revision=revision,
                                     num_proc=workers)

def add_column_without_types(example: dict[str, Any], column: str) -> dict[str, Any]:
    # Delete type annotations and definitions
    content = example[column]
    stripped = util.delete_types(content)
    example[COLUMN_WITHOUT_TYPES] = stripped
    return example

def run_baseline(
    example: dict[str, Any],
    typeinf: TypeInference,
    column: str
) -> dict[str, Any]:
    # TODO: For now, assume TypeScript with type annotations and definitions
    # that we strip. Later we can preprocess the dataset or look at JavaScript
    # datasets.
    content = example[column]
    stripped = example[COLUMN_WITHOUT_TYPES]

    # Run type inference
    output = typeinf.infer_with_definitions(stripped)

    # TODO: Evaluate with accuracy. Save to "accuracy" column
    # Later we can type check (using TypeScript LSP or compiler)

    return {
        "hexsha": example["hexsha"],
        "max_stars_repo_path": example["max_stars_repo_path"],
        "max_stars_repo_name": example["max_stars_repo_name"],
        "content": content,
        "output": output,
        "accuracy": 0.0
    }

def main():
    args = parse_args()

    typeinf = TypeInference(Model())
    dataset = load_dataset(args.dataset, args.split, args.revision, args.workers)

    # Add column without types, then filter to remove empty rows (since removing
    # types may end up removing everything)
    num_examples = len(dataset)
    dataset = dataset.map(lambda e: add_column_without_types(e, args.content_column),
                          num_proc=args.workers)
    dataset = dataset.filter(lambda e: not util.is_empty(e[COLUMN_WITHOUT_TYPES]),
                             num_proc=args.workers)
    num_removed = num_examples - len(dataset)

    # Run the baseline experiment
    result = dataset.map(lambda e: run_baseline(e, typeinf, COLUMN_WITHOUT_TYPES),
                         num_proc=args.workers)

    # Print results statistics
    print("Number of examples in the original:", num_examples)
    print("Number of examples skipped:", num_removed)
    accuracy = pd.DataFrame({"accuracy": result["accuracy"]})
    print(accuracy.describe())

    # TODO: save result dataset to disk

if __name__ == "__main__":
    main()
