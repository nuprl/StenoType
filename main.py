from datasets import (Dataset,
                      DatasetDict,
                      IterableDataset,
                      IterableDatasetDict)
from evaluate import EvaluationModule
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import Any, Optional
import argparse
import datasets
import evaluate
import pandas as pd
import os

from model import Model
from type_inference import TypeInference
import util

COLUMN_WITHOUT_TYPES = "content_without_types"
OUTPUT_COLUMN = "output"

def parse_args():
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

def save_dataset(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    output: Optional[str],
    workers: int
) -> None:
    if not output:
        return

    print(f"Saving results to {output}")
    if output.endswith(".parquet"):
        dataset.to_parquet(output)
    elif output.endswith(".jsonl"):
        dataset.to_json(output)
    else:
        dataset.save_to_disk(output, num_proc=workers)

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

    return {
        "hexsha": example["hexsha"],
        "max_stars_repo_path": example["max_stars_repo_path"],
        "max_stars_repo_name": example["max_stars_repo_name"],
        "content": content,
        OUTPUT_COLUMN: output,
    }

def compute_accuracy(
    example: dict[str, Any],
    metric: EvaluationModule,
    tokenizer: PreTrainedTokenizer,
    original_column: str,
    output_column: str
) -> dict[str, Any]:
    original = example[original_column]
    output = example[output_column]

    # Tokenize the original and output, and pad them to the same length
    # NumPy tensors may be more memory efficient than Python lists
    original_tokens, output_tokens = tokenizer([original, output],
                                               padding=True,
                                               return_tensors="np")["input_ids"]

    example["accuracy"] = metric.compute(references=original_tokens,
                                         predictions=output_tokens)["accuracy"]

    return example

def main():
    args = parse_args()

    model = Model()
    typeinf = TypeInference(model)
    dataset = load_dataset(args.dataset, args.split, args.revision, args.workers)

    # Add column without types, then filter to remove empty rows (since removing
    # types may end up removing everything)
    num_examples = len(dataset)
    dataset = dataset.map(lambda e: add_column_without_types(e, args.content_column),
                          num_proc=args.workers, desc="Removing types")
    dataset = dataset.filter(lambda e: not util.is_empty(e[COLUMN_WITHOUT_TYPES]),
                             num_proc=args.workers, desc="Removing empty examples")
    num_removed = num_examples - len(dataset)

    # Run the baseline experiment
    dataset = dataset.map(lambda e: run_baseline(e, typeinf, COLUMN_WITHOUT_TYPES),
                          num_proc=args.workers, desc="Inferring types")

    # Evaluate the result
    # TODO: For now, use accuracy; later we can type check (e.g. using tsc or LSP)
    accuracy_metric = evaluate.load("accuracy")

    dataset = dataset.map(lambda e: compute_accuracy(e,
                                                     accuracy_metric,
                                                     model.tokenizer,
                                                     args.content_column,
                                                     OUTPUT_COLUMN),
                          num_proc=args.workers, desc="Evaluating results")

    # Print results statistics
    print("Number of examples in the original:", num_examples)
    print("Number of examples skipped:", num_removed)
    accuracy = pd.DataFrame({"accuracy": dataset["accuracy"]})
    print(accuracy.describe())

    # Save result dataset to disk
    save_dataset(dataset, args.output, args.workers)

if __name__ == "__main__":
    main()
