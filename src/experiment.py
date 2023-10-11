from datasets import Dataset, IterableDataset
from enum import Enum
from pathlib import Path
from typing import Any
import argparse
import functools

from evaluation import run_evaluation
from model import Model, Tokenizer
from type_inference import TypeInference
from util import transform
import util

class ExperimentType(Enum):
    APPROACH_1 = 1
    APPROACH_2 = 2
    APPROACH_3 = 3

def _add_column_without_types(
    example: dict[str, Any]
) -> dict[str, Any]:
    # Delete type annotations and definitions
    content = example["content"]
    stripped = transform.delete_types(content)
    example["content_without_types"] = stripped
    return example

def _prepare_dataset(
    dataset: Dataset | IterableDataset,
    model: Model,
    tokenizer: Tokenizer,
    workers: int
) -> Dataset | IterableDataset:
    # Remove type annotations and definitions, add as new column
    dataset = dataset.map(
        _add_column_without_types,
        num_proc=workers,
        desc="Removing types"
    )

    # Remove empty rows (since removing types may end up removing everything)
    dataset = dataset.filter(
        lambda e: not transform.is_empty(e["content_without_types"]),
        num_proc=workers,
        desc="Removing empty examples"
    )

    # Remove examples that are too long
    input_size = model.INPUT_SIZE
    dataset = dataset.filter(
        lambda e: len(tokenizer(e["content_without_types"])) < input_size,
        num_proc=workers,
        desc="Removing large examples"
    )

    return dataset

def _infer_on_example(
    example: dict[str, Any],
    approach: ExperimentType,
    typeinf: TypeInference
) -> dict[str, Any]:
    # For now, we're assuming TypeScript with type annotations and definitions removed.
    stripped = example["content_without_types"]

    # Run type inference, depending on the approach we're using
    match approach:
        case ExperimentType.APPROACH_1:
            # One-shot: use the instruction "Add type annotations and interfaces"
            output = typeinf.infer_with_definitions(stripped)
        case ExperimentType.APPROACH_2:
            output = typeinf._edit(stripped, [
                "Add type aliases and interfaces",
                "Add type annotations"
            ])
        case ExperimentType.APPROACH_3:
            output = typeinf._edit(stripped, [
                "Add type annotations"
                "Add type aliases and interfaces",
            ])

    if output:
        example["output"] = output
        example["error"] = False
    else:
        example["output"] = ""
        example["error"] = True

    return example

def _run_inference(
    dataset: Dataset | IterableDataset,
    model: Model,
    tokenizer: Tokenizer,
    approach: ExperimentType,
    workers: int
) -> Dataset | IterableDataset:
    typeinf = TypeInference(model)
    dataset = _prepare_dataset(dataset, model, tokenizer, workers)

    # TODO: batch completions, or generate multiple completions
    with util.timer():
        dataset = dataset.map(
            functools.partial(_infer_on_example, approach=approach, typeinf=typeinf),
            desc="Inferring types"
        )

    dataset = dataset.select_columns([
        "hexsha",
        "max_stars_repo_path",
        "max_stars_repo_name",
        "content",
        "content_without_types",
        "output",
        "error",
    ])

    return dataset

def run_experiment(
    dataset: Dataset | IterableDataset,
    model_name: str,
    approach: ExperimentType,
    args: argparse.Namespace
):
    # TODO: For now, the output name is {model_name}.parquet. Later we might
    # have different experiments for a model, so we will need different names.
    results_path = str(Path(args.results_directory, model_name).with_suffix(".parquet"))
    if Path(results_path).exists():
        print(f"error: output {results_path} already exists, please delete or rename!")
        exit(2)

    model_path = str(Path(args.models_directory, model_name))
    model = Model(model_path)
    tokenizer = Tokenizer(model_path)

    # Run inference
    num_examples = len(dataset)
    dataset = _run_inference(dataset, model, tokenizer, approach, args.workers)
    num_removed = num_examples - len(dataset)

    # Run evaluation
    dataset = run_evaluation(
        dataset,
        tokenizer.tokenizer,
        num_examples,
        num_removed,
        args.workers
    )

    # Save results to disk
    util.save_dataset(dataset, results_path, args.workers)
