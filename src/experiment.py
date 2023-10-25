from datasets import Dataset, IterableDataset
from enum import Enum
from typing import Any
import argparse
import functools
import gc
import random
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from model import Model, Tokenizer
from util import transform
import util

class ExperimentType(Enum):
    """
    One-shot: use the instruction "Add type annotations and interfaces".
    """
    APPROACH_1 = 1

    """
    Two steps: generate num_completions completions for the first instruction,
    then one completion for each of the results.
    Instructions:
      1. Add type aliases and interfaces.
      2. Add type annotations.
    """
    APPROACH_2 = 2

    """
    Two steps: generate num_completions completions for the first instruction,
    then one completion for each of the results.
    Instructions:
      1. Add type annotations.
      2. Add type aliases and interfaces.
    """
    APPROACH_3 = 3

    """
    Multiple steps:
      1. Generate num_completions completions for the instruction:
           "Add type annotations"
      2. Now handle each completion separately. While there are undefined
         type annotations, generate a definition for the undefined type T, with
         the instruction:
           "Add a type alias or interface for T"
    """
    APPROACH_4 = 4

class ExperimentConfig:
    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        model_name: str,
        approach: ExperimentType
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.approach = approach

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
    model: Model,
    approach: ExperimentType,
    num_completions: int
) -> dict[str, Any]:
    # For now, we're assuming TypeScript with type annotations and definitions removed.
    stripped = example["content_without_types"]

    # Run type inference, depending on the approach we're using
    match approach:
        case ExperimentType.APPROACH_1:
            # One-shot: use the instruction "Add type annotations and interfaces"
            prompt = (stripped, "Add type annotations and interfaces")
            prompts = [prompt] * num_completions
            final_outputs = model.edit_batch(prompts)
        case ExperimentType.APPROACH_2:
            # Two steps: generate num_completions completions for the first
            # instruction, then one completion for each of the results
            prompt = (stripped, "Add type aliases and interfaces")
            prompts = [prompt] * num_completions
            outputs = model.edit_batch(prompts)
            prompts = [(o, "Add type annotations") for o in outputs]
            final_outputs = model.edit_batch(prompts)
        case ExperimentType.APPROACH_3:
            # Two steps: generate num_completions completions for the first
            # instruction, then one completion for each of the results
            prompt = (stripped, "Add type annotations")
            prompts = [prompt] * num_completions
            outputs = model.edit_batch(prompts)
            prompts = [(o, "Add type aliases and interfaces") for o in outputs]
            final_outputs = model.edit_batch(prompts)
        case ExperimentType.APPROACH_4:
            # TODO: test this on a small dataset
            # First generate num_completions completions to add type annotations
            prompt = (stripped, "Add type annotations")
            prompts = [prompt] * num_completions
            outputs = model.edit_batch(prompts)
            final_outputs = []
            # Set a limit of 10 tries
            counter = 1
            while True:
                # Get list of undefined types for each output
                outputs_and_undef = [(o, transform.get_undefined_type_names(o))
                                    for o in outputs]
                # Partition the outputs on whether they have undefined types
                done, todo = util.partition_list(lambda p: not p[1],
                                                 outputs_and_undef)
                # Append the code outputs (not the types) to final_outputs
                final_outputs += [d for d, _ in done]
                if not todo or counter > 10:
                    break
                # For each completion in todo, pick one of the undefined types
                # and construct a prompt
                prompts = [(code,
                            "Add a type alias or interface for "
                            f"{random.choice(types)}")
                           for code, types in todo]
                outputs = model.edit_batch(prompts)
                counter += 1

    results = [{"output": o, "error": o == ""} for o in final_outputs]
    example["results"] = results

    return example

def _run_inference(
    dataset: Dataset | IterableDataset,
    model: Model,
    tokenizer: Tokenizer,
    approach: ExperimentType,
    num_completions: int,
    workers: int
) -> Dataset | IterableDataset:
    dataset = _prepare_dataset(dataset, model, tokenizer, workers)

    with util.timer():
        dataset = dataset.map(
            functools.partial(
                _infer_on_example,
                model=model,
                approach=approach,
                num_completions=num_completions
            ),
            desc="Inferring types"
        )

    dataset = dataset.select_columns([
        "hexsha",
        "max_stars_repo_path",
        "max_stars_repo_name",
        "content",
        "content_without_types",
        "results",
    ])

    return dataset

def run_experiment(config: ExperimentConfig, args: argparse.Namespace):
    # For now, the output name is {model_name}.parquet. Later we might have
    # different experiments for a model, so we will need different names.
    results_path = util.get_results_name(config.model_name, args.results_directory)

    model_path = util.get_model_path(config.model_name, args.models_directory)
    model = Model(model_path)
    tokenizer = Tokenizer(model_path)
    dataset = config.dataset

    num_examples = len(dataset)
    dataset = _run_inference(
        dataset,
        model,
        tokenizer,
        config.approach,
        args.num_completions,
        args.workers
    )
    num_removed = num_examples - len(dataset)

    # Save results to disk
    util.save_dataset(dataset, results_path, args.workers)

    print("Number of examples in original:", num_examples)
    print("Number of examples skipped:", num_removed)

    # Cleanup: destroy Model/vLLM so we can start the next experiment
    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()
