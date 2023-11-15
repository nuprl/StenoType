from datasets import Dataset, IterableDataset
from pathlib import Path
from typing import Any, Callable, Optional
import argparse
import functools
import random

from model import Model, Tokenizer
from util import transform
import util


class DatasetConfig:
    dataset: Optional[Dataset | IterableDataset] = None

    def __init__(
        self,
        name: str,
        dataset_path: str,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        num_proc: Optional[int] = None,
        streaming: Optional[bool] = None,
    ):
        self.name = name
        self.dataset_path = dataset_path
        self.split = split
        self.revision = revision
        self.num_proc = num_proc
        self.streaming = streaming

    def get(self):
        if not self.dataset:
            self.dataset = util.load_dataset(
                self.dataset_path,
                split=self.split,
                revision=self.revision,
                num_proc=self.num_proc,
                streaming=self.streaming,
            )
        return self.dataset


class ExperimentConfig:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        model_name: str,
        approach: Callable[[Model, int, str], list[str]],
    ):
        self.dataset_config = dataset_config
        self.model_name = model_name
        self.approach = approach

    def _output_path(self, results_directory: str, subdir: str) -> str:
        output_dir = Path(results_directory, subdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.model_name}_{self.approach.__name__}_{self.dataset_config.name}"
        )
        return str(Path(output_dir, filename).with_suffix(".parquet"))

    def infer_output_path(self, results_directory: str) -> str:
        return self._output_path(results_directory, "0_after_infer")

    def eval_output_path(self, results_directory: str) -> str:
        return self._output_path(results_directory, "1_after_eval")

    def summary_output_path(self, results_directory: str) -> str:
        return self._output_path(results_directory, "2_after_summary")


def approach1(model: Model, num_completions: int, original: str) -> list[str]:
    """
    One-shot: use the instruction "Add type annotations and interfaces".
    """
    prompt = (original, "Add type annotations and interfaces")
    prompts = [prompt] * num_completions
    return model.edit_batch(prompts)


def approach2(model: Model, num_completions: int, original: str) -> list[str]:
    """
    Two steps: generate num_completions completions for the first instruction,
    then one completion for each of the results.
    Instructions:
      1. Add type aliases and interfaces.
      2. Add type annotations.
    """
    prompt = (original, "Add type aliases and interfaces")
    prompts = [prompt] * num_completions
    outputs = model.edit_batch(prompts)
    prompts = [(o, "Add type annotations") for o in outputs]
    return model.edit_batch(prompts)


def approach3(model: Model, num_completions: int, original: str) -> list[str]:
    """
    Two steps: generate num_completions completions for the first instruction,
    then one completion for each of the results.
    Instructions:
      1. Add type annotations.
      2. Add type aliases and interfaces.
    """
    prompt = (original, "Add type annotations")
    prompts = [prompt] * num_completions
    outputs = model.edit_batch(prompts)
    prompts = [(o, "Add type aliases and interfaces") for o in outputs]
    return model.edit_batch(prompts)


def approach4(model: Model, num_completions: int, original: str) -> list[str]:
    """
    Multiple steps:
      1. Generate num_completions completions for the instruction:
           "Add type annotations"
      2. Now handle each completion separately. While there are undefined
         type annotations, generate a definition for the undefined type T, with
         the instruction:
           "Add a type alias or interface for T"
    """
    MAX_TRIES = 5

    # First generate num_completions completions to add type annotations
    prompt = (original, "Add type annotations")
    prompts = [prompt] * num_completions
    outputs = model.edit_batch(prompts)

    # Now loop to add type definitions
    final_outputs = []
    counter = 1
    while True:
        # Get list of undefined types for each output
        outputs_and_undef = [
            (o, transform.get_undefined_type_names(o)) for o in outputs
        ]

        # Partition the outputs on whether they have undefined types
        done, todo = util.partition_list(lambda p: not p[1], outputs_and_undef)

        # Append the code outputs (not the types) to final_outputs
        final_outputs += [d for d, _ in done]

        # Exit the loop if there's nothing left or we exceeded MAX_TRIES
        if not todo or counter > MAX_TRIES:
            # Append the not-completed outputs to final_outputs
            final_outputs += [t for t, _ in todo]
            break

        # For each completion in todo, pick one of the undefined types
        # and construct a prompt
        prompts = [
            (code, f"Add a type alias or interface for {random.choice(types)}")
            for code, types in todo
        ]
        outputs = model.edit_batch(prompts)
        counter += 1

    return final_outputs


def _add_column_without_types(example: dict[str, Any]) -> dict[str, Any]:
    # Delete type annotations and definitions
    content = example["content"]
    stripped = transform.delete_types(content, delete_comments=True)
    example["content_without_types"] = stripped
    return example


def _infer_on_example(
    example: dict[str, Any],
    model: Model,
    tokenizer: Tokenizer,
    approach: Callable[[Model, int, str], list[str]],
    num_completions: int,
) -> dict[str, Any]:
    stripped = example["content_without_types"]

    input_size = model.INPUT_SIZE
    if len(tokenizer(stripped)) >= input_size:
        # If example is too long, skip
        final_outputs = [""] * num_completions
    else:
        # Run type inference, depending on the approach we're using
        final_outputs = approach(model, num_completions, stripped)

    results = [{"output": o, "error": o == ""} for o in final_outputs]
    example["results"] = results

    return example


def _add_name_column(example: dict[str, Any]) -> dict[str, Any]:
    repo_name = example["max_stars_repo_name"]
    repo_path = example["max_stars_repo_path"]
    example["name"] = f"{repo_name}/{repo_path}"
    return example


def _run_inference(
    dataset: Dataset | IterableDataset,
    model: Model,
    tokenizer: Tokenizer,
    approach: Callable[[Model, int, str], list[str]],
    num_completions: int,
    workers: int,
) -> Dataset | IterableDataset:
    # Remove type annotations and definitions, add as new column
    dataset = dataset.map(
        _add_column_without_types, num_proc=workers, desc="Removing types"
    )

    with util.timer():
        dataset = dataset.map(
            functools.partial(
                _infer_on_example,
                model=model,
                tokenizer=tokenizer,
                approach=approach,
                num_completions=num_completions,
            ),
            desc="Inferring types",
        )

    # Add "name" column if dataset is from the Stack
    if (
        "max_stars_repo_name" in dataset.column_names
        and "max_stars_repo_name" in dataset.column_names
    ):
        dataset = dataset.map(_add_name_column, num_proc=workers)

    # Only keep the minimum necessary columns
    dataset = dataset.select_columns(
        ["name", "content", "content_without_types", "results"]
    )

    return dataset


def run_experiment(config: ExperimentConfig, args: argparse.Namespace):
    # For now, the output name is {model_name}.parquet. Later we might have
    # different experiments for a model, so we will need different names.
    results_path = config.infer_output_path(args.results_directory)

    model_path = util.get_model_path(config.model_name, args.models_directory)
    model = Model(model_path)
    tokenizer = Tokenizer(model_path)
    dataset = config.dataset_config.get()

    num_examples = len(dataset)
    dataset = _run_inference(
        dataset, model, tokenizer, config.approach, args.num_completions, args.workers
    )

    # Save results to disk
    util.save_dataset(dataset, results_path, args.workers)

    print("Number of examples in original:", num_examples)

    # Cleanup: destroy Model/vLLM so we can start the next experiment
    del model
    util.empty_gpu()
