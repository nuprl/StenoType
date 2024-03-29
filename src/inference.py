from datasets import Dataset, IterableDataset
from pathlib import Path
from subprocess import DEVNULL, PIPE
from typing import Any, Callable, Optional
import argparse
import functools
import random
import shutil
import subprocess
import tarfile
import tempfile

from model import Model
from util import transform
import util

TSC_PATH = shutil.which("tsc")
if not TSC_PATH:
    print("Could not find tsc")
    exit(1)


class DatasetConfig:
    dataset: Optional[Dataset | IterableDataset] = None

    def __init__(
        self,
        short_name: str,
        datasets_path: str,
        dataset_name: str,
        type_decls: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        num_proc: Optional[int] = None,
        streaming: Optional[bool] = None,
    ):
        self.short_name = short_name
        self.dataset_path = str(Path(datasets_path, dataset_name).resolve())
        self.type_decls = (
            str(Path(datasets_path, type_decls).resolve()) if type_decls else None
        )
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


class Config:
    def __init__(
        self,
        model_name: str,
        approach: Optional[Callable[[Model, int, str], list[str]]],
        dataset_config: DatasetConfig,
    ):
        self.dataset_config = dataset_config
        self.model_name = model_name
        self.approach = approach
        self.approach_name = self.approach.__name__ if self.approach else ""
        self.filename = (
            f"{self.model_name}_{self.approach_name}_{self.dataset_config.short_name}"
        )

    def _output_path(self, results_directory: str, subdir: str) -> str:
        output_dir = Path(results_directory, subdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(Path(output_dir, self.filename).with_suffix(".parquet"))

    def infer_output_path(self, results_directory: str) -> str:
        return self._output_path(results_directory, "0_after_infer")

    def eval_output_path(self, results_directory: str) -> str:
        return self._output_path(results_directory, "1_after_eval")

    def summary_output_path(self, results_directory: str) -> str:
        return self._output_path(results_directory, "2_after_summary")


def approach1(
    model: Model, num_completions: int, original: str
) -> tuple[list[str], list[list[dict]]]:
    """
    One-shot: use the instruction "Add type annotations and interfaces".

    Returns a pair:
        - list of outputs (strings)
        - list of intermediate results (each intermediate result is a list of
          dictionaries with "input" and "prompt" keys)
    """
    prompts = [
        (original, "Add type annotations and interfaces")
        for _ in range(num_completions)
    ]
    # Add the prompts to the intermediate results (each item is a list of steps)
    intermediate = [[{"input": p[0], "prompt": p[1]}] for p in prompts]
    outputs = model.edit_batch(prompts)
    return outputs, intermediate


def approach2(
    model: Model, num_completions: int, original: str
) -> tuple[list[str], list[list[dict]]]:
    """
    Two steps: generate num_completions completions for the first instruction,
    then one completion for each of the results.
    Instructions:
      1. Add type aliases and interfaces.
      2. Add type annotations.

    Returns a pair:
        - list of outputs (strings)
        - list of intermediate results (each intermediate result is a list of
          dictionaries with "input" and "prompt" keys)
    """
    prompts = [
        (original, "Add type aliases and interfaces") for _ in range(num_completions)
    ]
    # Add the prompts to the intermediate results (each item is a list of steps)
    intermediate = [[{"input": p[0], "prompt": p[1]}] for p in prompts]
    outputs = model.edit_batch(prompts)

    prompts = [(o, "Add type annotations") for o in outputs]
    # Append the last input/prompt to each intermediate result
    for i, p in zip(intermediate, prompts):
        i.append({"input": p[0], "prompt": p[1]})
    outputs = model.edit_batch(prompts)
    return outputs, intermediate


def approach3(
    model: Model, num_completions: int, original: str
) -> tuple[list[str], list[list[dict]]]:
    """
    Two steps: generate num_completions completions for the first instruction,
    then one completion for each of the results.
    Instructions:
      1. Add type annotations.
      2. Add type aliases and interfaces.

    Returns a pair:
        - list of outputs (strings)
        - list of intermediate results (each intermediate result is a list of
          dictionaries with "input" and "prompt" keys)
    """
    prompts = [(original, "Add type annotations") for _ in range(num_completions)]
    # Add the prompts to the intermediate results (each item is a list of steps)
    intermediate = [[{"input": p[0], "prompt": p[1]}] for p in prompts]
    outputs = model.edit_batch(prompts)

    prompts = [(o, "Add type aliases and interfaces") for o in outputs]
    # Append the last input/prompt to each intermediate result
    for i, p in zip(intermediate, prompts):
        i.append({"input": p[0], "prompt": p[1]})
    outputs = model.edit_batch(prompts)
    return outputs, intermediate


def approach4(
    model: Model, num_completions: int, original: str
) -> tuple[list[str], list[list[dict]]]:
    """
    Multiple steps:
      1. Generate num_completions completions for the instruction:
           "Add type annotations"
      2. Now handle each completion separately. While there are undefined
         type annotations, generate a definition for the undefined type T, with
         the instruction:
           "Add a type alias or interface for T"

    Returns a pair:
        - list of outputs (strings)
        - list of intermediate results (each intermediate result is a list of
          dictionaries with "input" and "prompt" keys)
    """
    MAX_TRIES = 5

    # First generate num_completions completions to add type annotations
    prompts = [(original, "Add type annotations") for _ in range(num_completions)]
    # Add the prompts to the intermediate results (each item is a list of steps)
    intermediate = [[{"input": p[0], "prompt": p[1]}] for p in prompts]
    outputs = model.edit_batch(prompts)

    # Now loop to add type definitions
    final_outputs, final_intermediate = [], []
    counter = 1
    while True:
        # For each output, make a triple:
        # (output, undefined types, intermediate results)
        output_triples: list[tuple[str, list[str], list[dict]]] = [
            (o, transform.get_undefined_type_names(o), i)
            for o, i in zip(outputs, intermediate)
        ]

        # Partition the outputs on whether they have undefined types
        # done and todo have the same type as output_triples
        done, todo = util.partition_list(lambda p: not p[1], output_triples)

        # Extract the outputs and intermediate results from done,
        # and append to return variables
        final_outputs += [o for o, _, _ in done]
        final_intermediate += [i for _, _, i in done]

        # Exit the loop if there's nothing left or we exceeded MAX_TRIES
        if not todo or counter > MAX_TRIES:
            # Extract the outputs and intermediate results from todo,
            # and append to return variables
            final_outputs += [o for o, _, _ in todo]
            final_intermediate += [i for _, _, i in todo]
            break

        # For each completion in todo, pick one of the undefined types
        # and construct a prompt
        prompts = [
            (code, f"Add a type alias or interface for {random.choice(types)}")
            for code, types, _ in todo
        ]
        # Append the last input/prompt to each intermediate result
        intermediate = [i for _, _, i in todo]
        for i, p in zip(intermediate, prompts):
            i.append({"input": p[0], "prompt": p[1]})
        outputs = model.edit_batch(prompts)
        counter += 1

    return final_outputs, final_intermediate


def _add_column_without_types(example: dict[str, Any]) -> dict[str, Any]:
    # Delete type annotations and definitions
    content = example["content"]
    stripped = transform.delete_types(content, delete_comments=True)
    example["content_without_types"] = stripped
    return example


def _infer_on_example(
    example: dict[str, Any],
    model: Model,
    approach: Callable[[Model, int, str], tuple[list[str], list[list[dict]]]],
    num_completions: int,
) -> dict[str, Any]:
    stripped = example["content_without_types"]

    # Run type inference, depending on the approach we're using
    # If the input is too long, the output will be empty and we'll consider that an error
    outputs, intermediates = approach(model, num_completions, stripped)

    results = [
        {"output": o, "error": o == "", "steps": i}
        for o, i in zip(outputs, intermediates)
    ]
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
    approach: Optional[Callable[[Model, int, str], list[str]]],
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


def _run_tsc_on_example(
    example: dict[str, Any],
    tmpdir: str,
    is_js: bool,
) -> dict[str, Any]:
    stripped = example["content_without_types"]

    extension = ".js" if is_js else ".ts"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=extension, dir=tmpdir, encoding="utf-8"
    ) as f:
        print(stripped, file=f, end="", flush=True)
        tmpfile = Path(f.name)

        # Run tsc on temp file
        args = [
            str(TSC_PATH),
            "--declaration",
            "--allowJs",
            "--emitDeclarationOnly",
            "--esModuleInterop",
            "--moduleResolution", "node",
            "--target", "es2022",
            "--lib", "es2022,dom",
            str(tmpfile),
        ]  # fmt: skip
        result = subprocess.run(
            args, stdout=PIPE, stderr=DEVNULL, encoding="utf-8", cwd=tmpdir
        )

        dts_file = tmpfile.with_suffix(".d.ts")
        if dts_file.exists():
            # Need to read .d.ts file
            with open(tmpfile.with_suffix(".d.ts"), "r") as dts:
                dts_contents = dts.read()
        else:
            dts_contents = ""

        example["results"] = [
            {
                "output": dts_contents,
                "errors": result.stdout,
                "error": dts_contents == "",
            }
        ]

        tmpfile.with_suffix(".d.ts").unlink(missing_ok=True)

    return example


def _run_tsc_inference(
    dataset: Dataset | IterableDataset,
    type_decls: Optional[str],
    workers: int,
) -> Dataset | IterableDataset:
    # Remove type annotations and definitions, add as new column
    dataset = dataset.map(
        _add_column_without_types, num_proc=workers, desc="Removing types"
    )

    with util.timer():
        with tempfile.TemporaryDirectory() as tmpdir:
            if type_decls:
                with tarfile.open(type_decls) as tar:
                    tar.extractall(tmpdir)

            dataset = dataset.map(
                functools.partial(
                    _run_tsc_on_example,
                    tmpdir=tmpdir,
                    is_js=type_decls is not None,
                ),
                num_proc=workers,
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


def run_inference(config: Config, args: argparse.Namespace):
    results_path = config.infer_output_path(args.results_directory)

    dataset = config.dataset_config.get()
    if config.model_name == "tsc":
        dataset = _run_tsc_inference(
            dataset, config.dataset_config.type_decls, args.workers
        )
    else:
        model_path = util.get_model_path(config.model_name, args.models_directory)
        with Model(model_path) as model:
            dataset = _run_inference(
                dataset, model, config.approach, args.num_completions, args.workers
            )

    # Save results to disk
    util.save_dataset(dataset, results_path, args.workers)
    print()
