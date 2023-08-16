from datasets import Dataset, IterableDataset
from functools import partial
from pathlib import Path
from subprocess import DEVNULL, PIPE
from transformers import PreTrainedTokenizer
from typing import Any
import evaluate
import json
import Levenshtein
import numpy as np
import subprocess

from util import ROOT_DIR

ACCURACY_METRIC = evaluate.load("accuracy")

def _accuracy(
    tokenizer: PreTrainedTokenizer,
    original: str,
    output: str
) -> float:
    # Tokenize the original and output, and pad them to the same length
    # NumPy tensors may be more memory efficient than Python lists
    original_tokens, output_tokens = tokenizer(
        [original, output],
        padding=True,
        return_attention_mask=False,
        return_tensors="np"
    )["input_ids"]

    return ACCURACY_METRIC.compute(
        references=original_tokens,
        predictions=output_tokens
    )["accuracy"]

def _levenshtein(original: str, output: str) -> float:
    return Levenshtein.ratio(original, output)

def _typescript(contents: str) -> tuple[int, int]:
    args = ["node", str(Path(ROOT_DIR, "ts", "main.js").resolve())]
    result = subprocess.run(
        args, input=contents, stdout=PIPE, stderr=DEVNULL, encoding="utf-8"
    )
    data = json.loads(result.stdout)

    return data["type_errors"], data["parse_errors"]

def _evaluate_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    original_column: str,
    output_column: str
) -> dict[str, Any]:
    original = example[original_column]
    output = example[output_column]

    example["accuracy"] = _accuracy(tokenizer, original, output)
    example["levenshtein"] = _levenshtein(original, output)
    example["type_errors"], example["parse_errors"] = _typescript(output)

    return example

def run_evaluation(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizer,
    num_examples: int,
    num_removed: int,
    content_column: str,
    output_column: str,
    error_column: str,
    workers: int
) -> Dataset | IterableDataset:
    # Remove examples that had errors
    num_runs = len(dataset)
    dataset = dataset.filter(
        lambda e: not e[error_column],
        num_proc=workers,
        desc="Removing failed runs"
    )
    num_errors = num_runs - len(dataset)

    dataset = dataset.map(
        partial(
            _evaluate_example,
            tokenizer=tokenizer,
            original_column=content_column,
            output_column=output_column
        ),
        num_proc=workers,
        desc="Evaluating results"
    )

    num_typechecked = len([d for d in dataset
                           if d["type_errors"] == 0 and d["parse_errors"] == 0])
    pct_typechecked = num_typechecked / len(dataset)

    # Print result statistics
    print("Number of examples in the original:", num_examples)
    print("Number of examples skipped:", num_removed)
    print("Number of examples failed:", num_errors)
    print("Number of examples that type checked:", num_typechecked)

    print()
    print(f"Accuracy: {np.mean(dataset['accuracy']):.1%}")
    print(f"Levenshtein: {np.mean(dataset['levenshtein']):.1%}")
    print(f"Type checked: {pct_typechecked:.1%}")
    print(f"Type errors: {np.mean(dataset['type_errors']):.1f}")
    print(f"Parse errors: {np.mean(dataset['parse_errors']):.1f}")
    print()

    return dataset
