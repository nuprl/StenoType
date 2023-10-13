from functools import partial
from pathlib import Path
from subprocess import DEVNULL, PIPE
from transformers import PreTrainedTokenizer
from typing import Any
import argparse
import evaluate
import json
import Levenshtein
import subprocess

from model import Tokenizer
from experiment import ExperimentConfig
from util import ROOT_DIR
import util

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

def _evaluate_one_completion(
    completion: dict[str, Any],
    tokenizer: Tokenizer,
    original: str
) -> dict[str, Any]:
    if completion["error"]:
        return completion

    output = completion["output"]

    completion["token_count"] = len(tokenizer(output))
    completion["accuracy"] = _accuracy(tokenizer.tokenizer, original, output)
    completion["levenshtein"] = _levenshtein(original, output)

    te, pe = _typescript(output)
    completion["type_errors"] = te
    completion["parse_errors"] = pe
    completion["type_checks"] = te == 0 and pe == 0

    return completion

def _evaluate_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer
) -> dict[str, Any]:
    original = example["content"]
    completions = example["results"]
    example["results"] = [_evaluate_one_completion(c, tokenizer, original)
                          for c in completions]

    return example

def run_evaluation(config: ExperimentConfig, args: argparse.Namespace):
    # For now, the output name is {model_name}.parquet. Later we might have
    # different experiments for a model, so we will need different names.
    results_path = util.get_results_name(config.model_name, args.results_directory)
    dataset = util.load_dataset(results_path)

    model_path = util.get_model_path(config.model_name, args.models_directory)
    tokenizer = Tokenizer(model_path)

    # If we already processed this, early return
    first_result = dataset[0]["results"][0]
    if len(first_result.keys()) > 2:
        print(f"Skipping {results_path} because it was already processed")
        return

    # TODO: could be more efficient (but more complicated) to give each task
    # a worker, instead of each example a worker
    dataset = dataset.map(
        partial(_evaluate_example, tokenizer=tokenizer),
        num_proc=args.workers,
        desc="Evaluating results"
    )

    # Save dataset
    util.save_dataset(dataset, results_path, args.workers)

    # Print result statistics
    # print("Number of examples failed:", num_errors)
    # print("Number of examples that type checked:", num_typechecked)
    #
    # print(f"Accuracy: {np.mean(dataset['accuracy']):.1%}")
    # print(f"Levenshtein: {np.mean(dataset['levenshtein']):.1%}")
    # print(f"Type checked: {pct_typechecked:.1%}")
    # print(f"Type errors: {np.mean(dataset['type_errors']):.1f}")
    # print(f"Parse errors: {np.mean(dataset['parse_errors']):.1f}")
