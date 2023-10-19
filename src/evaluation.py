from functools import partial
from pathlib import Path
from subprocess import DEVNULL, PIPE
from transformers import PreTrainedTokenizer
from typing import Any, Optional
import argparse
import evaluate
import json
import Levenshtein
import numpy as np
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

def _typescript(contents: str) -> Optional[tuple[int, int]]:
    args = ["node", str(Path(ROOT_DIR, "ts", "main.js").resolve())]
    result = subprocess.run(
        args, input=contents, stdout=PIPE, stderr=DEVNULL, encoding="utf-8"
    )
    if len(result.stdout) > 0:
        data = json.loads(result.stdout)
        if data:
            return data["type_errors"], data["parse_errors"]

    return None

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

    res = _typescript(output)
    if res:
        te, pe = res
        completion["type_errors"] = te
        completion["parse_errors"] = pe
        completion["type_checks"] = te == 0 and pe == 0
    else:
        completion["error"] = True

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
    if all(len(c.keys()) > 2 for r in dataset["results"] for c in r):
        print(f"Skipping {results_path} because it was already processed")
        return

    dataset = dataset.map(
        partial(_evaluate_example, tokenizer=tokenizer),
        num_proc=args.workers,
        desc="Evaluating results"
    )

    # Save dataset
    util.save_dataset(dataset, results_path, args.workers)

def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Parameters:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    A numerically stable script for calculating an unbiased estimate of pass@k.
    Taken from Fig. 3 of the OpenAI Codex paper, "Evaluating Large Language
    Models Trained on Code," https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _summarize_example(example: dict[str, Any]) -> dict[str, Any]:
    results = example["results"]

    num_completions = len(results)
    num_type_checks = len([r for r in results if r["type_checks"]])
    pct_type_checks = num_type_checks / num_completions
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    avg_levenshtein = np.mean([r["levenshtein"] for r in results])
    avg_type_errors = np.mean([r["type_errors"] for r in results])
    avg_parse_errors = np.mean([r["parse_errors"] for r in results])
    pass_1 = _pass_at_k(num_completions, num_type_checks, 1)

    example["num_completions"] = num_completions
    example["num_type_checks"] = num_type_checks
    example["pct_type_checks"] = pct_type_checks
    example["avg_accuracy"] = avg_accuracy
    example["avg_levenshtein"] = avg_levenshtein
    example["avg_type_errors"] = avg_type_errors
    example["avg_parse_errors"] = avg_parse_errors
    example["pass@1"] = pass_1

    return example

def _remove_errors(example: dict[str, Any]) -> dict[str, Any]:
    example["results"] = [r for r in example["results"] if not r["error"]]
    return example

def _summarize_dataset(
    config: ExperimentConfig,
    args: argparse.Namespace
) -> dict[str, Any]:
    results_path = util.get_results_name(config.model_name, args.results_directory)
    dataset = util.load_dataset(results_path)

    # Filter out results with errors
    all_completions = len([r for d in dataset for r in d["results"]])
    dataset = dataset.map(
        _remove_errors,
        num_proc=args.workers,
        desc="Removing errors"
    )
    no_errors = len([r for d in dataset for r in d["results"]])

    # Summarize each example
    dataset = dataset.map(
        _summarize_example,
        num_proc=args.workers,
        desc="Summarizing results"
    )
    util.save_dataset(dataset, results_path, args.workers)

    # Summarize dataset
    total_completions = np.sum(dataset["num_completions"])
    total_errors = all_completions - no_errors
    total_type_checks = np.sum(dataset["num_type_checks"])
    pct_errors = total_errors / total_completions
    pct_type_checks = total_type_checks / total_completions
    avg_accuracy = np.mean([r["accuracy"]
                            for d in dataset for r in d["results"]])
    avg_levenshtein = np.mean([r["levenshtein"]
                               for d in dataset for r in d["results"]])
    avg_type_errors = np.mean([r["type_errors"]
                               for d in dataset for r in d["results"]])
    avg_parse_errors = np.mean([r["parse_errors"]
                                for d in dataset for r in d["results"]])
    pass_1 = np.mean(dataset["pass@1"])

    return {
        "model": config.model_name,
        "num_problems": len(dataset),
        "total_completions": total_completions,
        "total_errors": total_errors,
        "total_type_checks": total_type_checks,
        "pct_errors": pct_errors,
        "pct_type_checks": pct_type_checks,
        "avg_accuracy": avg_accuracy,
        "avg_levenshtein": avg_levenshtein,
        "avg_type_errors": avg_type_errors,
        "avg_parse_errors": avg_parse_errors,
        "pass@1": pass_1,
    }

def summarize_results(configs: list[ExperimentConfig], args: argparse.Namespace):
    summaries = []
    for config in configs:
        # Note: _summarize_dataset adds columns to the dataset and writes to disk
        summary = _summarize_dataset(config, args)
        summaries.append(summary)

        # Print the summary
        print(f"===Stats for model {config.model_name}===")
        print(f"Number of problems: {summary['num_problems']}")
        print(f"Total completions: {summary['total_completions']}")
        print(f"Errors: {summary['total_errors']} "
              f"({summary['pct_errors']:.1%})")
        print(f"Type checks: {summary['total_type_checks']} "
              f"({summary['pct_type_checks']:.1%})")
        print(f"Accuracy: {summary['avg_accuracy']:.1%}")
        print(f"Levenshtein: {summary['avg_levenshtein']:.1%}")
        print(f"Type errors: {summary['avg_type_errors']:.1f}")
        print(f"Parse errors: {summary['avg_parse_errors']:.1f}")
        print(f"pass@1 (type checking): {summary['pass@1']:.1%}")
        print()

    # Write to jsonl file
    path = Path(args.results_directory, "summary.jsonl")
    util.write_jsonl(path, summaries, util.NpEncoder)
