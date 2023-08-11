from pathlib import Path
from subprocess import DEVNULL, PIPE
from transformers import PreTrainedTokenizer
import evaluate
import json
import Levenshtein
import subprocess

from util import ROOT_DIR

ACCURACY_METRIC = evaluate.load("accuracy")

def accuracy(
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

def levenshtein(original: str, output: str) -> float:
    return Levenshtein.ratio(original, output)

def typescript(contents: str) -> tuple[int, int]:
    # TODO: seems to return too many type errors, maybe a missing lib in the config?
    args = [
        str(Path(ROOT_DIR, "ts", "node_modules", ".bin", "ts-node").resolve()),
        str(Path(ROOT_DIR, "ts", "main.ts").resolve())
    ]
    result = subprocess.run(
        args, input=contents, stdout=PIPE, stderr=DEVNULL, encoding="utf-8"
    )
    data = json.loads(result.stdout)

    return data["type_errors"], data["parse_errors"]
