from transformers import PreTrainedTokenizer
import evaluate
import Levenshtein

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
