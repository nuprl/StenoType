from datasets import (
    Dataset,
    IterableDataset,
    load_dataset
)
from pathlib import Path
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    logging,
    set_seed
)
from typing import Optional
import argparse

from finetune_lib import DatasetConfig
import finetune_lib as finetune
import util

"""
Edit the constants and configuration below to use for your own fine-tuning task.

Customize the `get_content` function to process a dataset element to get the
training text. This could be as simple as returning a column (e.g. getting the
value associated with the "content" key), or it may involve text from multiple
columns, or it may involve doing some other processing.

The `get_dataset` function allows you to customize how a dataset is loaded,
whether it is loaded from the Hugging Face Hub, loaded from disk, or requires
additional processing, e.g. interleaving and filtering multiple datasets.
"""

MODEL_PATH = str(Path(
    Path(__file__).parent,
    "..",
    "..",
    "models",
    "starcoderbase"
).resolve())

# We are using a very large dataset, so it's not feasible to download the whole
# thing. Instead, we stream the dataset and use an estimate for the number of
# tokens. This estimate was derived (in a separate script) by computing the size
# of the dataset (just the training columns) and estimating the bytes per token
# ratio.

# The original dataset has 7.1B tokens, but we multiply by 2 to account for
# the training format
TOTAL_TOKENS = 7_100_000_000 * 2

# We pack the tokens into a ConstantLengthDataset where each example has 8K tokens
SEQUENCE_LENGTH = 8192
# Roughly 1.7M examples
NUM_EXAMPLES = TOTAL_TOKENS // SEQUENCE_LENGTH

# TODO: adjust these numbers
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

# TODO: This doesn't account for multiple GPUs/processes
# Roughly 108K steps
MAX_STEPS = (EPOCHS * NUM_EXAMPLES) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

# Arguments for the training loop
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
TRAINING_ARGS = TrainingArguments(
    output_dir="../checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=5e-6,
    weight_decay=0.05,
    max_steps=MAX_STEPS,
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    logging_steps=1000,
    save_steps=5000,
    save_total_limit=5,
    bf16=True,
    fp16=False,
    local_rank=0,
    dataloader_drop_last=True,
    eval_steps=1000,
    run_name="StarCoder-finetuned",
    optim="adamw_torch",
    report_to="wandb",
    ddp_find_unused_parameters=False,
    resume_from_checkpoint=False,
    gradient_checkpointing=True,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["c_proj", "c_attn", "q_attn"],
)

def get_content(element: dict) -> Optional[str]:
    """
    Given an input example, i.e. a string containing the contents of a
    TypeScript file, process it and return the updated example for training.

    Specifically, we process it into the StarCoder edit format, e.g.

        <commit_before>{code without types}
        <commit_msg>{instruction}
        <commit_after>{original code}

    Returns None if the processed element is invalid; i.e. removing all the types
    removes the entire content, or there were no types to remove.
    """
    COMMIT_BEFORE = "<commit_before>"
    COMMIT_MSG = "<commit_msg>"
    COMMIT_AFTER = "<commit_after>"
    instruction = "Add type annotations and interfaces"

    with_types = element["content"].strip()
    without_types = util.delete_types(with_types).strip()

    if util.is_empty(without_types) or with_types == without_types:
        return None

    return (
        f"{COMMIT_BEFORE}{without_types}"
        f"{COMMIT_MSG}{instruction}"
        f"{COMMIT_AFTER}{with_types}"
    )

DATASET_CONFIG = DatasetConfig(
    get_content=get_content,
    streaming=True,
    size_valid_set=10_000,
    shuffle_buffer=5_000,
    seq_length=SEQUENCE_LENGTH,
)

def get_dataset(
    num_workers: int
) -> Dataset | IterableDataset:
    return load_dataset(
        "nuprl/ts-training",
        split="train",
        revision="v1.1p1",
        num_proc=num_workers if not DATASET_CONFIG.streaming else None,
        streaming=DATASET_CONFIG.streaming,
    )

"""
Edits should not be required after this point.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(TRAINING_ARGS.output_dir).mkdir(exist_ok=True)
    logging.set_verbosity_error()

    dataset = get_dataset(args.num_workers)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    train_dataset, eval_dataset = finetune.create_datasets(
        dataset,
        tokenizer,
        DATASET_CONFIG,
        args.seed
    )
    finetune.run_training(
        MODEL_PATH,
        TRAINING_ARGS,
        LORA_CONFIG,
        train_dataset,
        eval_dataset
    )

if __name__ == "__main__":
    main()
