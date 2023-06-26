from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
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
import argparse

from finetune_lib import DatasetConfig
import finetune_lib as finetune

MODEL_PATH = str(Path(
    Path(__file__).parent,
    "..",
    "..",
    "models",
    "starcoderbase"
).resolve())

# TODO: multiply by 2 for the training format
TOTAL_TOKENS = 7_100_000_000
# TODO: calculate number of steps

# Arguments for the training loop
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
TRAINING_ARGS = TrainingArguments(
    output_dir="../checkpoints",
    # overwrite_output_dir, # TODO: may want this for resuming checkpoints
    evaluation_strategy="steps",
    per_device_train_batch_size=1, # TODO: should this be larger?
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    weight_decay=0.05,
    max_steps=10_000, # TODO, need to compute from other args and num_tokens
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=100, # TODO: maybe want a larger number
    save_steps=1000, # TODO: maybe want a larger number
    # save_total_limit, # TODO
    bf16=True,
    fp16=False,
    local_rank=0,
    dataloader_drop_last=True,
    eval_steps=100, # TODO: maybe want a larger number
    run_name="StarCoder-finetuned",
    report_to="wandb",
    ddp_find_unused_parameters=False,
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

def get_content(element: dict) -> str:
    """
    Given an input example, i.e. a string containing the contents of a
    TypeScript file, process it and return the updated example for training.

    Specifically, we process it into the StarCoder edit format, e.g.

        <commit_before>{code without types}
        <commit_msg>{instruction}
        <commit_after>{original code}

    """
    # COMMIT_BEFORE = "<commit_before>"
    # COMMIT_MSG = "<commit_msg>"
    # COMMIT_AFTER = "<commit_after>"

    # TODO
    return element["content"]

DATASET_CONFIG = DatasetConfig(
    get_content=get_content,
    streaming=True,
    size_valid_set=10_000,
    shuffle_buffer=5_000,
    seq_length=8192,
)

def get_dataset(
    num_workers: int
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    return load_dataset(
        "nuprl/ts-training",
        split="train",
        revision="v1.1p1",
        use_auth_token=True,
        num_proc=num_workers if not DATASET_CONFIG.streaming else None,
        streaming=DATASET_CONFIG.streaming,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(TRAINING_ARGS.output_dir).mkdir(exist_ok=True)
    logging.set_verbosity_error()

    dataset = get_dataset(args.num_workers)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=True)

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
