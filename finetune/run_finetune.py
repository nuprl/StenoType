from datasets import Dataset, IterableDataset
from pathlib import Path
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, logging, set_seed
from typing import Optional
import argparse
import os

from finetune_lib import DatasetConfig
import finetune_lib as finetune
import get_training_example
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

MODEL_PATH = str(
    Path(Path(__file__).parent, "..", "..", "models", "starcoderbase-7b").resolve()
)

# We are using a very large dataset, so it's not feasible to download the whole
# thing. Instead, we stream the dataset and use an estimate for the number of
# tokens. This estimate was derived (in a separate script) by computing the size
# of the dataset (just the training columns) and estimating the bytes per token
# ratio.

# The original dataset has 7.1B tokens, but we multiply by 2 to account for
# the training format
TOTAL_TOKENS = 7_100_000_000 * 2

########## StarCoder-1B on an A100/H100
# We pack the tokens into a ConstantLengthDataset,
# where each example has SEQUENCE_LENGTH tokens
# SEQUENCE_LENGTH = 8 * 1024
# EPOCHS = 1
# BATCH_SIZE = 2
# GRADIENT_ACCUMULATION_STEPS = 4

# Roughly 1.7M examples
# Roughly 217K / NUM_GPUS steps
########## StarCoder-1B on an A100/H100

########## StarCoder-7B on an A100 with LoRA
# We pack the tokens into a ConstantLengthDataset,
# where each example has SEQUENCE_LENGTH tokens
SEQUENCE_LENGTH = 8 * 1024
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

# Roughly 1.7M examples
# Roughly 433K / NUM_GPUS steps
########## StarCoder-7B on an A100 with LoRA

NUM_EXAMPLES = TOTAL_TOKENS // SEQUENCE_LENGTH

# LOCAL_WORLD_SIZE is --nproc-per-node when using torchrun,
# i.e., the number of processes/GPUs to use on a single node (machine)
if "LOCAL_WORLD_SIZE" in os.environ:
    NUM_GPUS = int(os.environ["LOCAL_WORLD_SIZE"])
else:
    NUM_GPUS = 1

MAX_STEPS = (EPOCHS * NUM_EXAMPLES) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
MAX_STEPS = MAX_STEPS // NUM_GPUS

# Arguments for the training loop
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
TRAINING_ARGS = TrainingArguments(
    output_dir="../checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=2.5e-5,
    weight_decay=0.05,
    max_steps=1000,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=1,
    save_steps=250,  # save_steps must be a multiple of eval_steps
    bf16=True,
    fp16=False,
    local_rank=0,
    dataloader_drop_last=True,
    eval_steps=50,  # save_steps must be a multiple of eval_steps
    run_name="StarCoder-finetuned",
    load_best_model_at_end=True,  # needed for LoRA callbacks
    optim="adamw_torch",
    report_to="wandb",
    ddp_find_unused_parameters=False,
    resume_from_checkpoint=False,  # only set to True if there is an existing checkpoint!
    gradient_checkpointing=True,
)

# If not using LoRA, set to None
# LORA_CONFIG: Optional[LoraConfig] = None
LORA_CONFIG: Optional[LoraConfig] = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_proj", "c_attn", "q_attn"],
)

DATASET_CONFIG = DatasetConfig(
    get_content=get_training_example.default,
    streaming=True,
    size_valid_set=10_000,
    shuffle_buffer=5_000,
    seq_length=SEQUENCE_LENGTH,
)


def get_dataset(num_workers: int) -> Dataset | IterableDataset:
    dataset = util.load_dataset("../../datasets/ts-training-get4")
    return dataset.to_iterable_dataset(num_shards=100)


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
        dataset, tokenizer, DATASET_CONFIG, args.seed
    )
    finetune.run_training(
        MODEL_PATH, TRAINING_ARGS, LORA_CONFIG, train_dataset, eval_dataset
    )


if __name__ == "__main__":
    main()
