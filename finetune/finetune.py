from accelerate import Accelerator
from datasets import (Dataset,
                      DatasetDict,
                      IterableDataset,
                      IterableDatasetDict,
                      load_dataset)
from pathlib import Path
from peft import (LoraConfig,
                  get_peft_model,
                  prepare_model_for_int8_training,
                  set_peft_model_state_dict)
from torch.utils.data import IterableDataset as TorchIterableDataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          PreTrainedTokenizer,
                          Trainer,
                          TrainingArguments,
                          TrainerCallback,
                          TrainerControl,
                          TrainerState,
                          logging,
                          set_seed)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import argparse
import numpy as np
import scipy.stats as stats
import torch

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        checkpoint_folder = Path(args.output_dir,
                                 f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = Path(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control

class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        print(f"Loading best peft model from {state.best_model_checkpoint} "
              f"(score: {state.best_metric}).")
        best_model_path = Path(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

# TODO: refactor this into a config file
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="bigcode/starcoderbase")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/CodeAlpaca_20K")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=10000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--data_column", type=str, default="content")

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")

    # TODO: --bf16 will always be true and can't be unset
    parser.add_argument("--bf16", action="store_true", default=True)

    # TODO: --no_gradient_checkpointing will always be false and can't be set
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()

def tokenized_length(tokenizer: PreTrainedTokenizer, text: str) -> int:
    # Assuming NumPy tensors consume less memory than Python lists.
    return tokenizer(text, return_tensors="np")["input_ids"].shape[1]

def estimate_chars_per_token(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    tokenizer: AutoTokenizer,
    content_column: str,
    num_samples: int = 10_000,
    shuffle_buffer_size: int = 1_000,
    confidence_interval: float = 0.90,
) -> float:
    """
    Estimate the average number of characters per token in the dataset.
    """
    shuffled_dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # streaming=True produces an IterableDataset, which does not support
    # set_transform or select, which is why we have the explicit loop below.
    chars_per_token_list = []
    for count, item in tqdm(
        enumerate(shuffled_dataset), total=num_samples, desc="Estimating dataset size"
    ):
        if count == num_samples:
            break
        content = item[content_column]
        num_tokens = tokenized_length(tokenizer, content)
        chars_per_token_list.append(len(content) / num_tokens)

    chars_per_token_stats = stats.bootstrap(
        [chars_per_token_list], np.mean, confidence_level=confidence_interval
    )

    low = chars_per_token_stats.confidence_interval.low
    high = chars_per_token_stats.confidence_interval.high
    return round((low + high) / 2, 2)

def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || "
          f"all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param}")

class ConstantLengthDataset(TorchIterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream
    of text files.
        Args:
            tokenizer (PreTrainedTokenizer): The processor used for proccessing data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate
                number of tokens in text buffer.
            data_column (str): Column in the dataset with content to read.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
        infinite: bool = False,
        seq_length: int = 8192,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
        data_column: str = "content",
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.infinite = infinite
        self.seq_length = seq_length
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.data_column = data_column

    def _buffer_generator(self):
        """
        Iterate over the provided dataset: append to a buffer and yield the
        buffer once it is large enough.
        """
        buffer, buffer_len = [], 0
        while True:
            for e in self.dataset:
                # If buffer hits max size, yield and then reset buffer
                if buffer_len >= self.max_buffer_size:
                    yield buffer
                    buffer, buffer_len = [], 0
                # Add content from dataset to the buffer
                to_add = e[self.data_column]
                buffer.append(to_add)
                buffer_len += len(to_add)

            if not self.infinite:
                # If end of dataset reached before end of buffer, and we aren't
                # looping infinitely, yield buffer
                yield buffer
                break

    def __iter__(self):
        """
        Iterate over buffers: for each buffer, tokenize the inputs and pack
        them and yield one token sequence (of length seq_length) at a time.
        """
        for buffer in self._buffer_generator():
            # Tokenize the buffer
            tokenized_inputs = self.tokenizer(buffer)["input_ids"]
            all_token_ids = []

            # Pack the inputs, separating them with self.concat_token_id
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            # Iterate through all tokens, seq_length at a time, so we can yield
            # a sequence of length seq_length
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                # Only yield if we have seq_length tokens
                if len(input_ids) == self.seq_length:
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }

# TODO: maybe some of this could be refactored, we can't always just give the
# dataset; some processing might be required
def create_datasets(tokenizer, args):
    """
    Loads the dataset, creates train/valid splits, and transforms
    into ConstantLengthDataset.
    """
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. "
              f"Size of the validation set: {len(valid_data)}")

    chars_per_token = estimate_chars_per_token(train_data,
                                               tokenizer,
                                               args.data_column)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        data_column=args.data_column,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        data_column=args.data_column,
    )
    return train_dataset, valid_dataset

# TODO: need to resume from checkpoint
def run_training(args, train_data, val_data):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["c_proj", "c_attn", "q_attn"]
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback]
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(Path(args.output_dir, "final_checkpoint/"))

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)

if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    Path(args.output_dir).mkdir(exist_ok=True)

    logging.set_verbosity_error()

    main(args)
