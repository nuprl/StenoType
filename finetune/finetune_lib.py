from accelerate import Accelerator
from dataclasses import dataclass
from datasets import (Dataset,
                      DatasetDict,
                      IterableDataset,
                      IterableDatasetDict)
from pathlib import Path
from peft import (LoraConfig,
                  get_peft_model,
                  prepare_model_for_int8_training,
                  set_peft_model_state_dict)
from torch.utils.data import IterableDataset as TorchIterableDataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          PreTrainedTokenizer,
                          Trainer,
                          TrainingArguments,
                          TrainerCallback,
                          TrainerControl,
                          TrainerState)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import argparse
import numpy as np
import scipy.stats as stats
import torch

@dataclass
class DatasetConfig:
    """
    Configuration for loading (and processing) the dataset.

    Args:
        streaming (`bool`): If set to `True`, streams the dataset instead of
            downloading the entire dataset. Defaults to `False`.
        size_valid_set (`int`): If streaming, take this many elements from the dataset
            to use for the validation set. Defaults to `10_000`.
        shuffle_buffer (`int`): Size of the shuffle buffer. Defaults to `1000`.
        data_column (`str`): Column of the dataset to use. Defaults to `"content"`.
        seq_length (`int`): Length of token sequences to use for the
            `ConstantLengthDataset`. Defaults to `2048`.
    """
    streaming: bool = False
    size_valid_set: int = 10_000
    shuffle_buffer: int = 1000
    data_column: str = "content"
    seq_length: int = 2048

class WrappedTokenizer:
    """
    A wrapper class for PreTrainedTokenizer, that transforms the input text
    before calling the tokenizer.

    This is useful if you need to transform the input examples in your dataset
    (for training or fine-tuning) and don't want to preprocess the entire dataset.
    For this use case, subclass WrappedTokenizer and override the `transform` method.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str, *args, **kwargs):
        """Call the underlying tokenizer on the transformed text."""
        return self.tokenizer(self.transform(text), *args, **kwargs)

    @property
    def unwrap(self):
        """Return the underlying tokenizer."""
        return self.tokenizer

    def transform(self, text: str) -> str:
        """
        Transform the input text.

        For WrappedTokenizer, this is simply the identity function.
        """
        return text

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

def tokenized_length(tokenizer: WrappedTokenizer, text: str) -> int:
    # Assuming NumPy tensors consume less memory than Python lists.
    return tokenizer(text, return_tensors="np")["input_ids"].shape[1]

def estimate_chars_per_token(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    tokenizer: WrappedTokenizer,
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
            tokenizer (WrappedTokenizer): The processor used for proccessing data.
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
        tokenizer: WrappedTokenizer,
        dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
        seq_length: int,
        chars_per_token: float,
        data_column: str,
        num_of_sequences: int = 1024,
        infinite: bool = False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.unwrap.eos_token_id
        self.dataset = dataset
        self.infinite = infinite
        self.seq_length = seq_length
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.data_column = data_column

    # TODO: this isn't right, probably want to tokenize first (since we need to
    # transform), and then add the concat_token in between
    # Right now we're packing everything and only adding the concat_token between
    # buffers, not between inputs
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

def create_datasets(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    tokenizer: WrappedTokenizer,
    config: DatasetConfig,
    seed: int
) -> tuple[ConstantLengthDataset, ConstantLengthDataset]:
    """
    Given a dataset, create the train/valid splits and transform into
    ConstantLengthDataset.
    """
    if config.streaming:
        print("Loading the dataset in streaming mode")
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer,
                                  seed=seed)
        valid_data = dataset.take(config.size_valid_set)
        train_data = dataset.skip(config.size_valid_set)
        train_data = train_data.shuffle(buffer_size=config.shuffle_buffer,
                                        seed=seed)
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. "
              f"Size of the validation set: {len(valid_data)}")

    chars_per_token = estimate_chars_per_token(train_data,
                                               tokenizer,
                                               config.data_column)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        dataset=train_data,
        seq_length=config.seq_length,
        chars_per_token=chars_per_token,
        data_column=config.data_column,
        infinite=True,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        dataset=valid_data,
        seq_length=config.seq_length,
        chars_per_token=chars_per_token,
        data_column=config.data_column,
        infinite=False,
    )
    return train_dataset, valid_dataset

# TODO: need to resume from checkpoint
def run_training(
    args: argparse.Namespace,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
    train_data: TorchIterableDataset,
    val_data: TorchIterableDataset,
):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_auth_token=True,
        use_cache=not training_args.gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    print("Starting main loop")
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
    model.save_pretrained(Path(training_args.output_dir, "final_checkpoint/"))
