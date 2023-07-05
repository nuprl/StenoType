from accelerate import Accelerator
from collections.abc import Callable
from dataclasses import dataclass
from datasets import (
    Dataset,
    IterableDataset,
)
from pathlib import Path
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Optional
import torch

def default_get_content(element: dict) -> Optional[str]:
    return element["content"]

@dataclass
class DatasetConfig:
    """
    Configuration for loading (and processing) the dataset.
        Args:
            get_content (Callable[[dict], Optional[str]]): Function that
                extracts text from a dataset element, to be used for training.
                Returns None if the text is invalid and should be skipped.
            streaming (`bool`): If set to `True`, streams the dataset instead of
                downloading the entire dataset. Defaults to `False`.
            size_valid_set (`int`): If streaming, take this many elements from the
                dataset to use for the validation set. Defaults to `10_000`.
            shuffle_buffer (`int`): Size of the shuffle buffer. Defaults to `1000`.
            seq_length (`int`): Length of token sequences to use for the
                `ConstantLengthDataset`. Defaults to `2048`.
    """
    get_content: Callable[[dict], Optional[str]] = default_get_content
    streaming: bool = False
    size_valid_set: int = 10_000
    shuffle_buffer: int = 1000
    seq_length: int = 2048

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        checkpoint_folder = Path(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
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
        print(
            f"Loading best peft model from {state.best_model_checkpoint} "
            f"(score: {state.best_metric})."
        )
        best_model_path = Path(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

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
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )

class ConstantLengthDataset(TorchIterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from a dataset.
        Args:
            tokenizer (PreTrainedTokenizer): The processor used for proccessing data.
            dataset (dataset.Dataset): Dataset with text files.
            seq_length (int): Length of token sequences to return.
            get_content (Callable[[dict], Optional[str]]): Function that
                extracts text from a dataset element, to be used for training.
            infinite (bool): If True the iterator is reset after dataset reaches end.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset | IterableDataset,
        seq_length: int,
        get_content: Callable[[dict], Optional[str]],
        infinite: bool = False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.get_content = get_content
        self.infinite = infinite

    def _return_tensor(self, input_ids, attention_mask):
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask
        }

    def _pad(self, buffer: list):
        """
        Given a buffer that is shorter than self.seq_length, pad it with
        self.concat_token_id and create an attention mask that indicates the
        padded tokens.
        """
        assert (len(buffer) > 0 and len(buffer) < self.seq_length)

        pad_size = self.seq_length - len(buffer)
        padded_input_ids = torch.cat([
            torch.LongTensor(buffer),
            torch.full(
                size=(pad_size,),
                fill_value=self.concat_token_id,
                dtype=torch.long
            )
        ])
        attention_mask = torch.cat([
            torch.ones(len(buffer), dtype=torch.long),
            torch.zeros(pad_size, dtype=torch.long)
        ])

        yield self._return_tensor(padded_input_ids, attention_mask)

    def __iter__(self):
        """
        Iterate over the provided dataset. For each dataset element, extract the
        input text (which may simply take the content from one column, or
        transform the content, or use multiple columns), tokenize, append the
        concat token, and add to the buffer.

        Yield slices of length self.seq_length from the buffer.
        """
        max_attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        buffer = []
        while True:
            for e in self.dataset:
                content = self.get_content(e)

                # Skip over this element if the content is not valid
                if not content:
                    continue

                tokens = self.tokenizer(
                    content,
                    return_attention_mask=False
                )["input_ids"]
                buffer.extend(tokens + [self.concat_token_id])

                # If buffer exceeds max size, yield sequences of length
                # self.seq_length from the buffer, then delete those tokens
                # from the buffer
                while len(buffer) >= self.seq_length:
                    input_ids = torch.LongTensor(buffer[:self.seq_length])
                    del buffer[:self.seq_length]
                    yield self._return_tensor(input_ids, max_attention_mask)

            if not self.infinite:
                if len(buffer) > 0:
                    yield from self._pad(buffer)
                break

def create_datasets(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizer,
    config: DatasetConfig,
    seed: int
) -> tuple[ConstantLengthDataset, ConstantLengthDataset]:
    """
    Given a dataset, create the train/valid splits and transform into
    ConstantLengthDataset.
    """
    if config.streaming:
        print("Loading the dataset in streaming mode")
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer, seed=seed)
        valid_data = dataset.take(config.size_valid_set)
        train_data = dataset.skip(config.size_valid_set)
        train_data = train_data.shuffle(
            buffer_size=config.shuffle_buffer,
            seed=seed
        )
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. "
            f"Size of the validation set: {len(valid_data)}"
        )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        dataset=train_data,
        seq_length=config.seq_length,
        get_content=config.get_content,
        infinite=True,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        dataset=valid_data,
        seq_length=config.seq_length,
        get_content=config.get_content,
        infinite=False,
    )
    return train_dataset, valid_dataset

def run_training(
    model_path: str,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
    train_data: TorchIterableDataset,
    val_data: TorchIterableDataset,
):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=not training_args.gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_kbit_training(model)
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
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    model.save_pretrained(Path(training_args.output_dir, "final_checkpoint/"))
