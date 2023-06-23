from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetConfig:
    """
    Configuration for loading (and processing) the dataset.

    Args:
        path (`str`): Path or name of dataset
        data_dir (`str`, optional): Data directory to load
        split (`str`): Dataset split to load. Defaults to `train`.
        revision (`str`, optional): Dataset revision. Defaults to `None`, i.e. the
            main branch.
        streaming (`bool`): If set to `True`, streams the dataset instead of
            downloading the entire dataset. Defaults to `False`.
        size_valid_set (`int`): If streaming, take this many elements from the dataset
            to use for the validation set. Defaults to `10_000`.
        shuffle_buffer (`int`): Size of the shuffle buffer. Defaults to `1000`.
        data_column (`str`): Column of the dataset to use. Defaults to `"content"`.
        seq_length (`int`): Length of token sequences to use for the
            `ConstantLengthDataset`. Defaults to `2048`.
    """
    path: str
    data_dir: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    streaming: bool = False
    size_valid_set: int = 10_000
    shuffle_buffer: int = 1000
    data_column: str = "content"
    seq_length: int = 2048
