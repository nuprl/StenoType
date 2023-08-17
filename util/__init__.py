from contextlib import contextmanager
from datetime import datetime
from datasets import Dataset, IterableDataset
from pathlib import Path
from typing import Optional
import datasets

ROOT_DIR = Path(Path(__file__).parent).parent

def load_dataset(
    dataset: str,
    split: Optional[str],
    revision: Optional[str],
    workers: Optional[int]
) -> Dataset | IterableDataset :
    """
    Load a dataset. Tries to interpret dataset as a path and loads a local file
    (in Parquet or JSON Lines format) or directory. If the path does not exist,
    load the dataset from the Hugging Face Hub.
    """
    if Path(dataset).exists():
        print(f"Loading dataset from {dataset} from disk...", flush=True)
        if dataset.endswith(".parquet"):
            return Dataset.from_parquet(dataset)
        elif dataset.endswith(".jsonl"):
            return Dataset.from_json(dataset)
        else:
            return datasets.load_from_disk(dataset)
    else:
        print(f"Loading dataset {dataset} from the Hugging Face Hub...", flush=True)
        return datasets.load_dataset(
            dataset,
            split=split,
            revision=revision,
            num_proc=workers
        )

def save_dataset(
    dataset: Dataset | IterableDataset,
    output: Optional[str],
    workers: int
) -> None:
    if not output:
        return

    print(f"Saving results to {output}")
    if output.endswith(".parquet"):
        dataset.to_parquet(output)
    elif output.endswith(".jsonl"):
        dataset.to_json(output)
    else:
        dataset.save_to_disk(output, num_proc=workers)

@contextmanager
def timer():
    start_time = datetime.now()
    yield
    end_time = datetime.now()
    duration = end_time - start_time
    total_seconds = round(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time: {hours}:{minutes:02}:{seconds:02}")
