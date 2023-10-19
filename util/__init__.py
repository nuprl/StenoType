from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from datasets import Dataset, IterableDataset
from pathlib import Path
from typing import Any, Generator, Optional, Type
import datasets
import json
import numpy as np
import os

ROOT_DIR = Path(Path(__file__).parent).parent

def cpu_count() -> int:
    # os.cpu_count() is the number of CPUs on the system,
    # not the number available to the current process
    return len(os.sched_getaffinity(0))

def load_dataset(
    dataset: str,
    split: Optional[str] = None,
    revision: Optional[str] = None,
    workers: Optional[int] = None
) -> Dataset | IterableDataset:
    """
    Load a dataset. Tries to interpret dataset as a path and loads a local file
    (in Parquet or JSON Lines format) or directory. If the path does not exist,
    load the dataset from the Hugging Face Hub.
    """
    if Path(dataset).exists():
        print(f"Loading dataset {dataset} from disk...", flush=True)
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
    output: str,
    workers: int
):
    print(f"Saving results to {output}")
    if output.endswith(".parquet"):
        dataset.to_parquet(output)
    elif output.endswith(".jsonl"):
        dataset.to_json(output)
    else:
        dataset.save_to_disk(output, num_proc=workers)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def read_jsonl(path: Path | str) -> Generator[dict, None, None]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(
        path: Path | str,
        data: Iterable[Any],
        encoder: Optional[Type[json.JSONEncoder]] = None
):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, cls=encoder) + "\n")

def get_results_name(model_name: str, results_directory: str) -> str:
    return str(Path(results_directory, model_name).with_suffix(".parquet"))

def get_model_path(model_name: str, models_directory: str) -> str:
    return str(Path(models_directory, model_name))

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
