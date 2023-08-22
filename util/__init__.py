from contextlib import contextmanager
from datetime import datetime
from datasets import Dataset, IterableDataset
from pathlib import Path
from typing import Any, Optional
import datasets
import difflib

ROOT_DIR = Path(Path(__file__).parent).parent

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

def print_diff(
    before: str,
    after: str,
    fromfile: str = "before",
    tofile: str = "after",
    color: bool = False
) -> None:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[39m"

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines, after_lines, fromfile=fromfile, tofile=tofile
    )

    for d in diff:
        if color:
            if d.startswith("@@"):
                print(YELLOW + d, end=RESET)
            elif d.startswith("+"):
                print(GREEN + d, end=RESET)
            elif d.startswith("-"):
                print(RED + d, end=RESET)
        else:
            print(d, end="")
    print()

def print_result(example: dict[str, Any], i: Optional[int] = None) -> None:
    name = example["max_stars_repo_name"] + " " + example["max_stars_repo_path"]
    original_code = example["content"]
    input_code = example["content_without_types"]
    output_code = example["output"]
    index = f"{i} " if i is not None else ""

    print("===REPO===")
    print(index + name)
    print("===ORIGINAL===")
    print(original_code)
    print("===INPUT===")
    print(input_code)
    print("===OUTPUT===")
    print(output_code)
    print("===DIFF ORIGINAL/OUTPUT===")
    print_diff(
        original_code,
        output_code,
        fromfile="original",
        tofile="output",
        color=True
    )
    print("===DIFF INPUT/OUTPUT===")
    print_diff(
        input_code,
        output_code,
        fromfile="input",
        tofile="output",
        color=True
    )
    print("===RESULTS===")
    print(f"Accuracy {example['accuracy']:.2%}\n"
            f"Levenshtein {example['levenshtein']:.2%}\n"
            f"Type errors {example['type_errors']}\n"
            f"Parse errors {example['parse_errors']}")
    print("===REPO===")
    print(index + name)
    input("===EOF===")
