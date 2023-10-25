from concurrent import futures
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
from datetime import datetime
from pathlib import Path
from subprocess import PIPE
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Any, Optional
import argparse
import functools
import pickle
import random
import re
import subprocess

from util import transform
import util

DEFAULT_CUTOFF = datetime(2021, 12, 31)

TSC_PATH = Path(util.ROOT_DIR, "ts", "node_modules", ".bin", "tsc").resolve()
if not Path(TSC_PATH).exists():
    print("Could not find tsc: {}".format(TSC_PATH))
    exit(1)


def parse_args() -> argparse.Namespace:
    cpu_count = util.cpu_count()
    # Dictionaries maintain insertion order
    valid_filters = dict.fromkeys(
        ["annotations", "types", "loc", "functions", "loc_per_function", "tokens"]
    )

    parser = argparse.ArgumentParser(
        "Extracts and processes TypeScript files from The Stack"
    )

    parser.add_argument("--dataset", type=str, help="directory to read dataset")
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="load dataset from Hugging Face, otherwise load from DATASET",
    )
    parser.add_argument("--checkpoint", type=str, help="file for saving checkpoints")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=1000,
        help="number of steps to save a checkpoint, defaults to 1000",
    )
    parser.add_argument("--output", type=str, help="directory to write dataset to")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use, defaults to {cpu_count}, "
        "the number of processors on the machine",
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="number or proportion (as a percentage) of elements to sample",
    )
    parser.add_argument(
        "--skim",
        action="store_true",
        help="browse through the dataset, one example at a time",
    )

    group = parser.add_argument_group(title="task to run")
    group.add_argument(
        "--parse", action="store_true", help="filter dataset for files that parse"
    )
    group.add_argument(
        "--typecheck",
        action="store_true",
        help="filter dataset for files that type check",
    )
    group.add_argument(
        "--metrics", action="store_true", help="add columns for dataset metrics"
    )
    group.add_argument(
        "--tokenize",
        action="store_true",
        help="add a column with estimated number of tokens",
    )
    group.add_argument(
        "--filter",
        type=str,
        help="comma-separated list of filters to apply, must be any of: "
        f"{','.join([f for f in valid_filters])}",
    )
    group.add_argument(
        "--cutoff",
        nargs="?",
        const=DEFAULT_CUTOFF.strftime("%Y-%m-%d"),
        help="filter dataset for files after the specified cutoff date, "
        f"in YYYY-MM-DD format; defaults to {DEFAULT_CUTOFF.strftime('%Y-%m-%d')}",
    )

    args = parser.parse_args()
    if not (args.dataset or args.from_hf):
        parser.print_usage()
        print("error: must provide --dataset or --from-hf")
        exit(2)
    elif args.dataset and args.from_hf:
        parser.print_usage()
        print("error: must provide only one of --dataset and --from-hf")
        exit(2)

    if args.dataset and not Path(args.dataset).exists():
        parser.print_usage()
        print(f"error: directory does not exist: {args.dataset}")
        exit(2)

    if args.filter:
        filters = args.filter.split(",")
        invalid_filters = [f for f in filters if f not in valid_filters]
        if invalid_filters:
            parser.print_usage()
            print(
                "error: the following filters are not valid: "
                f"{','.join(invalid_filters)}"
            )
            print("       must be any of: " f"{','.join([f for f in valid_filters])}")
            exit(2)

    if args.cutoff is not None:
        try:
            args.cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d")
        except ValueError:
            parser.print_usage()
            print("error: --cutoff argument must be in YYYY-MM-DD format")
            exit(2)

    if args.output and not (
        args.output.endswith(".parquet") or args.output.endswith(".jsonl")
    ):
        Path(args.output).mkdir(parents=True, exist_ok=True)

    return args


def load(from_hf: bool, dataset_dir: str, workers: int) -> Dataset | IterableDataset:
    if from_hf:
        print("Loading dataset from Hugging Face...", flush=True)
        return load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/typescript",
            split="train",
            revision="v1.1",
            num_proc=workers,
        )
    elif dataset_dir.endswith(".parquet"):
        print(f"Loading dataset from Parquet file ({dataset_dir})...", flush=True)
        return Dataset.from_parquet(dataset_dir)
    else:
        print(f"Loading dataset from disk ({dataset_dir})...", flush=True)
        return load_from_disk(dataset_dir)


def is_typescript(example: dict[str, Any]) -> bool:
    # Remove non-ts extensions (i.e. tsx extensions)
    if example["ext"] != "ts":
        return False

    content = example["content"]

    # Qt TS (translations), in XML format
    if "<!DOCTYPE TS>" in content:
        return False

    # TSurf data file
    if "GOCAD TSurf" in content:
        return False

    # Time series data
    if "@problemName" in content:
        return False

    return True


def filter_parses(
    dataset: Dataset | IterableDataset, workers: int
) -> Dataset | IterableDataset:
    print("Filtering for actual TypeScript files")
    dataset = dataset.filter(is_typescript, num_proc=workers)
    print("Number of TypeScript files:", len(dataset))

    print("Parsing files")
    dataset = dataset.filter(
        lambda e: transform.is_valid_syntax(e["content"]), num_proc=workers
    )
    print("Files that parse: ", len(dataset))

    return dataset


def load_checkpoint(checkpoint_file: str) -> dict[str, bool]:
    checkpoint = {}
    if checkpoint_file and Path(checkpoint_file).exists():
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
    return checkpoint


def save_checkpoint(
    checkpoint: dict[str, bool], checkpoint_file: str, message: Optional[str] = None
) -> None:
    if checkpoint_file:
        if message:
            print(message, flush=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)


def self_contained(content: str) -> bool:
    IMPORT_RE = re.compile("^\s*import(\*|\{|\s)")
    REQUIRE_RE = re.compile("^.*\s+require\s*\(.+\)")
    EXPORT_RE = re.compile("export[^\w_$]")
    EXPORT_RE2 = re.compile(
        "export[^\w_$]\s*(function|class|interface|type|"
        "enum|const|declare|default|async)"
    )
    matches = any(
        line
        for line in content.splitlines()
        if IMPORT_RE.match(line)
        or REQUIRE_RE.match(line)
        or (EXPORT_RE.match(line) and not EXPORT_RE2.match(line))
    )
    return not matches


def run_tsc(key: str, content: str) -> tuple[str, bool]:
    with NamedTemporaryFile(mode="w", suffix=".ts", encoding="utf-8") as f:
        # Save content to temp file
        print(content, file=f, end="", flush=True)
        tmpfile = Path(f.name)

        # Run tsc on temp file
        args = [str(TSC_PATH), "--noEmit", "--lib", "es2021", str(tmpfile)]
        result = subprocess.run(
            args, stdout=PIPE, stderr=PIPE, encoding="utf-8", cwd=tmpfile.parent
        )

        return key, result.returncode == 0


def filter_typechecks(
    dataset: Dataset | IterableDataset,
    checkpoint_file: str,
    checkpoint_steps: int,
    workers: int,
) -> Dataset | IterableDataset:
    # Load checkpoint file if it exists
    checkpoint = load_checkpoint(checkpoint_file)

    print("Filtering out import/export/require", flush=True)
    filtered = dataset.filter(lambda d: self_contained(d["content"]), num_proc=workers)

    if len(checkpoint) == 0:
        to_typecheck = filtered
    else:
        # This is slow, but setting up workers is even slower
        print("Already typechecked:", len(checkpoint), flush=True)
        to_typecheck = [
            f
            for f in tqdm(filtered, desc="Applying checkpoint")
            if f["hexsha"] not in checkpoint
        ]

    print("To typecheck:", len(to_typecheck), flush=True)
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        fs = [
            executor.submit(run_tsc, d["hexsha"], d["content"])
            for d in tqdm(to_typecheck, desc="Preparing workers")
        ]
        for i, f in enumerate(tqdm(fs, desc="Type checking", miniters=1)):
            key, result = f.result()
            checkpoint[key] = result
            if (i + 1) % checkpoint_steps == 0:
                save_checkpoint(
                    checkpoint, checkpoint_file, f"Saving checkpoint at step {i+1}"
                )

    save_checkpoint(checkpoint, checkpoint_file, "Saving final checkpoint")

    print("Filtering dataset for files that type check", flush=True)
    typechecks = filtered.filter(lambda d: checkpoint[d["hexsha"]], num_proc=workers)

    print("Original dataset size:", len(dataset))
    print("Filtered dataset size:", len(filtered))
    print("Type checks dataset size:", len(typechecks), flush=True)

    return typechecks


def count_annotation_sites(s: str) -> int:
    # Returns a list of strings, where each item is the substring between
    # annotation locations.
    chunks = transform.split_at_annotation_locations(s)
    # So we just need to subtract 1 from the length, to get the number of
    # annotation locations
    return len(chunks) - 1


def count_type_definitions(s: str) -> int:
    # Excludes classes and abstract classes
    captures = transform.run_query(
        s,
        """
        [
          (interface_declaration
            name: (type_identifier) @name)
          (type_alias_declaration
            name: (type_identifier) @name)
        ]
        """,
    )
    return len(captures)


def count_loc(s: str) -> int:
    # Delete types
    no_types = transform.delete_types(s)

    # Delete comments
    no_comments = transform.delete_comments(no_types)

    # Split string by newlines, delete empty lines
    lines = no_comments.split("\n")
    no_blanks = [line for line in lines if line.strip()]

    return len(no_blanks)


def count_functions(s: str) -> int:
    # Delete types
    no_types = transform.delete_types(s)

    captures = transform.run_query(
        no_types,
        """
        [
          (function_declaration) @func
          (function) @func
          (arrow_function) @func
          (method_definition) @func
        ]
        """,
    )
    return len(captures)


def loc_per_function(s: str) -> float:
    # Delete types
    no_types = transform.delete_types(s)

    captures = transform.run_query(
        no_types,
        """
        [
          (function_declaration
            body: (_) @body)
          (function
            body: (_) @body)
          (arrow_function
            body: (_) @body)
          (method_definition
            body: (_) @body)
        ]
        """,
    )
    nodes = [c[0] for c in captures]

    # Remove whitespace and open/close braces
    fun_loc = [count_loc(transform.node_to_str(n).strip().strip("{}")) for n in nodes]
    avg_loc = sum(fun_loc) / len(fun_loc) if fun_loc else 0.0

    return avg_loc


def add_metrics(example: dict[str, Any]) -> dict[str, Any]:
    content = example["content"]

    example["annotation_sites"] = count_annotation_sites(content)
    example["type_definitions"] = count_type_definitions(content)
    example["loc"] = count_loc(content)
    example["functions"] = count_functions(content)
    example["loc_per_function"] = loc_per_function(content)

    return example


def add_token_count(
    tokenizer: PreTrainedTokenizer, example: dict[str, Any]
) -> dict[str, Any]:
    token_count = len(tokenizer.encode(example["content"], add_special_tokens=True))
    example["estimated_tokens"] = token_count
    return example


def apply_filters(
    dataset: Dataset | IterableDataset, filter_list: str, workers: int
) -> Dataset | IterableDataset:
    filter_dict = {
        "annotations": ["annotation sites > 0", lambda d: d["annotation_sites"] > 0],
        "types": ["type definitions > 0", lambda d: d["type_definitions"] > 0],
        "loc": ["lines of code (without types) >= 50", lambda d: d["loc"] >= 50],
        "functions": ["number of functions > 0", lambda d: d["functions"] > 0],
        "loc_per_function": [
            "average lines of code per function >= 5",
            lambda d: d["loc_per_function"] >= 5,
        ],
        "tokens": ["total tokens <= 4096", lambda d: d["estimated_tokens"] <= 4096],
    }

    filters = filter_list.split(",")
    for f in filters:
        element = filter_dict[f]
        print(f"Applying filter: {f} ({element[0]})", flush=True)
        dataset = dataset.filter(element[1], num_proc=workers)
        print("Size after filtering:", len(dataset))

    return dataset


def parsedate(s: str) -> Optional[datetime]:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ") if s else None


def is_after_cutoff(example: dict[str, Any], cutoff: datetime) -> bool:
    stars_date = parsedate(example["max_stars_repo_stars_event_min_datetime"])
    forks_date = parsedate(example["max_forks_repo_forks_event_min_datetime"])
    issues_date = parsedate(example["max_issues_repo_issues_event_min_datetime"])

    timestamps = [t for t in [stars_date, forks_date, issues_date] if t is not None]

    if not timestamps:
        # If there is no timestamp, conservatively reject.
        # Affects about 10% of dataset.
        return False

    # We want ALL the timestamps to be after the cutoff
    return all(cutoff < t for t in timestamps)


def main():
    args = parse_args()

    dataset = load(args.from_hf, args.dataset, args.workers)
    print("Dataset size:", len(dataset))

    if args.parse:
        dataset = filter_parses(dataset, args.workers)

    if args.typecheck:
        dataset = filter_typechecks(
            dataset, args.checkpoint, args.checkpoint_steps, args.workers
        )

    if args.metrics:
        print("Adding metrics columns")
        dataset = dataset.map(add_metrics, num_proc=args.workers)

    if args.tokenize:
        from transformers import AutoTokenizer
        from transformers.utils import logging

        print("Adding estimated tokens column")
        # Supress warnings about token sequence length being too long
        logging.set_verbosity(logging.ERROR)
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
        dataset = dataset.map(
            lambda e: add_token_count(tokenizer, e), num_proc=args.workers
        )
        # Reset warning level
        logging.set_verbosity(logging.WARN)

    if args.filter:
        dataset = apply_filters(dataset, args.filter, args.workers)

    if args.cutoff:
        print(
            f"Filtering for files after the {args.cutoff.strftime('%Y-%m-%d')} "
            "cutoff",
            flush=True,
        )
        dataset = dataset.filter(
            functools.partial(is_after_cutoff, cutoff=args.cutoff),
            num_proc=args.workers,
        )
        print("Size after filtering:", len(dataset))

    if args.sample:
        sample = args.sample
        total = len(dataset)
        if sample.endswith("%"):
            pct = float(sample.rstrip("%")) / 100
            sample = round(total * pct)
        else:
            sample = int(sample)
        choices = random.sample(range(total), k=sample)
        dataset = dataset.select(choices)
        print("Size after sampling:", len(dataset))

    output = args.output
    if output:
        print("Saving result to", output, flush=True)
        if output.endswith(".parquet"):
            dataset.to_parquet(output)
        elif output.endswith(".jsonl"):
            dataset.to_json(output)
        else:
            dataset.save_to_disk(output, num_proc=args.workers)

    if args.skim:
        for d in dataset:
            print("===REPO===")
            print(d["max_stars_repo_name"], d["max_stars_repo_path"])
            print("===CONTENT===")
            print(d["content"])
            input("===EOF===")


if __name__ == "__main__":
    main()
