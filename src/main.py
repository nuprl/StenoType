from pathlib import Path
import argparse
import os

from experiment import run_experiment
import util

def parse_args() -> argparse.Namespace:
    # os.cpu_count() is the number of CPUs on the system,
    # not the number available to the current process
    cpu_count = len(os.sched_getaffinity(0))

    parser = argparse.ArgumentParser(
        description="Runs StarCoder to infer types for JavaScript")

    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port for the model server")
    parser.add_argument(
        "--devices",
        type=str,
        help="GPU devices to use")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use; defaults to {cpu_count}")
    parser.add_argument(
        "--view",
        type=str,
        help="browse through the given dataset, one example at a time")

    args = parser.parse_args()

    if not args.skim:
        if not args.devices:
            print("error: the following arguments are required: --devices")
            exit(2)

    output = args.output
    if output:
        if Path(output).exists():
            print(f"Output path {output} already exists, please delete, rename, or "
                  "choose a different output path!")
            exit(2)
        elif not (output.endswith(".parquet") or output.endswith(".jsonl")):
            Path(output).mkdir(parents=True, exist_ok=True)

    return args

def main():
    args = parse_args()

    # Early return, just view the (results) dataset
    if args.view:
        dataset = util.load_dataset(args.view)
        for i, d in enumerate(dataset):
            util.print_result(d, i)
        return

    # Right now we only have one evaluation dataset
    dataset = util.load_dataset(
        dataset="nuprl/ts-eval",
        split="test",
        revision="v1.1subset",
        workers=args.workers
    )

    # TODO: this kind of config only allows changing dataset and model
    # we might want to change how the inference is run
    # so maybe implement experiment as a class, so methods can be overridden
    run_experiment(dataset, "starcoderbase-1b", args)
    run_experiment(dataset, "stenotype-4b0794e", args)
    run_experiment(dataset, "stenotype-b476aae", args)
    run_experiment(dataset, "stenotype-381d16d", args)

if __name__ == "__main__":
    main()
