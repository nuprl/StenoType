from pathlib import Path
import argparse

from experiment import ExperimentType, run_experiment
import util

def parse_args() -> argparse.Namespace:
    cpu_count = util.cpu_count()

    parser = argparse.ArgumentParser(
        description="Runs StarCoder to infer types for JavaScript")

    parser.add_argument(
        "--models_directory",
        type=str,
        default="../models",
        help="directory to load models from")
    parser.add_argument(
        "--results_directory",
        type=str,
        default="results",
        help="directory to save results to")
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="port for the model server")
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

    if not args.view and not args.devices:
        print("error: the following arguments are required: --devices")
        exit(2)

    if args.devices:
        models_directory = Path(args.models_directory).resolve()
        args.models_directory = str(models_directory)
        if not models_directory.exists():
            print("error: cannot find models directory:", models_directory)

        results_directory = Path(args.results_directory).resolve()
        args.results_directory = str(results_directory)
        if not results_directory.exists():
            print("error: cannot find results directory:", results_directory)

        if not models_directory.exists() or not results_directory.exists():
            exit(2)

    return args

def main():
    args = parse_args()

    # Early return, just view the (results) dataset
    if args.view:
        dataset = util.load_dataset(args.view)
        for i, d in enumerate(dataset):
            util.print_result(d, i)
        return

    # TODO: Right now we only have one evaluation dataset
    dataset = util.load_dataset("../datasets/stenotype-eval-dataset-subset")

    # run_experiment(dataset, "starcoderbase-1b", ExperimentType.APPROACH_1, args)
    # run_experiment(
    #     dataset, "stenotype-75ce914-ckpt100", ExperimentType.APPROACH_1, args
    # )
    run_experiment(
        dataset, "stenotype-54d5802-ckpt100", ExperimentType.APPROACH_2, args
    )
    run_experiment(
        dataset, "stenotype-2b77ede-ckpt100", ExperimentType.APPROACH_3, args
    )

if __name__ == "__main__":
    main()
