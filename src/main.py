from pathlib import Path
import argparse

from experiment import ExperimentConfig, ExperimentType, run_experiment
import util

def parse_args() -> argparse.Namespace:
    cpu_count = util.cpu_count()

    parser = argparse.ArgumentParser(
        description="Evaluation runner for StenoType")

    parser.add_argument(
        "--models_directory",
        type=str,
        default="../models",
        help="directory to load models from; defaults to ../models")
    parser.add_argument(
        "--results_directory",
        type=str,
        default="results",
        help="directory to save results to; defaults to ./results")
    parser.add_argument(
        "--num_completions",
        type=int,
        default=20,
        help="number of completions to generate for each problem")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use; defaults to {cpu_count}")

    group = parser.add_argument_group(title="task to run")
    group.add_argument(
        "--infer",
        action="store_true",
        help="run inference")
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate and summarize results")
    group.add_argument(
        "--view",
        type=str,
        metavar="DATASET",
        help="browse through DATASET, one example at a time")

    args = parser.parse_args()

    if args.infer or args.evaluate:
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

    if not args.infer and not args.evaluate and not args.view:
        print("error: must provide one of --infer, --evaluate, --view")
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
    # maybe require all datasets to be on disk
    dataset = util.load_dataset("../datasets/stenotype-eval-dataset-subset")

    # Sometimes the dataset gets cached and nothing happens
    dataset.cleanup_cache_files()

    # TODO: maybe try multiple processes (data parallelism)
    configs = [
        ExperimentConfig(
            dataset,
            "starcoderbase-1b",
            ExperimentType.APPROACH_1),
        ExperimentConfig(
            dataset,
            "stenotype-75ce914-ckpt100",
            ExperimentType.APPROACH_1),
        ExperimentConfig(
            dataset,
            "stenotype-54d5802-ckpt100",
            ExperimentType.APPROACH_2),
        ExperimentConfig(
            dataset,
            "stenotype-2b77ede-ckpt100",
            ExperimentType.APPROACH_3),
    ]

    if args.infer:
        # Make sure results don't already exist
        results_paths = [util.get_results_name(c.model_name, args.results_directory)
                            for c in configs]
        results_exists = [path for path in results_paths if Path(path).exists()]
        for p in results_exists:
            print(f"error: output {p} already exists, please delete or rename!")
        if results_exists:
            exit(2)

    for c in configs:
        if args.infer:
            run_experiment(c, args)
        if args.evaluate:
            # TODO
            pass

if __name__ == "__main__":
    main()
