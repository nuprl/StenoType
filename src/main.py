from pathlib import Path
import argparse
import datasets

from evaluation import run_evaluation, summarize_results
from experiment import ExperimentConfig, run_experiment
import experiment
import util


def parse_args() -> argparse.Namespace:
    cpu_count = util.cpu_count()

    parser = argparse.ArgumentParser(description="Evaluation runner for StenoType")

    parser.add_argument(
        "--models_directory",
        type=str,
        default="../models",
        help="directory to load models from; defaults to ../models",
    )
    parser.add_argument(
        "--results_directory",
        type=str,
        default="results",
        help="directory to save results to; defaults to ./results",
    )
    parser.add_argument(
        "--num_completions",
        type=int,
        default=20,
        help="number of completions to generate for each problem",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use; defaults to {cpu_count}",
    )

    group = parser.add_argument_group(title="task to run")
    group.add_argument(
        "--infer",
        nargs="*",
        metavar="CONFIG",
        help="run inference (on the configurations indexed by [CONFIG ...], "
        "or all configurations if no indices given)",
    )
    group.add_argument("--evaluate", action="store_true", help="evaluate results")
    group.add_argument("--summarize", action="store_true", help="summarize results")
    group.add_argument(
        "--show_configs",
        action="store_true",
        help="print configuration indices (to be used with --infer)",
    )

    args = parser.parse_args()

    if args.infer is not None and any(not x.isdigit() for x in args.infer):
        print("error: provided indices must be integers")
        exit(2)

    if args.infer is not None or args.evaluate:
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

    if (
        args.infer is None
        and not args.evaluate
        and not args.summarize
        and not args.show_configs
    ):
        parser.print_usage()
        print("error: must provide one of --infer, --evaluate, --summarize")
        exit(2)

    return args


def main():
    # Don't cache datasets
    datasets.disable_caching()

    args = parse_args()

    # TODO: Right now we only have one evaluation dataset
    # maybe require all datasets to be on disk
    # also, this loads the dataset even when we don't need it (e.g. for --evaluate)
    dataset = util.load_dataset("../datasets/stenotype-eval-dataset-subset")

    configs = [
        ExperimentConfig(dataset, "starcoderbase-1b", experiment.approach1),
        ExperimentConfig(dataset, "starcoderbase-7b", experiment.approach1),
        ExperimentConfig(dataset, "stenotype-1b-75ce914-ckpt100", experiment.approach1),
        ExperimentConfig(dataset, "stenotype-1b-54d5802-ckpt100", experiment.approach2),
        ExperimentConfig(dataset, "stenotype-1b-2b77ede-ckpt100", experiment.approach3),
        ExperimentConfig(dataset, "stenotype-1b-7904b4a-ckpt200", experiment.approach1),
        ExperimentConfig(dataset, "stenotype-1b-7904b4a-ckpt600", experiment.approach1),
        ExperimentConfig(
            dataset, "stenotype-1b-7904b4a-ckpt1000", experiment.approach1
        ),
        ExperimentConfig(dataset, "stenotype-1b-1753dc0-ckpt200", experiment.approach3),
        ExperimentConfig(dataset, "stenotype-1b-1753dc0-ckpt600", experiment.approach3),
        ExperimentConfig(
            dataset, "stenotype-1b-1753dc0-ckpt1000", experiment.approach3
        ),
        ExperimentConfig(dataset, "starcoderbase-1b-approach4", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-1b-ef65cb9-ckpt250", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-1b-ef65cb9-ckpt500", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-1b-ef65cb9-ckpt750", experiment.approach4),
        ExperimentConfig(
            dataset, "stenotype-1b-ef65cb9-ckpt1000", experiment.approach4
        ),
        ExperimentConfig(dataset, "starcoderbase-7b-approach4", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-7b-a6d445d-ckpt250", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-7b-a6d445d-ckpt500", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-7b-a6d445d-ckpt750", experiment.approach4),
        ExperimentConfig(dataset, "stenotype-7b-a6d445d-ckpt1000", experiment.approach4),
    ]

    if args.show_configs:
        for i, c in enumerate(configs):
            print(i, c.model_name, c.approach.__name__)
        exit(0)

    if args.infer is not None:
        # If indices given, then select only those configs
        if args.infer:
            configs = [configs[int(i)] for i in args.infer]

        # Make sure results don't already exist
        results_paths = [
            util.get_results_name(c.model_name, args.results_directory) for c in configs
        ]
        results_exists = [path for path in results_paths if Path(path).exists()]
        for p in results_exists:
            print(f"error: output {p} already exists, please delete or rename!")
        if results_exists:
            exit(2)

    for c in configs:
        if args.infer is not None:
            run_experiment(c, args)
        if args.evaluate:
            run_evaluation(c, args)

    if args.summarize:
        summarize_results(configs, args)


if __name__ == "__main__":
    main()
