from pathlib import Path
import argparse
import datasets
import os

from evaluation import run_evaluation
from inference import (
    DatasetConfig as DatasetConf,
    Config,
    run_inference,
)
from summarize import summarize
import inference
import util


def parse_args() -> argparse.Namespace:
    cpu_count = util.cpu_count()
    models_path = Path(util.ROOT_DIR, "..", "models").resolve()
    datasets_path = Path(util.ROOT_DIR, "..", "datasets").resolve()
    results_path = Path(util.ROOT_DIR, "results").resolve()
    models_relpath = os.path.relpath(models_path)
    datasets_relpath = os.path.relpath(datasets_path)
    results_relpath = os.path.relpath(results_path)

    parser = argparse.ArgumentParser(description="Evaluation runner for StenoType")

    parser.add_argument(
        "--models_directory",
        type=str,
        default=models_path,
        help=f"directory to load models from; defaults to {models_relpath}",
    )
    parser.add_argument(
        "--datasets_directory",
        type=str,
        default=datasets_path,
        help=f"directory to load models from; defaults to {datasets_relpath}",
    )
    parser.add_argument(
        "--results_directory",
        type=str,
        default=results_path,
        help=f"directory to save results to; defaults to {results_relpath}",
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
    group.add_argument(
        "--evaluate",
        nargs="*",
        metavar="CONFIG",
        help="run evaluation (on the configurations indexed by [CONFIG ...], "
        "or all configurations if no indices given)",
    )
    group.add_argument("--summarize", action="store_true", help="summarize results")
    group.add_argument(
        "--show_configs",
        action="store_true",
        help="print configuration indices (to be used with --infer and --evaluate)",
    )

    args = parser.parse_args()

    if (args.infer is not None and any(not x.isdigit() for x in args.infer)) or (
        args.evaluate is not None and any(not x.isdigit() for x in args.evaluate)
    ):
        print("error: provided indices must be integers")
        exit(2)

    if args.infer is not None or args.evaluate is not None:
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
        and args.evaluate is None
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
    configs = []

    # TypeScript datasets
    ts_dataset = DatasetConf(
        short_name="ts",
        datasets_path=args.datasets_directory,
        dataset_name="stenotype-eval-dataset-subset",
    )
    configs += [
        Config("starcoderbase-1b", inference.approach1, ts_dataset),
        Config("stenotype-1b-75ce914-ckpt100", inference.approach1, ts_dataset),
        Config("stenotype-1b-54d5802-ckpt100", inference.approach2, ts_dataset),
        Config("stenotype-1b-2b77ede-ckpt100", inference.approach3, ts_dataset),
        Config("starcoderbase-1b", inference.approach4, ts_dataset),
        Config("stenotype-1b-ef65cb9-ckpt250", inference.approach4, ts_dataset),
        Config("stenotype-1b-ef65cb9-ckpt500", inference.approach4, ts_dataset),
        Config("stenotype-1b-ef65cb9-ckpt750", inference.approach4, ts_dataset),
        Config("stenotype-1b-ef65cb9-ckpt1000", inference.approach4, ts_dataset),
        Config("starcoderbase-7b", inference.approach1, ts_dataset),
        Config("starcoderbase-7b", inference.approach4, ts_dataset),
        Config("stenotype-7b-a6d445d-ckpt1000", inference.approach4, ts_dataset),
    ]

    # JavaScript datasets
    js_dataset = DatasetConf(
        short_name="js",
        datasets_path=args.datasets_directory,
        dataset_name="typeweaver-bundle-filtered-subset",
        type_decls="type_declarations.tar.gz",
    )
    configs += [
        Config("starcoderbase-1b", inference.approach1, js_dataset),
        Config("stenotype-1b-75ce914-ckpt100", inference.approach1, js_dataset),
        Config("stenotype-1b-54d5802-ckpt100", inference.approach2, js_dataset),
        Config("stenotype-1b-2b77ede-ckpt100", inference.approach3, js_dataset),
        Config("starcoderbase-1b", inference.approach4, js_dataset),
        Config("stenotype-1b-ef65cb9-ckpt250", inference.approach4, js_dataset),
        Config("stenotype-1b-ef65cb9-ckpt500", inference.approach4, js_dataset),
        Config("stenotype-1b-ef65cb9-ckpt750", inference.approach4, js_dataset),
        Config("stenotype-1b-ef65cb9-ckpt1000", inference.approach4, js_dataset),
        Config("starcoderbase-7b", inference.approach1, js_dataset),
        Config("starcoderbase-7b", inference.approach4, js_dataset),
        Config("stenotype-7b-a6d445d-ckpt1000", inference.approach4, js_dataset),
    ]

    if args.show_configs:
        for i, c in enumerate(configs):
            print(i, c.model_name, c.approach.__name__, c.dataset_config.short_name)
        exit(0)

    if args.infer is not None:
        # If indices given, then select only those configs
        infer_configs = configs
        if args.infer:
            indices = [int(i) for i in args.infer]
            bad_indices = [i for i in indices if i < 0 or i >= len(configs)]
            for i in bad_indices:
                print(f"error: configuration {i} is out of range!")
            if bad_indices:
                print(f"Configuration must be >=0 and <={len(configs) - 1}")
                exit(2)
            infer_configs = [configs[i] for i in indices]

        # Make sure results don't already exist
        results_paths = [
            c.infer_output_path(args.results_directory) for c in infer_configs
        ]
        results_exists = [path for path in results_paths if Path(path).exists()]
        for p in results_exists:
            print(f"error: output {p} already exists, please delete or rename!")
        if results_exists:
            exit(2)

        for c in infer_configs:
            run_inference(c, args)

    if args.evaluate is not None:
        # If indices given, then select only those configs
        eval_configs = configs
        if args.evaluate:
            indices = [int(i) for i in args.evaluate]
            bad_indices = [i for i in indices if i < 0 or i >= len(configs)]
            for i in bad_indices:
                print(f"error: configuration {i} is out of range!")
            if bad_indices:
                print(f"Configuration index must be >=0 and <={len(configs) - 1}")
                exit(2)
            eval_configs = [configs[i] for i in indices]

        # Make sure results don't already exist
        results_paths = [
            c.eval_output_path(args.results_directory) for c in eval_configs
        ]
        results_exists = [path for path in results_paths if Path(path).exists()]
        for p in results_exists:
            print(f"error: output {p} already exists, please delete or rename!")
        if results_exists:
            exit(2)

        for c in eval_configs:
            run_evaluation(c, args)

    if args.summarize:
        summarize(configs, args)


if __name__ == "__main__":
    main()
