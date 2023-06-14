from datasets import Dataset
from pathlib import Path
import argparse, datasets, os

from model import Model
from type_inference import TypeInference

def parse_args():
    cpu_count = os.cpu_count()

    parser = argparse.ArgumentParser(
        description="Runs StarCoder to infer types for JavaScript")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="load the input dataset from a .parquet file, .jsonl file, or local Hugging Face dataset; otherwise tries to load from the Hugging Face Hub")
    parser.add_argument(
        "--revision",
        type=str,
        help="Dataset revision, if loading from the Hub")
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split, if loading from the Hub")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"maximum number of workers to use, defaults to {cpu_count}")

    return parser.parse_args()

def load_dataset(dataset, split="train", revision="main", workers=1):
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
        return datasets.load_dataset(dataset,
                                     split=split,
                                     revision=revision,
                                     num_proc=workers)

def main():
    args = parse_args()

    model = Model()
    typeinf = TypeInference(model)

    # For now, use TS with no type annotations
    # Still need to strip type definitions
    # TODO: add a column with type definitions removed, or remove them on-the-fly?
    # For now, use content column (with type defs and anns) and remove on the fly
    # Later we can look try a JS dataset
    #dataset = load_dataset(args.dataset, args.split, args.revision, args.workers)

    #first_example = dataset[0]["content_without_annotations"]
    first_example = """
    function aaa(x,
                 y)
                 { return x + y; }
    function abc(x, y?) { return x; }
    function def(x = 0) { return x; }
    function ghi({x, y}) { return x + y; }
    let jkl = function(x, y?) { return x; }
    let mno = function(x = 0) { return x; }
    let pqr = function({x, y}) { return x + y; }
    const a = x1 => x1;
    const b = (x2) => x2;
    const c = x3 => x3;
    const d = (x4, y4) => x4 + y4;
    var {e, f} = abcdef;
    class A {
        x;
        y = 42;
    }
    """
    # result = typeinf.infer(first_example)
    result = model.edit(first_example, "Add types")
    print(result)

    # TODO
    # - Do simple type inference with edit prompt
    #   - What to do if the file is too large?
    # - Evaluate with accuracy
    #   - Later we can type check (using TypeScript LSP?)
    # - Then output results

if __name__ == "__main__":
    main()
