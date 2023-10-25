from pathlib import Path
from transformers import AutoTokenizer
import argparse
import shutil

MODEL_PATH = str(
    Path(Path(__file__).parent, "..", "..", "models", "starcoderbase-1b").resolve()
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default=MODEL_PATH)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    ckpt_files = ["config.json", "generation_config.json", "pytorch_model.bin"]
    ckpt_paths = [Path(args.checkpoint, f) for f in ckpt_files]

    missing = [str(p) for p in ckpt_paths if not p.exists()]
    if missing:
        print("Missing checkpoint files:")
        print("\n".join(missing))
        exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=args.force)

    pairs = [[p, Path(args.output, p.name)] for p in ckpt_paths]
    for src, dst in pairs:
        shutil.copy2(src, dst)

    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
