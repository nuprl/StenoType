from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

MODEL_PATH = str(
    Path(Path(__file__).parent, "..", "..", "models", "starcoderbase-1b").resolve()
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default=MODEL_PATH)
    parser.add_argument("--peft_model_path", type=str, required=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.push_to_hub:
        print("Saving to hub ...")
        model.push_to_hub(
            f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True
        )
        tokenizer.push_to_hub(
            f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True
        )
    else:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
