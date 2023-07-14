# Fine-tuning

Adapted from the [StarCoder fine-tuning
scripts](https://github.com/bigcode-project/starcoder/tree/main/finetune).

These scripts are used to fine-tune StarCoder for TypeScript type inference.

## Setup

These instructions assume you are using [conda](https://docs.conda.io/en/latest/)
and have followed the setup steps in `../README.md`.

Install `cudatoolkit`:

    conda install cudatoolkit

Install the `pytorch` version compatible with your version of CUDA. See
instructions [here](https://pytorch.org/get-started/locally/). For example:

    # This is an example, you may need to run a different command for
    # your version of CUDA!
    pip install torch --index-url https://download.pytorch.org/whl/cu118

Install the packages required for fine-tuning:

    pip install -r requirements.txt

Log into Hugging Face:

    huggingface-cli login

Log into `wandb`:

    wandb login

See the instructions in `../README.md` for downloading the
[StarCoderBase](https://huggingface.co/bigcode/starcoder) model.

## Running fine-tuning

To fine-tune cheaply and efficiently, we use Hugging Face 🤗's
[PEFT](https://github.com/huggingface/peft) as well as Tim Dettmers'
[bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

The fine-tuning is based on the
[ts-training](https://huggingface.co/datasets/nuprl/ts-training) dataset,
revision `v1.1p1`. **You will need to accept the agreement** on the dataset
page before you can use it for training.

The TypeScript files are processed into the StarCoder GitHub
commit format, to train the model to add types to untyped code:

    <commit_before>{code without types}<commit_msg>Add type annotations and interfaces<commit_after>{original code}

The fine-tuning script can be excuted by running:

```bash
python run_finetune.py
```

Update the configuration by editing `run_finetune.py`. The script takes two
optional command-line arguments:

    --seed 0
    --num_workers 4

### Multiple GPUs (on a single machine)

To fine-tune on multiple machines (on a single node), run:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 run_finetune.py
```

Use `CUDA_VISIBLE_DEVICES` to select the GPUs for fine-tuning, and
`--nproc-per-node` to specify the number of GPUs to use.

## Merging PEFT adapter layers

If you train a model with PEFT, you'll need to merge the adapter layers with
the base model if you want to run inference / evaluation. To do so, run:

```bash
python merge_peft_adapters.py \
  --peft_model_path ../checkpoints/checkpoint-1000
```

By default, the base model is assumed to be StarCoderBase. If the environment
variable `$MODELS_DIRECTORY` exists, then the model is assumed to be located in
`$MODELS_DIRECTORY/starcoderbase`; otherwise the location is assumed to be
`../../models/starcoderbase`. The model can be specified as a path or model ID,
e.g. `--model_name_or_path bigcode/starcoder`

By default, the merged model is saved to disk. Setting the `--push_to_hub`
argument will upload the merged model to the Hugging Face Hub.

## Customizing the fine-tuning script

The fine-tuning script `finetune_lib.py` is meant to be reusable to fine-tune
for other tasks. To use these scripts for your own task:

  1. Copy the files in this directory (`finetune/`).
  2. Edit `run_finetune.py` for your use case. This includes editing the
     location of the base model, changing the training hyperparameters,
     editing the `get_content` function which processes dataset examples for
     training, and updating how the dataset is loaded (which could be loaded
     from the Hugging Face Hub or disk).

You should not need to modify `finetune_lib.py` or the `main` function of
`run_finetune.py`.
