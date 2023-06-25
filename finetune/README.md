# Fine-tuning

Adapted from the [StarCoder finetuning
scripts](https://github.com/bigcode-project/starcoder/tree/main/finetune).

## Setup

Install the `pytorch` version compatible with your version of CUDA. See
instructions [here](https://pytorch.org/get-started/locally/).

Install the packages required for finetuning:

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

TODO dataset

TODO running the fine-tuning script

```bash
python finetune/finetune.py \
  --model_path="bigcode/starcoder"\
  --dataset_name="ArmelR/stack-exchange-instruction"\
  --subset="data/finetune"\
  --split="train"\
  --size_valid_set 10000\
  --streaming\
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="question"\
  --output_column_name="response"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```

TODO multiple GPUS

Similarly we can modify the command to account for the availability of GPUs:

```bash
python -m torch.distributed.launch \
  --nproc_per_node number_of_gpus finetune/finetune.py \
  --model_path="bigcode/starcoder"\
  --dataset_name="ArmelR/stack-exchange-instruction"\
  --subset="data/finetune"\
  --split="train"\
  --size_valid_set 10000\
  --streaming \
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="question"\
  --output_column_name="response"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```

## Merging PEFT adapter layers

If you train a model with PEFT, you'll need to merge the adapter layers with
the base model if you want to run inference / evaluation. To do so, run:

```bash
python merge_peft_adapters.py \
  --peft_model_path checkpoints/checkpoint-1000
```

By default, the base model is assumed to be StarCoderBase, and located in
`../../models/starcoderbase`. The model can be specified as a path or model ID,
e.g. `--model_name_or_path bigcode/starcoder`

By default, the merged model is saved to disk. Setting the `--push_to_hub`
argument will upload the merged model to the Hugging Face Hub.

## Customizing the fine-tuning script

TODO modify the `run_finetune.py` script with your own config
