# StenoType

Type migration with large language models for code. Migrates JavaScript to
TypeScript by predicting type annotations and generating type definitions.

## Instructions

1. Clone the repository:

```bash
git clone git@github.com:nuprl/StenoType.git
cd StenoType
git submodule update --init --recursive
```

2. Follow the instructions to set up
   [miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. Create a conda environment with Python 3.11 and install dependencies:

```bash
conda create -n gpu python=3.11
conda activate gpu
pip install -r requirements.txt
conda install nodejs=18.16.0
(cd ts && npm install && npx tsc)
```

4. StenoType will automatically start a container running a
   [`text_generation`](https://github.com/huggingface/text-generation-inference)
   server, with the
   [StarCoderBase-1b](https://huggingface.co/bigcode/starcoderbase-1b).
   model; however, the model must be downloaded first.

   a. Ensure that you have a Hugging Face account.

   b. Accept the agreement for
      [StarCoderBase-1b](https://huggingface.co/bigcode/starcoderbase-1b).

   c. On the command line, log into Hugging Face with `huggingface-cli login`.

   d. In a directory of your choosing, e.g. `../models`,
      run `git clone git@hf.co:bigcode/starcoderbase-1b`.

   e. To save space, you can delete the `.git` directory and
      `pytorch_model*.bin` files, _after_ they have been converted to
      `model*.safetensors` format. The conversion happens during the first run.

5. Accept the agreement for the
   [ts-eval](https://huggingface.co/datasets/nuprl/ts-eval) evaluation dataset.

6. Now you can run the experiments:

```bash
python src/main.py \
  --dataset nuprl/ts-eval \
  --revision v1.1subset \
  --split test \
  --workers 10 \
  --model ../models/starcoderbase-1b \
  --devices 0
```

## Dependencies

  * git
  * Python 3
  * [Podman](https://podman.io/) (or Docker) with the
    [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Using [Conda](https://docs.conda.io/en/latest/) or [virtual
environments](https://docs.python.org/3/library/venv.html) is recommended.
