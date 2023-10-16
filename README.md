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

4. Download the
   [StarCoderBase-1b](https://huggingface.co/bigcode/starcoderbase-1b)
   model:

   a. Ensure that you have a Hugging Face account.

   b. Accept the agreement for
      [StarCoderBase-1b](https://huggingface.co/bigcode/starcoderbase-1b).

   c. On the command line, log into Hugging Face with `huggingface-cli login`.

   d. In a directory of your choosing, e.g. `../models`,
      run `git clone git@hf.co:bigcode/starcoderbase-1b`.

   e. To save space, you can delete the `.git` directory (and possibly
      `pytorch_model*.bin` if `model*.safetensors` already exists).

5. Accept the agreement for the
   [ts-eval](https://huggingface.co/datasets/nuprl/ts-eval) evaluation dataset.

6. Now you can run the experiments:

```bash
# To run inference (this is very slow):
CUDA_VISIBLE_DEVICES=0 python src/main.py --infer --workers 10

# To evaluate (this is CPU-bound):
python src/main.py --evaluate --workers 24

# To generate dataset-level summaries (this is pretty fast):
python src/main.py --summarize --workers 10
```

## Dependencies

  * git
  * Python 3

Using [Conda](https://docs.conda.io/en/latest/) or [virtual
environments](https://docs.python.org/3/library/venv.html) is recommended.
