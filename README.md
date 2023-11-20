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
conda install -c conda-forge nodejs=20.8.1
npm install -g --no-save typescript@5.2.2
```

4. Download the
   [StarCoderBase-1b](https://huggingface.co/bigcode/starcoderbase-1b)
   and
   [StarCoderBase-7b](https://huggingface.co/bigcode/starcoderbase-7b)
   models:

   a. Ensure that you have a Hugging Face account.

   b. Accept the agreement for
      [StarCoderBase-1b](https://huggingface.co/bigcode/starcoderbase-1b) and
      [StarCoderBase-7b](https://huggingface.co/bigcode/starcoderbase-7b).

   c. On the command line, log into Hugging Face with `huggingface-cli login`.

   d. In a directory of your choosing, e.g. `../models`,
      run `git clone git@hf.co:bigcode/starcoderbase-1b`.

   e. To save space, you can delete the `.git` directory (and possibly
      `pytorch_model*.bin` if `model*.safetensors` already exists).

5. Accept the agreement for the
   [ts-eval](https://huggingface.co/datasets/nuprl/ts-eval) evaluation dataset.

6. Now you can run the experiments:

```bash
# See what configurations can be wron
python src/main.py --show_configs

# To run inference on config 0 (this is very slow):
python src/main.py --infer 0

# To evaluate (this is CPU-bound):
python src/main.py --evaluate

# To generate dataset-level summaries (this is pretty fast):
python src/main.py --summarize
```

7. To browse the results, you can use the viewer. Type "help" for help.

```bash
python src/viewer.py --dataset path/to/results/dataset.parquet
```

## Dependencies

  * git
  * Python 3

Using [Conda](https://docs.conda.io/en/latest/) or [virtual
environments](https://docs.python.org/3/library/venv.html) is recommended.
