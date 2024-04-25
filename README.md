# StenoType

Type migration with large language models for code. Migrates JavaScript to
TypeScript by predicting type annotations and generating type definitions.

This is the code repository for the dissertation [Predicting TypeScript Type
Annotations and Definitions With Machine Learning](http://hdl.handle.net/2047/D20653005),
specifically, Chapter 5.

The training dataset is on [Hugging Face](https://huggingface.co/datasets/nuprl/stenotype-training).
Parts of the code may refer to it as `ts-training-get4`. This is a preprocessed version of
[`ts-training`](https://huggingface.co/datasets/nuprl/ts-training), revision `v1.1p1`.

The final StenoType model is on [Hugging Face](https://huggingface.co/nuprl/stenotype).
You will need to accept the agreement to access the model. The code and results
may refer to this model as `stenotype-7b-a6d445d-ckpt1000`, as it was fine-tuned
based on commit [`a6d445d`](https://github.com/nuprl/StenoType/commit/a6d445d).

Figures and result summaries are in the `results/` directory. Full results are
on [Hugging Face](https://huggingface.co/datasets/nuprl/stenotype-results).

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
   [StarCoderBase-7b](https://huggingface.co/bigcode/starcoderbase-7b)
   and
   [StenoType](https://huggingface.co/nuprl/stenotype)
   models:

   a. Ensure that you have a Hugging Face account.

   b. Accept the agreements for
      [StarCoderBase-7b](https://huggingface.co/bigcode/starcoderbase-7b) and
      [StenoType](https://huggingface.co/nuprl/stenotype).

   c. On the command line, log into Hugging Face with `huggingface-cli login`.

   d. In a directory of your choosing, e.g. `../models`,
      run `git clone git@hf.co:bigcode/starcoderbase-7b` and
      `git clone git@hf.co:nuprl/stenotype`.

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
