# StenoType

Type migration with large language models for code. Migrates JavaScript to
TypeScript by predicting type annotations and generating type definitions.

## Instructions

Clone the repository and install dependencies:

    git clone git@github.com:nuprl/StenoType.git
    cd StenoType
    git submodule update --init --recursive
    pip install -r requirements.txt

StenoType uses the [StarCoderBase](https://huggingface.co/bigcode/starcoder)
model. If you have access to a server running the
[`text_generation`](https://github.com/huggingface/text-generation-inference)
service, you can save the URL in the file `.STARCODER_ENDPOINT`:

    printf "http://127.0.0.1:8787" > .STARCODER_ENDPOINT

Otherwise, you can run it locally:

  1. Ensure you have a Hugging Face account.
  2. Accept the agreement for
     [StarCoderBase](https://huggingface.co/bigcode/starcoder).
  3. On the command line, log into Hugging Face with `huggingface-cli login`.
  4. In `../models`, run `git clone git@hf.co:bigcode/starcoderbase`.
  5. To save space, you can delete the `.git` directory and `pytorch_model*.bin`
     files, _after_ they have been converted to `model*.safetensors` format.
  6. Run `./run_starcoderbase.sh`. This will start the server locally.

Accept the agreement for the
[ts-eval](https://huggingface.co/datasets/nuprl/ts-eval) evaluation dataset.

Now you can run the experiments:

```bash
python main/main.py \
  --dataset nuprl/ts-eval \
  --revision v1.1subset \
  --split test \
  --workers 2
```

## Dependencies

  * git
  * Python 3
  * [Podman](https://podman.io/) (or Docker) with the
    [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Using [Conda](https://docs.conda.io/en/latest/) or [virtual
environments](https://docs.python.org/3/library/venv.html) is recommended.
