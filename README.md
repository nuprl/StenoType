# StenoType

To set up:

    git clone git@github.com:nuprl/StenoType.git
    cd StenoType
    pip install -r requirements.txt
    git submodule update --init --recursive
    printf "url to StarCoder text generation service" > .STARCODER_ENDPOINT

The provided `run_starcoderbase.sh` script shows how to run the text_generation
container to run the service at `http://127.0.0.1:8787`. It assumes the
[StarCoderBase](https://huggingface.co/bigcode/starcoderbase) model has been
downloaded to `../models/starcoderbase`.

To run:

    python main/main.py \
        --dataset nuprl/ts-eval \
        --revision v1.1subset \
        --split test \
        --workers 2

## Dependencies

  * git
  * Python 3
