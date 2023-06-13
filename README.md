# StenoType

To set up:

    git clone git@github.com:nuprl/StenoType.git
    cd StenoType
    pip install -r requirements.txt
    git submodule update --init --recursive
    printf "url to StarCoder text generation service" > .STARCODER_ENDPOINT

To run:

    python main.py \
        --dataset nuprl/ts-eval \
        --revision v1.1subset \
        --split test \
        --workers 2
