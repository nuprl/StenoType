#!/bin/bash

# Assumes starcoderbase is saved in $MODELS_DIRECTORY; if the environment
# variable does not exist, look in ../models
: ${MODELS_DIRECTORY:=$(pwd)/../models}

# $MODELS_DIRECTORY is mounted to /data, and the default model is starcoderbase
# override by calling the script with `MODEL_NAME=starcoder ./run_starcoderbase.sh`
: ${MODEL_NAME:=starcoderbase-1b}

# use podman by default
# override by calling script with `DOCKER=docker ./run_starcoderbase.sh`
: ${DOCKER:=podman}

# use port 8787 by default
# override by calling script with `PORT=8080 ./run_starcoderbase.sh`
: ${PORT:=8787}

# use GPU 0 by default
# override by calling script with `DEVICES=2 ./run_starcoderbase.sh`
: ${DEVICES:=0}

$DOCKER run \
    -p $PORT:80 \
    -v $MODELS_DIRECTORY:/data \
    -e NVIDIA_VISIBLE_DEVICES=$DEVICES \
    -e HF_HUB_ENABLE_HF_TRANSER=0 \
    ghcr.io/huggingface/text-generation-inference:0.8 \
    --model-id /data/$MODEL_NAME \
    --max-input-length 8192 \
    --max-total-tokens 16384 \
    --max-waiting-tokens 65536
