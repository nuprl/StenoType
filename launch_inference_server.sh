#!/bin/bash

# Assumes starcoderbase is saved in $MODELS_DIRECTORY; if the environment
# variable does not exist, look in ../models
: ${MODELS_DIRECTORY:=$(pwd)/../models}

# $MODELS_DIRECTORY is mounted to /data, and the default model is starcoderbase-1b
# override by calling the script with `MODEL_NAME=starcoder ./launch_inference_server.sh`
: ${MODEL_NAME:=starcoderbase-1b}

# use podman by default
# override by calling script with `DOCKER=docker ./launch_inference_server.sh`
: ${DOCKER:=podman}

# use port 8787 by default
# override by calling script with `PORT=8080 ./launch_inference_server.sh`
: ${PORT:=8787}

# use GPU 0 by default
# override by calling script with `DEVICES=2 ./launch_inference_server.sh`
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
