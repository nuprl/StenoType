#!/bin/bash

# use podman by default
# override by calling script with `DOCKER=docker ./run_starcoderbase.sh`
: ${DOCKER:=podman}

$DOCKER run \
    -p 8787:80 \
    -v $(pwd)/../models:/data \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e HF_HUB_ENABLE_HF_TRANSER=0 \
    ghcr.io/huggingface/text-generation-inference:0.8 \
    --model-id /data/starcoderbase \
    --max-total-tokens 8192
