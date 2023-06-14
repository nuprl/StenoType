#!/bin/bash

# use podman by default
# override by calling script with `DOCKER=docker ./run_starcoderbase.sh`
: ${DOCKER:=podman}

# use GPU 0 by default
# override by calling script with `DEVICES=2 ./run_starcoderbase.sh`
: ${DEVICES:=0}

$DOCKER run \
    -p 8787:80 \
    -v $(pwd)/../models:/data \
    -e NVIDIA_VISIBLE_DEVICES=$DEVICES \
    -e HF_HUB_ENABLE_HF_TRANSER=0 \
    ghcr.io/huggingface/text-generation-inference:0.8 \
    --model-id /data/starcoderbase \
    --max-input-length 8192 \
    --max-total-tokens 16384 \
    --max-waiting-tokens 65536
