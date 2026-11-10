#!/usr/bin/env bash
# Usage: bash tools/dist_train.sh CONFIG NUM_GPUS [extra tools/run.py train args]
set -e
CONFIG=$1
GPUS=$2
torchrun --standalone --nproc_per_node="$GPUS" "$(dirname "$0")/run.py" train \
    "$CONFIG" --launcher pytorch "${@:3}"
