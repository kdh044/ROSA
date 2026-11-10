#!/usr/bin/env bash
# Usage: bash tools/dist_test.sh CONFIG CHECKPOINT NUM_GPUS [extra tools/run.py evaluate args]
set -e
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
torchrun --standalone --nproc_per_node="$GPUS" "$(dirname "$0")/run.py" evaluate \
    "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:4}"
