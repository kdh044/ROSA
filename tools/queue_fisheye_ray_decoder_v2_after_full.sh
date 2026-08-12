#!/usr/bin/env bash
set -euo pipefail

full_launcher_pid=6037
full_command_token="full_rosa_corrected_raw_uniform_fresh_e0_to_e20_seed3407"
full_output="/home/airlab/user/kdh/ROSA/results/raw_metrics/experiments/full_rosa_corrected_raw_uniform_fresh_e0_to_e20_seed3407"
v2_source="/home/airlab/user/kdh/.runs/rosa_ray_decoder_v2"
v2_output="/home/airlab/user/kdh/ROSA/results/raw_metrics/experiments/fisheye_ray_decoder_v2_corrected_raw_uniform_fresh_e0_to_e20_seed3407"
v2_preflight="/home/airlab/user/kdh/.runs/rosa_ray_decoder_v2_ddp_smoke_after_full"
detector_host="/home/airlab/user/kdh/ROSA/results/raw_metrics/experiments/v6_c0_corrected_sparse4d_raw_uniform_fresh_e0_to_e30_seed3407_rosa9c05051_host1b2bb59_evala28726e/execution_snapshot_v1/detector"
official_root="/home/airlab/user/kdh/dependencies/fisheye3dod"
data_root="/home/airlab/user/kdh/datasets/Fisheye3DODdataset"

while [[ -r "/proc/${full_launcher_pid}/cmdline" ]]; do
    full_command="$(tr '\0' ' ' < "/proc/${full_launcher_pid}/cmdline")"
    if [[ "${full_command}" != *"${full_command_token}"* ]]; then
        echo "ABORT: PID ${full_launcher_pid} was reused by another command" >&2
        exit 2
    fi
    sleep 30
done

if [[ ! -s "${full_output}/epoch_20.pth" ]]; then
    echo "ABORT: Full ROSA exited without epoch_20.pth" >&2
    exit 3
fi
if [[ -e "${v2_output}" ]]; then
    echo "ABORT: V2 output already exists: ${v2_output}" >&2
    exit 4
fi
if [[ -e "${v2_preflight}" ]]; then
    echo "ABORT: V2 DDP preflight output already exists: ${v2_preflight}" >&2
    exit 5
fi

mkdir -p "${v2_output}/execution_snapshot_v1"
tar \
    --exclude=.git \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.pth' \
    -C "${v2_source}" \
    -czf "${v2_output}/execution_snapshot_v1/rosa_ray_decoder_v2_source.tar.gz" \
    .

export FISHEYE3DOD_OFFICIAL_ROOT="${official_root}"
export ROSA_DETECTOR_HOST_ROOT="${detector_host}"
export FISHEYE3DOD_DATA_ROOT="${data_root}"
export PYTHONPATH="${detector_host}:${v2_source}/src"
export CUDA_VISIBLE_DEVICES=0,1

cd "${v2_source}"
/usr/bin/python3 /usr/local/bin/torchrun \
    --nproc_per_node=2 \
    tools/distributed_smoke_fisheye_ray_decoder_v2.py \
    configs/fisheye_ray_decoder_v2_raw_uniform_20e.py \
    --work-dir "${v2_preflight}"

/usr/bin/python3 /usr/local/bin/torchrun \
    --nproc_per_node=2 \
    tools/run.py train \
    configs/fisheye_ray_decoder_v2_raw_uniform_20e.py \
    --launcher pytorch \
    --work-dir "${v2_output}"

/usr/bin/python3 tools/summarize_fisheye_ray_decoder_v2.py \
    --work-dir "${v2_output}" \
    --baseline-reference \
    "${v2_source}/results/raw_metrics/evaluations/c0_raw_uniform_20e_reference.json" \
    --output "${v2_output}/final_metrics_vs_c0.json"
