#!/usr/bin/env bash
set -euo pipefail

source_root="/home/airlab/user/kdh/.runs/rosa_ray_decoder_v2"
output_root="/home/airlab/user/kdh/ROSA/results/raw_metrics/experiments/fisheye_box_decoder_v3_corrected_raw_uniform_fresh_e0_to_e20_seed3407"
preflight_root="${source_root}/work_dirs/fisheye_box_decoder_v3_ddp_preflight_queued"
detector_host="/home/airlab/user/kdh/ROSA/results/raw_metrics/experiments/v6_c0_corrected_sparse4d_raw_uniform_fresh_e0_to_e30_seed3407_rosa9c05051_host1b2bb59_evala28726e/execution_snapshot_v1/detector"
official_root="/home/airlab/user/kdh/dependencies/fisheye3dod"
data_root="/home/airlab/user/kdh/datasets/Fisheye3DODdataset"
max_idle_memory_mib=4096

if [[ -e "${output_root}" ]]; then
    echo "ABORT: V3 output already exists: ${output_root}" >&2
    exit 2
fi
if [[ -e "${preflight_root}" ]]; then
    echo "ABORT: queued V3 preflight output already exists: ${preflight_root}" >&2
    exit 3
fi

consecutive_idle_checks=0
while (( consecutive_idle_checks < 2 )); do
    mapfile -t used_memory < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    )
    if (( ${#used_memory[@]} < 2 )); then
        echo "ABORT: two visible GPUs are required" >&2
        exit 4
    fi
    if (( used_memory[0] <= max_idle_memory_mib && used_memory[1] <= max_idle_memory_mib )); then
        ((consecutive_idle_checks += 1))
    else
        consecutive_idle_checks=0
    fi
    echo "WAIT gpu_memory_mib=${used_memory[*]} idle_checks=${consecutive_idle_checks}/2"
    if (( consecutive_idle_checks < 2 )); then
        sleep 30
    fi
done

mkdir -p "${output_root}/execution_snapshot_v1"
tar \
    --exclude=.git \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.pth' \
    --exclude='results' \
    --exclude='work_dirs' \
    -C "${source_root}" \
    -czf "${output_root}/execution_snapshot_v1/fisheye_box_decoder_v3_source.tar.gz" \
    .

export FISHEYE3DOD_OFFICIAL_ROOT="${official_root}"
export ROSA_DETECTOR_HOST_ROOT="${detector_host}"
export FISHEYE3DOD_DATA_ROOT="${data_root}"
export PYTHONPATH="${source_root}/src:${source_root}:${detector_host}:${official_root}"
export CUDA_VISIBLE_DEVICES=0,1

cd "${source_root}"
/usr/bin/python3 /usr/local/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    tools/distributed_smoke_fisheye_ray_decoder_v2.py \
    configs/fisheye_box_decoder_v3_raw_uniform_20e.py \
    --work-dir "${preflight_root}"

/usr/bin/python3 /usr/local/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    tools/run.py train \
    configs/fisheye_box_decoder_v3_raw_uniform_20e.py \
    --launcher pytorch \
    --work-dir "${output_root}"

echo "V3_TRAINING_COMPLETE output=${output_root}"
