#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: DENSE_REF_OUTPUT_ROOT=... DENSE_REF_COHORT=... DENSE_REF_CALIBRATION_COHORT=... DENSE_REF_EVALUATION_OK_COHORT=... DENSE_REF_EVALUATION_NG_COHORT=... DENSE_REF_CHECKPOINT=... DENSE_REF_CHECKPOINT_SHA256=... $0 [--dry-run] GPU0 GPU1" >&2
}

dry_run=0
if [[ ${1:-} == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
gpu0=$1
gpu1=$2
if ! [[ $gpu0 =~ ^[0-3]$ && $gpu1 =~ ^[0-3]$ && $gpu0 != "$gpu1" ]]; then
  echo "expected two distinct physical GPU IDs in 0..3" >&2
  exit 2
fi

: "${DENSE_REF_OUTPUT_ROOT:?DENSE_REF_OUTPUT_ROOT is required}"
: "${DENSE_REF_COHORT:?DENSE_REF_COHORT is required}"
: "${DENSE_REF_CALIBRATION_COHORT:?DENSE_REF_CALIBRATION_COHORT is required}"
: "${DENSE_REF_EVALUATION_OK_COHORT:?DENSE_REF_EVALUATION_OK_COHORT is required}"
: "${DENSE_REF_EVALUATION_NG_COHORT:?DENSE_REF_EVALUATION_NG_COHORT is required}"
: "${DENSE_REF_CHECKPOINT:?DENSE_REF_CHECKPOINT is required}"
: "${DENSE_REF_CHECKPOINT_SHA256:?DENSE_REF_CHECKPOINT_SHA256 is required}"
root_dir=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}
output_root=$DENSE_REF_OUTPUT_ROOT
cohort=$DENSE_REF_COHORT
checkpoint=$DENSE_REF_CHECKPOINT
checkpoint_sha=$DENSE_REF_CHECKPOINT_SHA256
tile_batch_size=${DENSE_REF_TILE_BATCH_SIZE:-64}
reference_chunk_size=${DENSE_REF_REFERENCE_CHUNK_SIZE:-1}
reference_image_cache_size=${DENSE_REF_REFERENCE_IMAGE_CACHE_SIZE:-5}
decode_workers=${DENSE_REF_DECODE_WORKERS:-2}
if ! [[ $tile_batch_size =~ ^[1-9][0-9]*$ && $reference_chunk_size =~ ^[1-9][0-9]*$ && $reference_image_cache_size =~ ^[1-9][0-9]*$ && $decode_workers =~ ^[1-9][0-9]*$ ]] || (( reference_image_cache_size > 5 )); then
  echo "runtime knobs must be positive integers and reference image cache size must be at most 5" >&2
  exit 2
fi
runtime_config_json=$(printf '{"tile_batch_size":%s,"reference_chunk_size":%s,"reference_image_cache_size":%s,"decode_workers":%s}' "$tile_batch_size" "$reference_chunk_size" "$reference_image_cache_size" "$decode_workers")

runner_command() {
  local rank=$1
  local gpu=$2
  printf 'CUDA_VISIBLE_DEVICES=%s HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TIMM_OFFLINE=1 %q -m dense_ref.runner --rank %s --world-size 2 --cohort %q --output-dir %q --checkpoint %q --checkpoint-sha256 %q --physical-gpu-index %s --precision fp16 --tile-batch-size %s --reference-chunk-size %s --reference-image-cache-size %s --decode-workers %s\n' \
    "$gpu" "$python_bin" "$rank" "$cohort" "$output_root" "$checkpoint" "$checkpoint_sha" "$gpu" "$tile_batch_size" "$reference_chunk_size" "$reference_image_cache_size" "$decode_workers"
}

runner_command 0 "$gpu0"
runner_command 1 "$gpu1"
printf '%q -m dense_ref.evaluate --mode full --cohort %q --calibration-cohort %q --evaluation-ok-cohort %q --evaluation-ng-cohort %q --output-dir %q --controller-receipt %q\n' \
  "$python_bin" "$cohort" "$DENSE_REF_CALIBRATION_COHORT" "$DENSE_REF_EVALUATION_OK_COHORT" "$DENSE_REF_EVALUATION_NG_COHORT" "$output_root" "$output_root/controller_launch.json"
printf 'controller runtime_config=%s\n' "$runtime_config_json"
if [[ $dry_run -eq 1 ]]; then
  exit 0
fi

[[ -f $cohort && -f $checkpoint && -f $DENSE_REF_CALIBRATION_COHORT && -f $DENSE_REF_EVALUATION_OK_COHORT && -f $DENSE_REF_EVALUATION_NG_COHORT ]] || { echo "cohort/checkpoint inputs must exist" >&2; exit 2; }
[[ $checkpoint_sha =~ ^[0-9a-fA-F]{64}$ ]] || { echo "DENSE_REF_CHECKPOINT_SHA256 must be 64 hexadecimal characters" >&2; exit 2; }
mkdir -p "$output_root"
launch_tmp=$(mktemp "$output_root/.controller_launch.json.XXXXXX.tmp")
PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$gpu0" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TIMM_OFFLINE=1 \
  "$python_bin" -m dense_ref.runner --rank 0 --world-size 2 --cohort "$cohort" --output-dir "$output_root" --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" --physical-gpu-index "$gpu0" --precision fp16 --tile-batch-size "$tile_batch_size" --reference-chunk-size "$reference_chunk_size" --reference-image-cache-size "$reference_image_cache_size" --decode-workers "$decode_workers" &
pid0=$!
PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$gpu1" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TIMM_OFFLINE=1 \
  "$python_bin" -m dense_ref.runner --rank 1 --world-size 2 --cohort "$cohort" --output-dir "$output_root" --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" --physical-gpu-index "$gpu1" --precision fp16 --tile-batch-size "$tile_batch_size" --reference-chunk-size "$reference_chunk_size" --reference-image-cache-size "$reference_image_cache_size" --decode-workers "$decode_workers" &
pid1=$!
printf '{"mode":"full","runtime_config":%s,"children":[{"rank":0,"pid":%s,"physical_gpu_index":%s},{"rank":1,"pid":%s,"physical_gpu_index":%s}]}' "$runtime_config_json" "$pid0" "$gpu0" "$pid1" "$gpu1" > "$launch_tmp"
mv "$launch_tmp" "$output_root/controller_launch.json"

status=0
if ! wait "$pid0"; then status=1; fi
if ! wait "$pid1"; then status=1; fi
if [[ $status -ne 0 ]]; then
  exit "$status"
fi
PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m dense_ref.evaluate --mode full --cohort "$cohort" --calibration-cohort "$DENSE_REF_CALIBRATION_COHORT" --evaluation-ok-cohort "$DENSE_REF_EVALUATION_OK_COHORT" --evaluation-ng-cohort "$DENSE_REF_EVALUATION_NG_COHORT" --output-dir "$output_root" --controller-receipt "$output_root/controller_launch.json"
