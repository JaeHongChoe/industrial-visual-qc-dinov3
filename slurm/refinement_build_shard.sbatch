#!/bin/bash
#SBATCH --job-name=patch_qc_refine_shard
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%A_%a.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%A_%a.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
source_manifest=${PATCHQC_REFINEMENT_SOURCE_MANIFEST:?set PATCHQC_REFINEMENT_SOURCE_MANIFEST}
checkpoint=${PATCHQC_REFINEMENT_CHECKPOINT:?set PATCHQC_REFINEMENT_CHECKPOINT}
shard_count=${PATCHQC_REFINEMENT_SHARD_COUNT:?set PATCHQC_REFINEMENT_SHARD_COUNT}
shard_index=${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}
batch_size=${PATCHQC_REFINEMENT_BATCH_SIZE:-1024}
workers=${PATCHQC_REFINEMENT_WORKERS:-${SLURM_CPUS_PER_TASK}}
prefetch_sources=${PATCHQC_REFINEMENT_PREFETCH_SOURCES:-2}
prefetch_batches=${PATCHQC_REFINEMENT_PREFETCH_BATCHES:-2}
progress_interval_seconds=${PATCHQC_REFINEMENT_PROGRESS_INTERVAL_SECONDS:-30}
limit_sources=${PATCHQC_REFINEMENT_LIMIT_SOURCES:-0}
run_parent=${PATCHQC_REFINEMENT_SHARD_RUN_PARENT:-${project_root}/outputs/candidate_refinement_v1_shards_${SLURM_ARRAY_JOB_ID}}
run_root=${run_parent}/shard_${shard_index}
export PYTHONPATH="${project_root}/src"
export PYTHONUNBUFFERED=1
source "${project_root}/slurm/refinement_runtime.sh"
mkdir -p "${run_parent}"
mkdir "${run_root}"
cd "${project_root}"

if (( shard_count <= 1 || shard_index < 0 || shard_index >= shard_count || batch_size <= 0 || limit_sources < 0 )); then
  echo "invalid cache shard or batch configuration" >&2
  exit 2
fi
validate_refinement_source_loading "${SLURM_CPUS_PER_TASK}" "${workers}" "${prefetch_sources}" "${prefetch_batches}"
validate_positive_integer "${progress_interval_seconds}" "progress interval"

"${python_bin}" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one scheduler-visible GPU, got {torch.cuda.device_count()}")
print("allocated_gpu", torch.cuda.get_device_name(0))
PY

nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}" -s pucm -d 1 -o DT > "${run_root}/gpu_cache_shard_telemetry.log" &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true' EXIT

limit_args=()
if (( limit_sources > 0 )); then
  limit_args+=(--limit-sources "${limit_sources}")
fi

"${python_bin}" -m patch_qc.refinement_cli build-cache \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml" \
  --source-manifest "${source_manifest}" \
  --checkpoint "${checkpoint}" \
  --batch-size "${batch_size}" \
  --workers "${workers}" \
  --prefetch-sources "${prefetch_sources}" \
  --prefetch-batches "${prefetch_batches}" \
  --progress-interval-seconds "${progress_interval_seconds}" \
  --shard-count "${shard_count}" \
  --shard-index "${shard_index}" \
  "${limit_args[@]}"
