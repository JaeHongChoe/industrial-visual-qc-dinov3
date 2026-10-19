#!/bin/bash
#SBATCH --job-name=patch_qc_refine_cache_bench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
source_manifest=${PATCHQC_REFINEMENT_SOURCE_MANIFEST:?set PATCHQC_REFINEMENT_SOURCE_MANIFEST}
checkpoint=${PATCHQC_REFINEMENT_CHECKPOINT:?set PATCHQC_REFINEMENT_CHECKPOINT}
limit_sources=${PATCHQC_REFINEMENT_LIMIT_SOURCES:-12}
batch_size=${PATCHQC_REFINEMENT_BATCH_SIZE:-512}
workers=${PATCHQC_REFINEMENT_WORKERS:-${SLURM_CPUS_PER_TASK}}
prefetch_sources=${PATCHQC_REFINEMENT_PREFETCH_SOURCES:-2}
prefetch_batches=${PATCHQC_REFINEMENT_PREFETCH_BATCHES:-2}
progress_interval_seconds=${PATCHQC_REFINEMENT_PROGRESS_INTERVAL_SECONDS:-30}
run_root=${PATCHQC_REFINEMENT_RUN_ROOT:-${project_root}/outputs/candidate_refinement_v1_cache_benchmark_${SLURM_JOB_ID}}
export PYTHONPATH="${project_root}/src"
export PYTHONUNBUFFERED=1
source "${project_root}/slurm/refinement_runtime.sh"
mkdir -p "${project_root}/outputs"
mkdir "${run_root}"
cd "${project_root}"

if (( limit_sources <= 0 || batch_size <= 0 )); then
  echo "PATCHQC_REFINEMENT_LIMIT_SOURCES and PATCHQC_REFINEMENT_BATCH_SIZE must be positive" >&2
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

nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}" -s pucm -d 1 -o DT > "${run_root}/gpu_cache_benchmark_telemetry.log" &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true' EXIT

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
  --limit-sources "${limit_sources}"
printf '%s\n' "${run_root}" > "${project_root}/outputs/latest_refinement_cache_benchmark_dir.txt"
