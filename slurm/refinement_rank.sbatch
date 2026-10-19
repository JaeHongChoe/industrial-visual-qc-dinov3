#!/bin/bash
#SBATCH --job-name=patch_qc_refine_rank
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
run_root=${PATCHQC_REFINEMENT_RUN_ROOT:?set PATCHQC_REFINEMENT_RUN_ROOT to the frozen run}
request=${PATCHQC_REFINEMENT_RANK_REQUEST:?set PATCHQC_REFINEMENT_RANK_REQUEST}
checkpoint=${PATCHQC_REFINEMENT_CHECKPOINT:?set PATCHQC_REFINEMENT_CHECKPOINT}
output=${PATCHQC_REFINEMENT_RANK_OUTPUT:-${run_root}/ranking}
export PYTHONPATH="${project_root}/src"
export PATCHQC_REFINEMENT_WORKERS=${SLURM_CPUS_PER_TASK}
cd "${project_root}"

if [[ ! -f "${run_root}/frozen_policy.json" ]]; then
  echo "frozen policy is required" >&2
  exit 2
fi
if [[ -e "${output}" ]]; then
  echo "ranking output already exists: ${output}" >&2
  exit 2
fi

"${python_bin}" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one scheduler-visible GPU, got {torch.cuda.device_count()}")
print("allocated_gpu", torch.cuda.get_device_name(0))
PY

nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}" -s pucm -d 1 -o DT > "${run_root}/gpu_rank_${SLURM_JOB_ID}_telemetry.log" &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true' EXIT

"${python_bin}" -m patch_qc.refinement_cli rank-images \
  --run-root "${run_root}" \
  --request "${request}" \
  --config "${project_root}/configs/refinement_v1.yaml" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output}"
