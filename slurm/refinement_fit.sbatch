#!/bin/bash
#SBATCH --job-name=patch_qc_refine_fit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
run_root=${PATCHQC_REFINEMENT_RUN_ROOT:?set PATCHQC_REFINEMENT_RUN_ROOT to a completed build directory}
checkpoint=${PATCHQC_REFINEMENT_CHECKPOINT:?set PATCHQC_REFINEMENT_CHECKPOINT}
export PYTHONPATH="${project_root}/src"
export PATCHQC_REFINEMENT_WORKERS=${SLURM_CPUS_PER_TASK}
cd "${project_root}"

"${python_bin}" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one scheduler-visible GPU, got {torch.cuda.device_count()}")
print("allocated_gpu", torch.cuda.get_device_name(0))
PY

nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}" -s pucm -d 1 -o DT > "${run_root}/gpu_fit_telemetry.log" &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true' EXIT

"${python_bin}" -m patch_qc.refinement_cli build-anchors \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml" \
  --checkpoint "${checkpoint}"
"${python_bin}" -m patch_qc.refinement_cli fit-refiner \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml"
"${python_bin}" -m patch_qc.refinement_cli freeze-policy \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml"
