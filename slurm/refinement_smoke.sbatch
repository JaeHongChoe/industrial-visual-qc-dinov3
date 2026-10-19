#!/bin/bash
#SBATCH --job-name=patch_qc_refine_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
source_manifest=${PATCHQC_REFINEMENT_SOURCE_MANIFEST:?set PATCHQC_REFINEMENT_SOURCE_MANIFEST}
checkpoint=${PATCHQC_REFINEMENT_CHECKPOINT:?set PATCHQC_REFINEMENT_CHECKPOINT}
run_root="${project_root}/outputs/candidate_refinement_v1_smoke_${SLURM_JOB_ID}"
export PYTHONPATH="${project_root}/src"
export PATCHQC_REFINEMENT_WORKERS=${SLURM_CPUS_PER_TASK}
mkdir -p "${project_root}/outputs"
mkdir "${run_root}"
cd "${project_root}"

"${python_bin}" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one scheduler-visible GPU, got {torch.cuda.device_count()}")
print("allocated_gpu", torch.cuda.get_device_name(0))
PY

nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}" -s pucm -d 1 -o DT > "${run_root}/gpu_telemetry.log" &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true' EXIT

"${python_bin}" -m pytest -q
"${python_bin}" -m patch_qc.refinement_cli build-cache \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml" \
  --source-manifest "${source_manifest}" \
  --checkpoint "${checkpoint}" \
  --limit-sources 12
"${python_bin}" -m patch_qc.refinement_cli build-anchors \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml" \
  --checkpoint "${checkpoint}" \
  --synthetic-count 18 \
  --synthetic-calibration-count 18
"${python_bin}" -m patch_qc.refinement_cli fit-refiner \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml"
"${python_bin}" -m patch_qc.refinement_cli freeze-policy \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml"
