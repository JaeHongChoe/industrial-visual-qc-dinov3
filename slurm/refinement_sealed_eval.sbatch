#!/bin/bash
#SBATCH --job-name=patch_qc_refine_sealed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
run_root=${PATCHQC_REFINEMENT_RUN_ROOT:?set PATCHQC_REFINEMENT_RUN_ROOT to the frozen run}
labelme_root=${PATCHQC_LABELME_ROOT:?set PATCHQC_LABELME_ROOT}
ranked_candidates=${PATCHQC_REFINED_CANDIDATES:?set PATCHQC_REFINED_CANDIDATES}
baseline_candidates=${PATCHQC_BASELINE_CANDIDATES:?set PATCHQC_BASELINE_CANDIDATES}
training_hashes=${PATCHQC_TRAINING_HASHES:?set PATCHQC_TRAINING_HASHES}
export PYTHONPATH="${project_root}/src"
cd "${project_root}"

if [[ ! -f "${run_root}/frozen_policy.json" ]]; then
  echo "frozen policy is required" >&2
  exit 2
fi
if [[ -e "${run_root}/sealed_evaluation" ]]; then
  echo "sealed evaluation output already exists" >&2
  exit 2
fi

"${python_bin}" -m patch_qc.refinement_cli sealed-evaluate \
  --run-root "${run_root}" \
  --labelme-root "${labelme_root}" \
  --ranked-candidates "${ranked_candidates}" \
  --baseline-candidates "${baseline_candidates}" \
  --training-hashes "${training_hashes}"
