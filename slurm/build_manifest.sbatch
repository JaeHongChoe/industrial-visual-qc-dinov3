#!/bin/bash
#SBATCH --job-name=patch_qc_manifest
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=/path/to/workspace/all_product_patch_binary_dinov3_poc
python_bin=${PYTHON_BIN:-python3}
run_stamp=$(date +%Y%m%d_%H%M%S)
artifact_dir="${project_root}/artifacts/manifest_${run_stamp}_${SLURM_JOB_ID}"
export PYTHONPATH="${project_root}/src"

"${python_bin}" -m patch_qc.cli build-manifest \
  --root /path/to/dataset/OK \
  --output-dir "${artifact_dir}" \
  --start-date 2026-05-01 \
  --seed 17
printf '%s\n' "${artifact_dir}" > "${project_root}/artifacts/latest_manifest_dir.txt"
