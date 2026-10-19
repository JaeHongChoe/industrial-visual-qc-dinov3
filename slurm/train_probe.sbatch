#!/bin/bash
#SBATCH --job-name=patch_qc_dinov3_patch
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=/path/to/workspace/all_product_patch_binary_dinov3_poc
python_bin=${PYTHON_BIN:-python3}
manifest_dir=$(<"${project_root}/artifacts/latest_manifest_dir.txt")
run_dir="${project_root}/outputs/probe_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
export PYTHONPATH="${project_root}/src"

"${python_bin}" - <<'PY'
import torch
print("cuda_visible_count", torch.cuda.device_count())
if torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one Slurm-visible GPU")
print("allocated_gpu", torch.cuda.get_device_name(0))
PY

"${python_bin}" -m patch_qc.train_probe \
  --config "${project_root}/configs/poc.yaml" \
  --manifest "${manifest_dir}/manifest.jsonl" \
  --output-dir "${run_dir}"
printf '%s\n' "${run_dir}" > "${project_root}/outputs/latest_probe_dir.txt"
