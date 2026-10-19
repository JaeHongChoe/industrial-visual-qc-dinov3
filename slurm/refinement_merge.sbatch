#!/bin/bash
#SBATCH --job-name=patch_qc_refine_merge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log
#SBATCH --error=/path/to/workspace/all_product_patch_binary_dinov3_poc/logs/%x_%j.log

set -euo pipefail
project_root=${PATCHQC_REFINEMENT_PROJECT_ROOT:-/path/to/workspace/all_product_patch_binary_dinov3_poc}
python_bin=${PYTHON_BIN:-python3}
run_root=${PATCHQC_REFINEMENT_RUN_ROOT:?set PATCHQC_REFINEMENT_RUN_ROOT to a new merged run}
shard_cache_list=${PATCHQC_REFINEMENT_SHARD_CACHES:?set colon-separated PATCHQC_REFINEMENT_SHARD_CACHES}
export PYTHONPATH="${project_root}/src"
cd "${project_root}"

IFS=: read -r -a shard_caches <<< "${shard_cache_list}"
if (( ${#shard_caches[@]} < 2 )); then
  echo "at least two shard caches are required" >&2
  exit 2
fi
merge_args=()
for shard_cache in "${shard_caches[@]}"; do
  merge_args+=(--shard-cache "${shard_cache}")
done

"${python_bin}" -m patch_qc.refinement_cli merge-cache \
  --run-root "${run_root}" \
  --config "${project_root}/configs/refinement_v1.yaml" \
  "${merge_args[@]}"
