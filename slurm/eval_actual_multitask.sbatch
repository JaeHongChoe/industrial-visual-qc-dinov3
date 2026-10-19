#!/bin/bash
#SBATCH --job-name=patch_qc_actual_eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=04:00:00

set -euo pipefail

: "${PATCHQC_PROJECT_ROOT:?PATCHQC_PROJECT_ROOT must name the isolated checkout}"
: "${PATCHQC_CODE_COMMIT:?PATCHQC_CODE_COMMIT must pin the isolated checkout}"
project_root=$(cd -- "${PATCHQC_PROJECT_ROOT}" && pwd)
actual_commit=$(git -C "${project_root}" rev-parse HEAD)
if [[ "${actual_commit}" != "${PATCHQC_CODE_COMMIT}" ]]; then
  echo "checkout mismatch: expected ${PATCHQC_CODE_COMMIT}, got ${actual_commit}" >&2
  exit 2
fi
if ! git -C "${project_root}" diff --quiet || ! git -C "${project_root}" diff --cached --quiet; then
  echo "isolated checkout has tracked modifications" >&2
  exit 2
fi
cd "${project_root}"

python_bin=${PYTHON_BIN:-python3}
config_path=${PATCHQC_CONFIG_PATH:-${project_root}/configs/synthetic_mask_multitask_smoke.yaml}
eval_manifest=${PATCHQC_EVAL_MANIFEST:-${project_root}/inputs/actual_ok_ng_eval.jsonl}
eval_manifest_sha256=${PATCHQC_EVAL_MANIFEST_SHA256:?PATCHQC_EVAL_MANIFEST_SHA256 must pin the frozen cohort}
base_checkpoint=${PATCHQC_BASE_CHECKPOINT:-/path/to/workspace/all_product_patch_binary_dinov3_poc/outputs/natural_full_10ep_6916/epoch_003.pt}
multitask_checkpoint=${PATCHQC_MULTITASK_CHECKPOINT:-/path/to/workspace/all_product_patch_multitask_seg_dinov3_poc_d1d6e2d/outputs/synthetic_mask_multitask_smoke_7008/checkpoint_epoch_001.pt}
multitask_sha256=${PATCHQC_MULTITASK_SHA256:-077c95591c01cf67d6126c7c28e85b0615d5ac55d2a340d1855707d053a9cde2}
training_receipt=${PATCHQC_TRAINING_RECEIPT:-/path/to/workspace/all_product_patch_multitask_seg_dinov3_poc_d1d6e2d/outputs/synthetic_mask_multitask_smoke_7008/run_receipt.json}
training_receipt_sha256=${PATCHQC_TRAINING_RECEIPT_SHA256:-4bc178df717a7be2d0c7cb64b52210798ed30bcf76819c670412c566a7b95f86}
output_dir=${PATCHQC_OUTPUT_DIR:-${project_root}/outputs/actual_ok_ng_multitask_${SLURM_JOB_ID}}
batch_size=${PATCHQC_BATCH_SIZE:-128}

export PYTHONPATH="${project_root}/src"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

mkdir -p "${project_root}/logs"

"${python_bin}" - <<'PY'
import json
import os
import subprocess

visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
if len(visible) != 1:
    raise SystemExit(f"expected one physical GPU selector, got {visible}")
selector = visible[0]
query = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
    check=True,
    capture_output=True,
    text=True,
)
gpus = []
for line in query.stdout.splitlines():
    fields = [value.strip() for value in line.split(",")]
    if len(fields) == 4:
        gpus.append({"index": fields[0], "uuid": fields[1], "free_mib": int(fields[2]), "total_mib": int(fields[3])})
assigned = next((row for row in gpus if selector in {row["index"], row["uuid"]}), gpus[0] if len(gpus) == 1 else None)
if assigned is None:
    raise SystemExit(f"could not resolve assigned GPU selector {selector!r}")
process_query = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
    check=True,
    capture_output=True,
    text=True,
)
existing = []
for line in process_query.stdout.splitlines():
    fields = [value.strip() for value in line.split(",")]
    if len(fields) == 3 and fields[0] == assigned["uuid"]:
        existing.append({"pid": fields[1], "used_mib": fields[2]})
if existing:
    raise SystemExit(f"assigned GPU {assigned['index']} {assigned['uuid']} already has compute processes: {existing}")
if assigned["free_mib"] < 70_000:
    raise SystemExit(f"assigned GPU {assigned['index']} has only {assigned['free_mib']} MiB free")
print(json.dumps({"gpu_preflight": assigned, "existing_compute_processes": 0}, sort_keys=True), flush=True)
PY

"${python_bin}" -m patch_qc.actual_multitask_eval \
  --config "${config_path}" \
  --eval-manifest "${eval_manifest}" \
  --base-checkpoint "${base_checkpoint}" \
  --multitask-checkpoint "${multitask_checkpoint}" \
  --training-receipt "${training_receipt}" \
  --expected-eval-manifest-sha256 "${eval_manifest_sha256}" \
  --expected-training-receipt-sha256 "${training_receipt_sha256}" \
  --expected-multitask-sha256 "${multitask_sha256}" \
  --output-dir "${output_dir}" \
  --batch-size "${batch_size}"
