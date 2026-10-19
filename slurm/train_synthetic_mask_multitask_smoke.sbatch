#!/bin/bash
#SBATCH --job-name=patch_qc_mask_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00

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
if [[ ! -f "${project_root}/configs/synthetic_mask_multitask_smoke.yaml" ]]; then
  echo "isolated checkout is missing the smoke config" >&2
  exit 2
fi
cd "${project_root}"

python_bin=${PYTHON_BIN:-python3}
manifest_path=${PATCHQC_MANIFEST_PATH:-/path/to/workspace/all_product_patch_binary_dinov3_poc/artifacts/manifest_20260827_184915_6892/manifest.jsonl}
base_checkpoint=${PATCHQC_BASE_CHECKPOINT:-/path/to/workspace/all_product_patch_binary_dinov3_poc/outputs/natural_full_10ep_6916/epoch_003.pt}
output_dir=${PATCHQC_OUTPUT_DIR:-${project_root}/outputs/synthetic_mask_multitask_smoke_${SLURM_JOB_ID}}

export PYTHONPATH="${project_root}/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

mkdir -p "${output_dir}" "${project_root}/logs"

"${python_bin}" - <<'PY'
import json
import os
import subprocess

visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
if len(visible) != 1:
    raise SystemExit(f"expected one physical GPU selector, got {visible}")
selector = visible[0]
query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
gpus = []
for line in query.stdout.splitlines():
    fields = [value.strip() for value in line.split(",")]
    if len(fields) == 4:
        gpus.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "free_mib": int(fields[2]),
                "total_mib": int(fields[3]),
            }
        )
assigned = next(
    (row for row in gpus if selector in {row["index"], row["uuid"]}),
    gpus[0] if len(gpus) == 1 else None,
)
if assigned is None:
    raise SystemExit(f"could not resolve assigned GPU selector {selector!r}")
process_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ],
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
    raise SystemExit(
        f"assigned GPU {assigned['index']} {assigned['uuid']} already has compute processes: {existing}"
    )
if assigned["free_mib"] < 70_000:
    raise SystemExit(
        f"assigned GPU {assigned['index']} has only {assigned['free_mib']} MiB free"
    )
print(json.dumps({"gpu_preflight": assigned, "existing_compute_processes": 0}, sort_keys=True), flush=True)
PY

"${python_bin}" - <<'PY'
import json
import os
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one Slurm-visible GPU, got {torch.cuda.device_count()}")
properties = torch.cuda.get_device_properties(0)
print(json.dumps({
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "device_name": torch.cuda.get_device_name(0),
    "device_uuid": str(getattr(properties, "uuid", "unavailable")),
}, sort_keys=True), flush=True)
PY

"${python_bin}" -m patch_qc.train_multitask \
  --config "${project_root}/configs/synthetic_mask_multitask_smoke.yaml" \
  --manifest "${manifest_path}" \
  --base-checkpoint "${base_checkpoint}" \
  --output-dir "${output_dir}"
