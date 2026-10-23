#!/usr/bin/env bash
set -euo pipefail

# Download NVIDIA's official Cosmos3-DROID dataset through the HF mirror.
# This is the dataset used by the Cosmos3 DROID action/forward-dynamics recipes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_DIR="${COSMOS3_DROID_DIR:-${ROOT_DIR}/datasets/cosmos3_droid}"
HF_ENV="${COSMOS3_HF_PYTHON:-${ROOT_DIR}/environment/world_models/cosmos3_nano/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_ATTEMPTS="${COSMOS3_DROID_MAX_ATTEMPTS:-5}"
if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || (( MAX_ATTEMPTS > 20 )); then
  echo "COSMOS3_DROID_MAX_ATTEMPTS must be between 1 and 20" >&2; exit 2
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${ROOT_DIR}/.cache/huggingface/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-600}"

if [[ ! -x "${HF_ENV}" ]]; then
  echo "Cosmos3 environment Python not found: ${HF_ENV}" >&2
  echo "Run bash scripts/setup/cosmos3_nano.sh first, or set COSMOS3_HF_PYTHON." >&2
  exit 1
fi

patterns=()
for pattern in "$@"; do patterns+=(--allow "$pattern"); done

attempt=1
while true; do
  if "${HF_ENV}" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
    --root "${ROOT_DIR}" --repo nvidia/Cosmos3-DROID --repo-type dataset \
    --destination "${DATASET_DIR}" "${patterns[@]}"; then
    break
  else
    status=$?
  fi
  if (( status == 2 )); then exit 2; fi
  if (( attempt >= MAX_ATTEMPTS )); then
    echo "Cosmos3-DROID download failed after ${MAX_ATTEMPTS} attempts." >&2
    exit 1
  fi
  echo "Download attempt ${attempt}/${MAX_ATTEMPTS} failed; resuming after 20 seconds..." >&2
  attempt=$((attempt + 1))
  sleep 20
done

# The shared helper retains resume metadata and creates a commit/hash manifest
# only after every selected file passes its upstream checksum. Legacy marker
# files alone are never treated as verified completion.

echo
echo "Cosmos3-DROID download finished."
echo "Dataset:   ${DATASET_DIR}"
echo "HF mirror: ${HF_ENDPOINT}"
echo "Version and file hashes: ${DATASET_DIR}/.vlaworldeval-download.json"
