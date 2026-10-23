#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
PYTHON="$ROOT_DIR/environment/tools/huggingface/bin/python"
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/download/droid_specialists.sh groot_n17_droid|pi0_fast_droid|fast_tokenizer|cosmos_policy_droid|dreamzero_droid|dreamzero_tokenizer|cosmos_joint_fd" >&2
  exit 2
fi
select_model() {
  case "$1" in
    groot_n17_droid) repo=nvidia/GR00T-N1.7-DROID; revision=05e7cc97e40dbd33b0890c35cc0214fcb0547ab5; destination=checkpoints/vla/groot/n1.7-droid ;;
    pi0_fast_droid) repo=RomanticQq/pi0_fast_droid; revision=db07bbbf6da1d16c3a0ee9fe087ee3bea192f8a8; destination=checkpoints/vla/openpi/pi0-fast-droid ;;
    fast_tokenizer) repo=physical-intelligence/fast; revision=main; destination=checkpoints/vla/openpi/fast-tokenizer ;;
    cosmos_policy_droid) repo=nvidia/Cosmos3-Nano-Policy-DROID; revision=805c0d6d46196ecdc789213a6de72303c383895b; destination=checkpoints/vla/cosmos3/nano-policy-droid ;;
    dreamzero_droid) repo=GEAR-Dreams/DreamZero-DROID; revision=96ad344138c66e82536422432ad742f015784942; destination=checkpoints/vla/dreamzero/droid ;;
    dreamzero_tokenizer) repo=google/umt5-xxl; revision=66cb9e7e85526fe440a945569e42c72fb6cbc0ad; destination=checkpoints/vla/dreamzero/umt5-tokenizer ;;
    cosmos_joint_fd) repo=anshks/cosmos3-nano-droid-fd-wm; revision=959e9e7245b979bb8d3df0917e2e649573c0fde9; destination=checkpoints/world_models/cosmos3_nano_droid_joint_fd/community_anshks_v1_best5600 ;;
    *) echo "Unknown DROID model: $1" >&2; exit 2 ;;
  esac
}
for model in "$@"; do select_model "$model"; done
for model in "$@"; do
  select_model "$model"
  patterns=()
  if [[ "$model" == cosmos_joint_fd ]]; then patterns=(--allow README.md --allow 'best-iter5600/*'); fi
  if [[ "$model" == dreamzero_tokenizer ]]; then
    patterns=(--allow 'tokenizer*' --allow spiece.model --allow special_tokens_map.json --allow added_tokens.json)
  fi
  downloaded=false
  for attempt in 1 2 3 4 5; do
    if "$PYTHON" "$ROOT_DIR/scripts/download/hf_snapshot.py" --root "$ROOT_DIR" \
      --repo "$repo" --revision "$revision" --destination "$ROOT_DIR/$destination" "${patterns[@]}"; then
      downloaded=true
      break
    fi
    if [[ "$attempt" -lt 5 ]]; then
      echo "Mirror transfer interrupted: $model; retaining partial files, retry $((attempt + 1))/5" >&2
      sleep 15
    fi
  done
  [[ "$downloaded" == true ]] || exit 1
  if [[ "$model" == pi0_fast_droid ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/checks/openpi_mirror.py"
  fi
done
