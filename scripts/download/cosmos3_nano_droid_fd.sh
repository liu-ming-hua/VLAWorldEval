#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/checkpoints/world_models/cosmos3_nano_droid_fd/community_jere_mybao_v1_20k_ema"
LOG_DIR="$PROJECT_ROOT/results/downloads"
mkdir -p "$MODEL_DIR" "$LOG_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT="https://hf-mirror.com"
ENV_DIR="${COSMOS_FD_ENV_DIR:-$PROJECT_ROOT/environment/world_models/cosmos3_nano_droid_fd}"
PYTHON="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Run setup/cosmos3_nano_droid_fd.sh first, or set COSMOS_FD_ENV_DIR to its environment" >&2
  exit 1
fi
"$PYTHON" "$PROJECT_ROOT/scripts/download/hf_snapshot.py" --root "$PROJECT_ROOT" \
  --repo jere-mybao/cosmos3-nano-droid-forward-dynamics --destination "$MODEL_DIR"
# Fetch only ancillary assets, not the large base models. Each snapshot keeps
# its immutable revision and remote checksum manifest in the FD asset tree.
ASSETS="$PROJECT_ROOT/checkpoints/world_models/cosmos3_nano_droid_fd"
"$PYTHON" "$PROJECT_ROOT/scripts/download/hf_snapshot.py" --root "$PROJECT_ROOT" \
  --repo Wan-AI/Wan2.2-TI2V-5B --destination "$ASSETS/vae" --allow 'Wan2.2_VAE.pth'
"$PYTHON" "$PROJECT_ROOT/scripts/download/hf_snapshot.py" --root "$PROJECT_ROOT" \
  --repo nvidia/Cosmos3-Nano --destination "$ASSETS/tokenizer_snapshot" --allow 'text_tokenizer/*'
mkdir -p "$ASSETS/text_tokenizer"
for source in "$ASSETS/tokenizer_snapshot/text_tokenizer/"*; do
  target="$ASSETS/text_tokenizer/$(basename "$source")"
  if [[ -e "$target" ]]; then
    if ! cmp -s "$source" "$target"; then
      echo "Existing tokenizer differs from the verified snapshot: $target; preserving it" >&2
      exit 1
    fi
  else
    cp -n "$source" "$target"
  fi
done
echo "FD model and ancillary assets verified; this does not imply inference passed."
