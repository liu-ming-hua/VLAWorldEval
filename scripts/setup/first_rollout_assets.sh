#!/usr/bin/env bash
# Explicit downloads only; inference lives in scripts/acceptance/first_rollout.py.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export UV_HTTP_TIMEOUT=600
export UV_INDEX_URL="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0= GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1=
CORE="$ROOT_DIR/environment/core/bin/python"
if [[ ! -x "$CORE" ]]; then
  echo 'Run bash scripts/setup/core.sh first' >&2
  exit 1
fi
bash scripts/setup/worldgym_runtime.sh
"$CORE" scripts/download/worldgym_data.py
HELPER="$ROOT_DIR/environment/tools/first_rollout_assets"
if [[ ! -x "$HELPER/bin/python" ]]; then
  uv venv "$HELPER" --python "$CORE"
fi
uv pip install --python "$HELPER/bin/python" --index-url "$UV_INDEX_URL" \
  'huggingface_hub==0.34.4' 'gdown==6.2.0'
"$HELPER/bin/python" scripts/download/worldgym_checkpoint.py
# The content lock is authoritative even if upstream changes an alias.
if [[ ! -s checkpoints/vla/openvla-7b/model.safetensors.index.json ]]; then
  "$HELPER/bin/python" scripts/download/hf_snapshot.py --root "$ROOT_DIR" \
    --repo openvla/openvla-7b --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
    --destination "$ROOT_DIR/checkpoints/vla/openvla-7b"
fi
VAE="$ROOT_DIR/checkpoints/world_models/worldgym/vae/sd3_vae"
if [[ ! -s "$VAE/config.json" || ! -s "$VAE/diffusion_pytorch_model.safetensors" ]]; then
  "$HELPER/bin/python" scripts/download/hf_snapshot.py --root "$ROOT_DIR" \
    --repo stabilityai/stable-diffusion-3-medium-diffusers \
    --destination "$ROOT_DIR/checkpoints/world_models/worldgym/vae/sd3_snapshot" \
    --allow 'vae/config.json' --allow 'vae/diffusion_pytorch_model.safetensors'
  mkdir -p "$VAE"
  cp -n checkpoints/world_models/worldgym/vae/sd3_snapshot/vae/config.json "$VAE/config.json"
  cp -n checkpoints/world_models/worldgym/vae/sd3_snapshot/vae/diffusion_pytorch_model.safetensors "$VAE/diffusion_pytorch_model.safetensors"
fi
"$CORE" - <<'PY'
import json
from pathlib import Path
from scripts.acceptance.first_rollout import verify_assets
root = Path.cwd()
print(verify_assets(root, json.loads((root/'configs/first_rollout_assets_v1.json').read_text())))
PY
