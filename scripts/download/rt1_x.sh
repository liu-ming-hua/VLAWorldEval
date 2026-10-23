#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
PYTHON="${RT1_DOWNLOAD_PYTHON:-$ROOT_DIR/environment/core/bin/python}"
ASSETS="${RT1_ASSET_ROOT:-$ROOT_DIR/checkpoints/vla/rt1-x}"
ASSETS="$(realpath -m "$ASSETS")"
if [[ "$ASSETS" != "$ROOT_DIR/"* || ! -x "$PYTHON" ]]; then
  echo "Use a project-local RT1_ASSET_ROOT and an installed RT1_DOWNLOAD_PYTHON" >&2; exit 1
fi
ARCHIVES="$ROOT_DIR/.cache/model_archives/rt1_x"
mkdir -p "$ARCHIVES"
cd "$ROOT_DIR"
# These official archives are not on HF; use direct HTTPS, never a proxy or
# an unverified third-party HF reupload. GCS generation/checksums pin the files.
"$PYTHON" -m vlaworldeval.http_download --root "$ROOT_DIR" \
  --output "$ARCHIVES/rt1_x_1696960741689538_verified.zip" \
  --url 'https://storage.googleapis.com/gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip?generation=1696960741689538' \
  --size 774610722 --md5 'AiGPtJYlimo3qOFNGbWZbA=='
"$PYTHON" -m vlaworldeval.archive --root "$ROOT_DIR" --archive "$ARCHIVES/rt1_x_1696960741689538_verified.zip" \
  --destination "$ASSETS/rt_1_x_tf_trained_for_002272480_step" --md5 'AiGPtJYlimo3qOFNGbWZbA==' \
  --strip-prefix rt_1_x_tf_trained_for_002272480_step
# tfhub.dev's old /5 tarball is no longer public. The official redirect points
# to this Google Kaggle large/2 export; the transport checksum is fixed below.
"$PYTHON" -m vlaworldeval.http_download --root "$ROOT_DIR" --output "$ARCHIVES/use_large_v2.tar.gz" \
  --url 'https://www.kaggle.com/api/v1/models/google/universal-sentence-encoder/tensorFlow2/large/2/download' \
  --size 550093654 --md5 'LL4OiUgnIwHaf1RFVNMowQ=='
"$PYTHON" -m vlaworldeval.archive --root "$ROOT_DIR" --archive "$ARCHIVES/use_large_v2.tar.gz" \
  --destination "$ASSETS/universal_sentence_encoder_large_v5" --md5 'LL4OiUgnIwHaf1RFVNMowQ=='
echo "RT-1-X and its text encoder verified. No model inference performed."
