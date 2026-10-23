#!/usr/bin/env bash
set -euo pipefail

# Create one isolated uv environment per VLA family. The environments are
# deliberately separate: OpenVLA/Octo use Transformers 4.49, SpatialVLA pins
# 4.48.1, and Octo/RT-1 require incompatible JAX/Flax/TensorFlow stacks.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${VLA_ENV_ROOT:-${ROOT_DIR}/environment/vla}"
ENV_ROOT="$(realpath -m "${ENV_ROOT}")"
if [[ "${ENV_ROOT}" != "${ROOT_DIR}/environment/"* ]]; then
  echo "VLA_ENV_ROOT must remain inside the project's environment directory" >&2
  exit 1
fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CACHE_DIR="${ROOT_DIR}/.uv-cache"

# Direct-connect only: do not inherit a local VPN or SOCKS proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_INDEX_URL="${PYPI_MIRROR}"
export PIP_INDEX_URL="${PYPI_MIRROR}"
export UV_CACHE_DIR="${CACHE_DIR}"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0=
export GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1=
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/huggingface"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}; set UV_BIN=/path/to/uv" >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${CACHE_DIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

install_env() {
  local name="$1"
  local extra="$2"
  local env_dir="${ENV_ROOT}/${name}"
  local override_args=()
  if [[ "${name}" == "openvla" ]]; then
    override_args+=(--constraint "${ROOT_DIR}/environment/locks/worldgym-openvla-py311.txt")
  elif [[ "${name}" == "rt1" ]]; then
    override_args+=(--override "${ROOT_DIR}/environment/locks/rt1-overrides.txt")
  elif [[ "${name}" == "octo" ]]; then
    override_args+=(--override "${ROOT_DIR}/environment/locks/octo-overrides.txt")
  fi

  echo "==> ${name}: ${env_dir}"
  mkdir -p "${ENV_ROOT}"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    "${UV_BIN}" venv "${env_dir}" --python "${PYTHON_VERSION}"
  fi

  # Non-editable installation keeps generated *.egg-info outside the source
  # tree and makes each model environment independently reproducible.
  "${UV_BIN}" pip install \
    --python "${env_dir}/bin/python" \
    --index-url "${PYPI_MIRROR}" \
    --link-mode=copy \
    "${override_args[@]}" \
    ".[${extra},worldgym,scoring]"
  if [[ "${name}" == "octo" ]]; then
    # Octo's released typing module refers to the removed JAX KeyArray alias.
    # The upstream README documents this compatibility fix for newer JAX.
    local typing_file="${env_dir}/lib/python${PYTHON_VERSION}/site-packages/octo/utils/typing.py"
    if [[ -f "${typing_file}" ]]; then
      sed -i 's/jax\.random\.KeyArray/jax.random.PRNGKey/' "${typing_file}"
    fi
  fi
  if [[ "${name}" == "rt1" ]]; then
    # The official RT-1-X release is a TensorFlow SavedModel.  TF-Agents' 
    # optional dependencies are not always pulled in by uv when the package
    # is installed from a mixed JAX/TF environment, so keep the runtime
    # requirements explicit and reproducible.
    "${UV_BIN}" pip install \
      --python "${env_dir}/bin/python" \
      --index-url "${PYPI_MIRROR}" \
      --link-mode=copy \
      "decorator==5.3.1" "dm-tree==0.1.10" "cloudpickle==3.1.2"
  fi
  echo "    ready: ${env_dir}/bin/python"
}

# Install only requested families, not all incompatible stacks by default.
# With no argument, preserve the existing four-family bootstrap behavior.
requested=("$@")
if [[ ${#requested[@]} -eq 0 ]]; then
  requested=(openvla spatialvla rt1 octo)
fi
for task_family in "${requested[@]}"; do
  case "${task_family}" in
    openvla|spatialvla|rt1|octo) ;;
    *) echo "Unknown VLA family: ${task_family}; no installation performed" >&2; exit 1 ;;
  esac
done
for task_family in "${requested[@]}"; do
  install_env "${task_family}" "${task_family}"
done

echo
echo "VLA environments are ready. Proxy variables were disabled."
echo "PyPI mirror: ${PYPI_MIRROR}"
