#!/usr/bin/env bash
set -euo pipefail

# Fast, weight-free project gate. Model-specific integration tests remain
# opt-in because their environments and checkpoints are intentionally isolated.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/environment/core/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/environment/core/bin/python"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif [[ -x "${ROOT_DIR}/environment/vla/openvla/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/environment/vla/openvla/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "${ROOT_DIR}"
echo "[1/5] shell syntax"
for task_script in scripts/setup/*.sh scripts/download/*.sh scripts/serve/*.sh scripts/ops/*.sh scripts/check_project.sh; do
  bash -n "${task_script}"
done

echo "[2/5] Python syntax"
"${PYTHON_BIN}" -m compileall -q adapters cli evaluation scripts vlaworldeval tests

echo "[3/5] static checks"
"${PYTHON_BIN}" -m ruff check adapters cli evaluation scripts tests vlaworldeval

echo "[4/5] framework tests"
"${PYTHON_BIN}" -m pytest

echo "[5/5] project metadata"
"${PYTHON_BIN}" -m cli validate >/dev/null
"${PYTHON_BIN}" scripts/checks/docs.py

echo "Project checks passed."
