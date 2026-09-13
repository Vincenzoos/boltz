#!/usr/bin/env bash
# Install Boltz into a conda env named "boltz" and prepare
# notebooks/Boltz_Remodel_UI.ipynb for use.
#
# Usage:
#   ./install_boltz.sh              # CUDA build (default)
#   ./install_boltz.sh --cpu        # CPU-only (no cuequivariance)
#   ./install_boltz.sh --force      # recreate env if it already exists
#   ./install_boltz.sh --python 3.11
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="boltz"
PYTHON_VERSION="3.11"
USE_CUDA=1
FORCE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --cpu           Install without [cuda] extras (CPU / non-CUDA GPUs)
  --cuda          Install with [cuda] extras (default)
  --force         Remove and recreate the conda env if it exists
  --python VER    Python version for the env (default: ${PYTHON_VERSION}; must be 3.10–3.12)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) USE_CUDA=0; shift ;;
    --cuda) USE_CUDA=1; shift ;;
    --force) FORCE=1; shift ;;
    --python)
      PYTHON_VERSION="${2:?--python requires a version}"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${PYTHON_VERSION}" in
  3.10|3.11|3.12) ;;
  *)
    echo "Error: Python must be 3.10, 3.11, or 3.12 (got ${PYTHON_VERSION})" >&2
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# Make conda available in non-interactive shells
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "==> Removing existing conda env '${ENV_NAME}'"
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "==> Conda env '${ENV_NAME}' already exists (use --force to recreate)"
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "==> Activating '${ENV_NAME}'"
conda activate "${ENV_NAME}"

echo "==> Upgrading pip / setuptools / wheel"
python -m pip install -U pip setuptools wheel

EXTRAS=""
if [[ "${USE_CUDA}" -eq 1 ]]; then
  EXTRAS="[cuda]"
  echo "==> Installing boltz editable with CUDA extras"
else
  echo "==> Installing boltz editable (CPU / no CUDA extras)"
fi

cd "${SCRIPT_DIR}"
python -m pip install -e ".${EXTRAS}"

echo "==> Installing notebook / UI dependencies"
python -m pip install \
  'ipywidgets>=8' \
  jupyterlab_widgets \
  widgetsnbextension \
  pyyaml \
  ipykernel \
  jupyterlab \
  notebook

echo "==> Registering Jupyter kernel '${ENV_NAME}'"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (boltz)"

echo
echo "Done."
echo "  conda activate ${ENV_NAME}"
echo "  jupyter lab notebooks/Boltz_Remodel_UI.ipynb"
echo "  # or open the notebook in Cursor/VS Code and select kernel: Python (boltz)"
if [[ "${USE_CUDA}" -eq 1 ]]; then
  echo
  echo "Note: CUDA extras require a matching NVIDIA driver / CUDA 12 stack."
  echo "      If install fails, re-run with: ./install_boltz.sh --cpu"
fi
