#!/usr/bin/env bash
# Install Boltz into a conda env under this project directory and prepare
# notebooks/Boltz_Remodel_UI.ipynb for use.
#
# Usage:
#   ./install_boltz.sh              # CUDA build (default)
#   ./install_boltz.sh --cpu        # CPU-only (no cuequivariance)
#   ./install_boltz.sh --force      # recreate env if it already exists
#   ./install_boltz.sh --python 3.11
#
# The environment and Conda package cache default to storage rather than
# the user's home filesystem. Override them with BOLTZ_ENV_PREFIX or
# BOLTZ_CONDA_PKGS_DIR if needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="boltz"
ENV_PREFIX="${BOLTZ_ENV_PREFIX:-${SCRIPT_DIR}/.conda-envs/${ENV_NAME}}"
CONDA_PKGS_DIR="${BOLTZ_CONDA_PKGS_DIR:-${SCRIPT_DIR}/.conda-pkgs}"
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
  echo "This script will not install Conda itself." >&2
  exit 1
fi

# Make conda available in non-interactive shells.
# shellcheck disable=SC1091
CONDA_BASE="$(conda info --base)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Error: conda shell integration not found under '${CONDA_BASE}'." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p "$(dirname "${ENV_PREFIX}")" "${CONDA_PKGS_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"

if [[ -d "${ENV_PREFIX}" ]]; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "==> Removing existing conda env '${ENV_PREFIX}'"
    conda env remove -p "${ENV_PREFIX}" -y
  else
    echo "==> Conda env '${ENV_PREFIX}' already exists (use --force to recreate)"
  fi
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "==> Creating conda env '${ENV_PREFIX}' (Python ${PYTHON_VERSION})"
  conda create -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip -y
fi

echo "==> Activating '${ENV_PREFIX}'"
conda activate "${ENV_PREFIX}"

EXTRAS=""
if [[ "${USE_CUDA}" -eq 1 ]]; then
  EXTRAS="[cuda]"
  echo "==> Installing boltz editable with CUDA extras"
else
  echo "==> Installing boltz editable (CPU / no CUDA extras)"
fi

cd "${SCRIPT_DIR}"
echo "==> Installing Boltz dependencies without a pip cache"
python -m pip install --no-cache-dir -e ".${EXTRAS}"

echo "==> Installing minimal notebook / UI dependencies without a pip cache"
python -m pip install --no-cache-dir \
  'ipywidgets>=8' \
  ipykernel \
  jupyterlab

echo "==> Registering Jupyter kernel '${ENV_NAME}'"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (boltz)"

echo
echo "Done."
echo "  conda activate ${ENV_PREFIX}"
echo "  jupyter lab notebooks/Boltz_Remodel_UI.ipynb"
echo "  # or open the notebook in Cursor/VS Code and select kernel: Python (boltz)"
if [[ "${USE_CUDA}" -eq 1 ]]; then
  echo
  echo "Note: CUDA extras require a matching NVIDIA driver / CUDA 12 stack."
  echo "      If install fails, re-run with: ./install_boltz.sh --cpu"
fi
