REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO
export NVGL="${NVGL:-$(dirname "$REPO")/nvidia-gl-610.43.02}"
export XDG_CACHE_HOME="$REPO/.cache"
export UV_CACHE_DIR="$REPO/.cache/uv"
export HF_HOME="$REPO/.cache/huggingface"
export HF_LEROBOT_HOME="$REPO/data/lerobot"
export OPENPI_DATA_HOME="$REPO/.cache/openpi"
export MS_ASSET_DIR="$REPO/.cache/maniskill"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PYTHONNOUSERSITE=1
export VK_ICD_FILENAMES="$NVGL/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_DIRS="$NVGL/egl_vendor.d"
export LD_LIBRARY_PATH="$NVGL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export OPENPI_PY="$REPO/third_party/openpi/.venv/bin/python"
export SIM_PY="$REPO/.venv-sim/bin/python"
