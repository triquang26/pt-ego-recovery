REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO
export XDG_CACHE_HOME="$REPO/.cache"
export UV_CACHE_DIR="$REPO/.cache/uv"
export HF_HOME="$REPO/.cache/huggingface"
export OPENPI_DATA_HOME="$REPO/.cache/openpi"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PY="$REPO/third_party/openpi/.venv/bin/python"
