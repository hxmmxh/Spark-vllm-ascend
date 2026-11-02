#!/usr/bin/env bash
# Install the deployment into the environment prepared by create_env.sh.
set -euo pipefail
pkg=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${1:?Usage: bootstrap_user.sh /absolute/work-directory}
case "$work" in /*) ;; *) echo 'Work directory must be absolute' >&2; exit 2;; esac
if [[ ! -x "$work/.venv/bin/python" ]]; then
  echo 'Create the Python environment with create_env.sh first.' >&2
  exit 2
fi
if [[ -e "$work/cann" ]]; then
  echo 'CANN installation already exists. Resume the failed stage using its logs.' >&2
  exit 2
fi
command -v uv >/dev/null || { echo 'uv is required to install pip into the deployment environment.' >&2; exit 2; }
mkdir -p "$work/logs" "$work/tmp"
export TMPDIR="$work/tmp"
export DEBUG=${DEBUG:-0}
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH
export VIRTUAL_ENV="$work/.venv" PATH="$work/.venv/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$work/cache/uv}"
"$work/.venv/bin/python" -c 'import sys; from pathlib import Path; assert sys.version_info[:2] == (3,12), sys.version; assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()' "$work/.venv"
profile_exports=$("$work/.venv/bin/python" "$pkg/src/tools/deployment/_project.py" --shell --check-host)
eval "$profile_exports"
fetch_options=()
if [[ "${SPARK_PUBLIC_PYPI_MIRROR:-0}" == 1 ]]; then fetch_options+=(--pypi-mirror); fi
"$work/.venv/bin/python" "$pkg/src/tools/deployment/fetch.py" --work "$work" --group environment "${fetch_options[@]}"
# Seed the hash-verified pip wheel into this environment without system pip.
uv pip install --no-config --python "$work/.venv/bin/python" \
  --no-index --no-deps "$work/downloads/pip-25.3-py3-none-any.whl"
export PIP_CACHE_DIR="$work/cache/pip"
export XDG_CACHE_HOME="$work/cache/xdg"
export CMAKE_BUILD_PARALLEL_LEVEL=64 MAX_JOBS=64
bash "$pkg/src/tools/deployment/install_cann.sh" "$work"
bash "$pkg/src/tools/deployment/install_python.sh" "$work"
