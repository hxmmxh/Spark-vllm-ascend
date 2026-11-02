#!/usr/bin/env bash
# Resume only the Python build stage in an already freshly installed user environment.
set -euo pipefail
pkg=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${1:?Missing work directory}
index=${SPARK_PYPI_INDEX:-https://repo.huaweicloud.com/repository/pypi/simple}
[[ -x "$work/.venv/bin/python" ]] || { echo 'Create the deployment environment first.' >&2; exit 2; }
export PATH="$work/.venv/bin:$PATH" PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH
export VIRTUAL_ENV="$work/.venv"
profile_exports=$(python "$pkg/src/tools/deployment/_project.py" --shell --check-host)
eval "$profile_exports"
set +u
source "$work/cann/ascend-toolkit/set_env.sh"
source "$work/cann/$SPARK_NNAL_ENV" --cxx_abi=1
set -u
export VIRTUAL_ENV="$work/.venv" PATH="$work/.venv/bin:$PATH"
eval "$profile_exports"
export PIP_CACHE_DIR="$work/cache/pip" XDG_CACHE_HOME="$work/cache/xdg" CMAKE_BUILD_PARALLEL_LEVEL=64 MAX_JOBS=64
mkdir -p "$work/tmp"
export TMPDIR="$work/tmp" GIT_CONFIG_GLOBAL="$work/git-build.config"
export SPARK_PYPI_INDEX="$index"
export PIP_CONSTRAINT="$pkg/deployment/resolved/constraints-python.txt"
python "$pkg/src/tools/deployment/local_wheels.py" --work "$work"
python -m pip install --index-url "$index" 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1' 'cmake==3.31.6' 'ninja==1.13.0' 'setuptools-scm==9.2.2' 'setuptools-rust==1.12.0' 'packaging==25.0' 'pybind11==3.0.1' 'nanobind==2.9.2' 'grpcio-tools==1.78.0'
python -m pip install --no-deps --require-hashes -r "$pkg/deployment/resolved/cpu-wheels.lock.txt"
# Explicit local requirements prevent pip from downloading the same DSL/Ascend wheel again.
# The base Triton dependency may overwrite its backend below; restore it afterward as usual.
python -m pip install --no-index --no-deps --require-hashes -r "$work/local-wheels.lock.txt"
python -m pip install --index-url "$index" \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  -r "$pkg/deployment/resolved/requirements.txt"
# Source builds can import Triton, so restore the backend before and after them.
restore_triton_backend() {
  if [[ -n "$SPARK_TRITON_WHEEL" ]]; then
    python -m pip install --no-index --no-deps --force-reinstall "$work/downloads/$SPARK_TRITON_WHEEL"
  fi
}
restore_triton_backend
# Reinstall the audited, hash-locked compiler wheel rather than a floating index artifact.
python -m pip install --no-deps --force-reinstall "$work/downloads/$SPARK_DSL_WHEEL"
python "$pkg/src/tools/deployment/install_sources.py" --work "$work"
python -m pip install --no-build-isolation --no-deps -e "$pkg"
# Dependency installation may have overwritten the shared native backend again.
restore_triton_backend
python -m pip check
python -m pip freeze --all > "$work/pip-freeze.txt"
echo 'SPARK_USER_ENVIRONMENT_BUILT'
