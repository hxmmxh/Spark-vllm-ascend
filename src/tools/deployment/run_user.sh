#!/usr/bin/env bash
# Select a physical device and use only the selected deployment environment.
set -euo pipefail
pkg=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${1:?Usage: run_user.sh WORK PHYSICAL_DEVICE COMMAND...}
device=${2:?Missing physical device}; shift 2
[[ "$work" = /* && "$device" =~ ^[0-9]+$ && $# -gt 0 ]]
[[ -x "$work/.venv/bin/python" ]] || { echo 'Create the deployment environment first.' >&2; exit 2; }
export PATH="$work/.venv/bin:$PATH"
unset VLLM_VERSION VLLM_VERSION_OVERRIDE SETUPTOOLS_SCM_PRETEND_VERSION
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH
export VIRTUAL_ENV="$work/.venv"
export PYTHONNOUSERSITE=1
while IFS= read -r spark_env_name; do
  case "$spark_env_name" in CANNBOTDSL_*|TRITON_*) unset "$spark_env_name";; esac
done < <(compgen -e)
mkdir -p "$work/tmp" "$work/cache"
export TMPDIR="$work/tmp" TRITON_CACHE_DIR="$work/cache/triton" TRITON_OVERRIDE_DIR="$work/cache/triton-override" TRITON_DUMP_DIR="$work/cache/triton-dump"
export TORCH_EXTENSIONS_DIR="$work/cache/torch-extensions" TORCH_HOME="$work/cache/torch" HF_HOME="$work/cache/hf"
profile_exports=$("$work/.venv/bin/python" "$pkg/src/tools/deployment/_project.py" --shell --check-host)
eval "$profile_exports"
export CANNBOTDSL_CACHE_DIR="$work/cache/cannbot"
set +u
source "$work/cann/ascend-toolkit/set_env.sh"
source "$work/cann/$SPARK_NNAL_ENV" --cxx_abi=1
set -u
export VIRTUAL_ENV="$work/.venv" PATH="$work/.venv/bin:$PATH"
eval "$profile_exports"
export SPARK_WORK="$work" ASCEND_RT_VISIBLE_DEVICES="$device" SPARK_PHYSICAL_DEVICE="$device"
runtime_exports=$("$work/.venv/bin/python" "$pkg/src/tools/deployment/runtime_environment.py" --work "$work")
eval "$runtime_exports"
# The editable installation supplies spark_vllm_ascend in every vLLM process.
export XDG_CACHE_HOME="$work/cache/xdg" PIP_CACHE_DIR="$work/cache/pip"
export VLLM_CACHE_ROOT="$work/cache/vllm" ASCEND_CACHE_PATH="$work/cache/ascend" TORCHINDUCTOR_CACHE_DIR="$work/cache/inductor"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_USE_V2_MODEL_RUNNER=0
exec "$@"
