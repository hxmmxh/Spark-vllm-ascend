#!/usr/bin/env bash
# Create the Python environment before running any project Python code.
set -euo pipefail
work=${1:?Usage: create_env.sh /absolute/new-work-directory}
case "$work" in /*) ;; *) echo 'Work directory must be absolute' >&2; exit 2;; esac
python_request=${SPARK_PYTHON_VERSION:-3.12}
if [[ ! "$python_request" =~ ^3\.12(\.[0-9]+)?$ ]]; then
  echo 'SPARK_PYTHON_VERSION must be 3.12 or a 3.12.x release (cp312 wheels).' >&2
  exit 2
fi
if [[ -e "$work/.venv" || -L "$work/.venv" ]]; then
  echo 'Refusing an existing environment; reuse it or select a new work directory.' >&2
  exit 2
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 2; }
mkdir -p "$work/logs" "$work/tmp"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export UV_CACHE_DIR="${UV_CACHE_DIR:-$work/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$work/python}"
uv venv --no-config --no-project --managed-python --python "$python_request" "$work/.venv"
"$work/.venv/bin/python" - "$work" "$python_request" "$(uv --version)" <<'PY'
import json
import platform
import sys
import sysconfig
from pathlib import Path

work = Path(sys.argv[1]).resolve()
assert sys.version_info[:2] == (3, 12), sys.version
assert platform.python_implementation() == 'CPython'
assert Path(sys.prefix).resolve() == work / '.venv', sys.prefix
include = Path(sysconfig.get_path('include'))
assert (include / 'Python.h').is_file(), f'Python development headers missing: {include}'
record = {'manager': sys.argv[3], 'requested_python': sys.argv[2],
          'python': platform.python_version(), 'executable': sys.executable,
          'environment': sys.prefix, 'base_prefix': sys.base_prefix,
          'include': str(include), 'host_arch': platform.machine()}
(work / 'python-environment.json').write_text(json.dumps(record, indent=2) + '\n')
print('SPARK_PYTHON_ENV_READY', json.dumps(record))
PY
