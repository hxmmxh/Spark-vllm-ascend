#!/usr/bin/env bash
# Install public, hash-verified toolkit/ops in this user's dedicated directory.
set -euo pipefail
pkg=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${1:?Missing work directory}
[[ -x "$work/.venv/bin/python" ]] || { echo 'Create the deployment environment first.' >&2; exit 2; }
export VIRTUAL_ENV="$work/.venv" PATH="$work/.venv/bin:$PATH"
profile_exports=$("$work/.venv/bin/python" "$pkg/src/tools/deployment/_project.py" --shell --check-host)
eval "$profile_exports"
for installer in \
  "$SPARK_CANN_TOOLKIT" \
  "$SPARK_CANN_OPS"; do
  bash "$work/downloads/$installer" --install --quiet --install-path="$work/cann" \
    > "$work/logs/$installer.log" 2>&1
done
# Only permit a compatibility alias whose target is inside this fresh install.
if [[ ! -f "$work/cann/ascend-toolkit/set_env.sh" ]]; then
  if [[ -f "$work/cann/cann/set_env.sh" && ! -e "$work/cann/ascend-toolkit" ]]; then
    ln -s cann "$work/cann/ascend-toolkit"
  fi
fi
[[ -f "$work/cann/ascend-toolkit/set_env.sh" ]]
"$work/.venv/bin/python" - "$work" <<'PY'
from pathlib import Path
import sys
w=Path(sys.argv[1]).resolve();env=w/'cann/ascend-toolkit/set_env.sh'
assert env.resolve().is_relative_to(w/'cann'),('Unexpected external CANN alias',str(env.resolve()))
PY

set +u
source "$work/cann/ascend-toolkit/set_env.sh"
set -u
export VIRTUAL_ENV="$work/.venv" PATH="$work/.venv/bin:$PATH"
bash "$work/downloads/$SPARK_NNAL_INSTALLER" \
  --install --quiet --whitelist=atb --install-path="$work/cann" \
  > "$work/logs/install-nnal.log" 2>&1
[[ -f "$work/cann/$SPARK_NNAL_ENV" ]]
