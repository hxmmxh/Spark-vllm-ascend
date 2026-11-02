# Shared deployment steps

[中文](common.md) | **English** · [Platforms](README.en.md)

For everyday use, complete steps 1–4, then go directly to step 6 to start the service. Step 5 benchmarks are optional. See [everyday use](../../README.en.md#everyday-use) for client connections.

## 1. Create the uv environment and deployment copy

Set the variables in your platform guide first. uv obtains a managed CPython 3.12 and creates `$WORK/.venv`; the script checks its development headers. That environment then prepares the deployment copy. Set `SPARK_PYTHON_VERSION` to a specific `3.12.x` release if needed.

```bash
bash "$BASE/src/tools/deployment/create_env.sh" "$WORK"
"$WORK/.venv/bin/python" "$BASE/src/tools/deployment/prepare_profile.py" --profile "$PROFILE" --dest "$REPO"
(cd "$REPO" && sha256sum -c DEPLOYMENT_SHA256SUMS)
export DEPLOY="$REPO/src/tools/deployment"
export VALIDATE="$REPO/src/tools/validation"
export NATIVE="$REPO/src/spark_vllm_ascend/native"
run() { bash "$DEPLOY/run_user.sh" "$WORK" "$DEVICE" "$@"; }
```

`run_user.sh` assigns an IPC directory at `/tmp/spark-ipc-<uid>-<hash>` for each work directory, keeping Unix socket paths within 107 bytes even with deeply nested work paths. To override it, set `VLLM_RPC_BASE_PATH` to a short absolute path owned by the current user; the launcher checks its length and permissions. It also merges existing `NO_PROXY` / `no_proxy` entries to keep loopback requests direct.

## 2. Install the environment and frameworks

Using the uv environment from step 1, the script downloads and verifies CANN, NNAL, DSL, and the profile-specific Triton wheel, installs locked dependencies, and builds vLLM / vLLM-Ascend from source. uv installs the verified pip wheel into this environment; subsequent dependency installation preserves the order required by Ascend/Triton. Native builds, serving, and tests use the same environment.

Both initial installation and standalone Python build recovery use 64 parallel build jobs: `CMAKE_BUILD_PARALLEL_LEVEL=64` and `MAX_JOBS=64`.

The installer rechecks and installs the downloaded DSL / ARM Triton wheels before resolving other dependencies, preventing pip from downloading the same packages again from an index. Other dependencies follow the lock file; the ARM Triton backend is restored after dependency and source installation.

```bash
DEBUG=0 PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=5 \
  bash "$DEPLOY/bootstrap_user.sh" "$WORK" > "$WORK/logs/install.log" 2>&1
run python -m pip check
run python "$DEPLOY/check_runtime.py" --work "$WORK" --profile "$PROFILE" \
  > "$WORK/logs/check-runtime.log" 2>&1
cat "$WORK/logs/check-runtime.log"
```

Expect `SPARK_USER_ENVIRONMENT_BUILT`, a clean `pip check`, and `RUNTIME_IMPORTS_PASS`.

## 3. Prepare the model

Expected markers: `MODEL_PREPARED 290` and `HEAD_CPU_EXACT`. Do not repeat preparation over existing `model/` or `head/` directories. If Hugging Face downloads fail, add `--hf-mirror` to fetch.py; hashes remain unchanged.

```bash
run python "$DEPLOY/fetch.py" --work "$WORK" --group model > "$WORK/logs/download-model.log" 2>&1
run python "$DEPLOY/prepare_model.py" --work "$WORK" > "$WORK/logs/prepare-model.log" 2>&1
run python "$DEPLOY/prepare_head.py" --work "$WORK" > "$WORK/logs/prepare-head.log" 2>&1
```

## 4. Build and validate on the device

The guard waits for an idle target NPU and invalidates a run if another process appears or telemetry fails. Expected markers: `HEAD_STRICT_PASS` and `SLOT4_STRICT_PASS`.

```bash
run python "$NATIVE/head/build.py" > "$WORK/logs/build-head.log" 2>&1
run python "$NATIVE/head_reference/build.py" > "$WORK/logs/build-head-reference.log" 2>&1
run python "$VALIDATE/guard.py" --out "$WORK/guards/head-01.json" -- \
  python "$VALIDATE/test_head.py" --work "$WORK" > "$WORK/logs/head-01.log" 2>&1
run python "$VALIDATE/guard.py" --out "$WORK/guards/slot4-01.json" -- \
  python "$VALIDATE/test_slot_mapping.py" > "$WORK/logs/slot4-01.log" 2>&1
```

## 5. Run three benchmark rounds (optional)

The controller starts three independent services, warming up each once. Do not run the persistent service from the next section at the same time.

The benchmark client connects directly for local health checks, model queries, and streaming requests, ignoring inherited HTTP proxy settings.

```bash
export BENCH_STAMP="$(date '+%Y%m%d-%H%M%S')"
run python "$REPO/src/tools/benchmark/benchmark.py" \
  --work "$WORK" --repo "$REPO" --port "$PORT" --rounds 3 \
  > "$WORK/logs/benchmark-$BENCH_STAMP.log" 2>&1
"$WORK/.venv/bin/python" "$DEPLOY/verify_results.py" --work "$WORK" --repo "$REPO" --device "$DEVICE"
cat "$WORK/benchmark-results.csv"
```

Acceptance requires `complete=true` in `deployment-benchmark-verification.json`. Keep `measurements/`, `service/`, and `guards/`; archive existing summaries before repeating this step. Full-model reference accuracy validation is not included.

## 6. Start an interactive service

Run this after completing steps 1–4. The service runs in the foreground, with logs at `$RUN_OUT/server.log`; press Ctrl-C to stop it. To restart later, restore your deployment variables and the `run` function. Stop this service before running formal benchmarks.

```bash
export RUN_ID="serve-$(date '+%Y%m%d-%H%M%S')"
export RUN_OUT="$WORK/service/$RUN_ID"
mkdir -p "$RUN_OUT"
run python "$VALIDATE/guard.py" --out "$WORK/guards/$RUN_ID.json" -- \
  python "$DEPLOY/serve.py" --work "$WORK" --out "$RUN_OUT" --port "$PORT" \
  > "$RUN_OUT/server.log" 2>&1
```

In another terminal, set the same port and wait for `/health` to return HTTP 200 before sending a request:

```bash
export PORT=18526
curl --noproxy 127.0.0.1 -fsS "http://127.0.0.1:$PORT/health"
curl --noproxy 127.0.0.1 -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark","messages":[{"role":"user","content":"Explain matrix multiplication briefly."}],"temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Recovery and repeat runs

- Python environment creation: inspect uv download logs and rerun step 1 with a new work directory. The script refuses to overwrite an existing `.venv`. Interpreter details are recorded in `$WORK/python-environment.json`.
- Downloads: repeat `fetch.py --work "$WORK" --group environment`. Retain failed files and inspect logs if checksum verification fails.
- Python build: after checking CANN/NNAL installation, run `DEBUG=0 bash "$DEPLOY/install_python.sh" "$WORK"`. This also restores the ARM Triton backend. Repeat runtime checks afterward.
- `Another process is using this dir` after an interrupted build: rerun `install_python.sh`. Source builds take an exclusive lock on the work directory and archive `kernel_meta.lock` files from confirmed exited processes under `$WORK/build-recovery/`. Locks with live or unverifiable owners are retained and reported; wait for that build to finish before resuming.
- Missing `libatb.so`: inspect `install-nnal.log` and the profile's NNAL environment script. An existing directory alone does not prove installation succeeded.
- Failed device checks: diagnose the logs and use fresh guard filenames. Rebuilding the head requires rerunning its validation. A failed benchmark batch must be rerun as a whole.
- Repeat benchmarks: reuse the original deployment copy and work directory, restore the variables and `run` function above, then execute step 5. Do not recreate the environment or overwrite prepared weights.
