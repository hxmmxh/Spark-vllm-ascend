---
name: deploy-spark-vllm-ascend
description: Deploy, resume installation, or start Spark inference on Ascend 950PR or 950DT using the Spark-vllm-ascend repository workflow. Use when the user asks to configure the inference environment, run the chat service, or repeat benchmarks.
---

# Deploy and use Spark

[中文](SKILL.md) | **English**

Use the repository scripts to deploy on the target Ascend 950 host and verify the service with a real chat request. Default to everyday use; run three benchmark rounds when the user requests performance testing.

## Locate the environment and guides

- Locate the repository from the user's path or current workspace and check that the project name in `pyproject.toml` is `spark-vllm-ascend`. Paths below are relative to the repository root. If this skill was copied separately, locate the full repository before reading its documents.
- Execution requires repository access and a terminal on the target host. A repository URL is enough to read the guides; if host access or hardware is missing, report that prerequisite without claiming deployment succeeded.
- Read the [requirements](../../deployment/README.en.md), then check `uname -m`, `npu-smi info`, and build tools to identify the actual environment. Select the guide using both the host architecture and NPU model:

| Host and NPU | Profile | Guide |
|---|---|---|
| x86_64 + Ascend950PR_957b | `x86-950pr` | [x86 deployment](../../deployment/x86-950pr.en.md) |
| aarch64 + Ascend950DT_9581 | `arm64-950dt` | [ARM deployment](../../deployment/arm64-950dt.en.md) |

Use the user's device, paths, and port when specified; otherwise choose an idle physical NPU, separate directories, and an available port. Device numbers in the guides are examples, not availability guarantees. Use the profile for SOC and locked dependencies rather than inferring the NPU model from the CPU architecture.

## Deploy or resume

Read the [shared deployment steps](../../deployment/common.en.md) and select the stages that match the task. Use the maintained documents and repository scripts for exact commands.

- **New deployment**: set `BASE`, `PROFILE`, `REPO`, `WORK`, `DEVICE`, and `PORT` from the platform guide, then complete steps 1–4. Create `$WORK/.venv` with uv, prepare the deployment copy, install frameworks, prepare the model, build native operators, and run device self-checks.
- **Start an existing environment**: locate its deployment copy and work directory, restore variables and the `run` function, and inspect `deployment-profile.json`, `python-environment.json`, and self-check records. Proceed to step 6 when the environment is complete. Do not repeat first-install checks that require nonexistent directories or overwrite an existing environment or weights.
- **Resume installation**: inspect the failed stage's logs and follow “Recovery and repeat runs” in the shared guide. Rerun the affected stage and its subsequent checks, preserving source files, dependency locks, and completed-stage records.

Use the same uv environment for installation, builds, serving, and tests; run device commands through the deployment copy's `run_user.sh`. Let the scripts manage CANN/NNAL, the Triton backend, communication directories, and proxy settings. Do not bypass them with system Python or a bare `vllm serve`. Preserve dependency installation order and download verification.

Check stage results such as `RUNTIME_IMPORTS_PASS`, `MODEL_PREPARED 290`, `HEAD_CPU_EXACT`, `HEAD_STRICT_PASS`, and `SLOT4_STRICT_PASS` in the logs. A checksum mismatch, unsupported platform, or failed self-check blocks dependent stages: keep the logs and report the specific cause instead of changing locks or skipping checks to continue.

## Start and verify the service

Follow step 6 in a terminal session that can remain running, and verify from another terminal:

1. Request `/health` on the actual `PORT`, wait for HTTP 200, and inspect `$RUN_OUT/server.log` during startup.
2. Send a `/v1/chat/completions` request with model `spark` using the [everyday-use example](../../../README.en.md#everyday-use). Confirm an actual valid reply; a running process alone is insufficient. Use the documented `--noproxy 127.0.0.1` for local curl requests.
3. Provide streaming, multi-turn, or SSH forwarding instructions as needed. The default service listens on `127.0.0.1`; use the documented tunnel for remote clients.

When performance testing is requested, follow step 5 for three rounds and `verify_results.py`, and require `complete=true` in its report. Do not run the step 6 service concurrently with benchmarks. Start the everyday service afterward if requested.

## Report the outcome

Provide the actual platform, physical device number, `REPO` / `WORK`, endpoint and model name, reusable start/stop instructions, log locations, and a request example. Distinguish completed checks from unverified items and failed stages. If benchmarks were not run, do not present the README's performance figures as results of this deployment.
