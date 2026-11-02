<p align="center">
  <img src="docs/spark-vllm-ascend-logo.png" alt="Spark vLLM Ascend" width="720">
</p>

# Spark-vllm-ascend

[中文](README.md) | **English**

Deploy **high-speed Spark inference on Ascend 950**, with support for x86_64 and ARM64 hosts. Built on vLLM / vLLM-Ascend, this repository provides installation, serving, and benchmarking workflows.

The current configuration supports **Spark-X2.6-4B Q4_0, one NPU, batch size 1, and BF16 KV**, with measured decode throughput of approximately **337–344 tokens/s** across the two platforms. See the test settings and results below.

## Deployment

| Platform | Guide |
|---|---|
| x86_64 + Ascend950PR_957b | [x86 deployment](docs/deployment/x86-950pr.en.md) |
| ARM64 + Ascend950DT_9581 | [ARM deployment](docs/deployment/arm64-950dt.en.md) |

```bash
git clone https://github.com/hxmmxh/Spark-vllm-ascend.git
cd Spark-vllm-ascend
```

Follow the platform guide to create a Python 3.12 environment with uv, select an idle physical NPU, and use a separate work directory. Both platforms use the same source with explicit dependency and build profiles. Model weights, CANN installers, and compiled libraries are obtained or built separately.

[Requirements and versions](docs/deployment/README.en.md) · [Benchmarks](docs/benchmarks.en.md)

## Deploy with an AI assistant

Use the [Spark deployment skill](docs/skills/deploy-spark-vllm-ascend/SKILL.en.md). Give it and the repository URL or local path to an AI coding assistant with file access and terminal execution on the target host, so it can follow the workflow to configure the environment, build, validate, and start the service.

## Everyday use

For the first deployment, follow your platform guide and steps 1–4 of the [shared deployment steps](docs/deployment/common.en.md), then start the service. Step 5 benchmarks are optional. Keep `REPO`, `WORK`, `DEVICE`, and `PORT` set in your deployment terminal and run:

```bash
export RUN_ID="serve-$(date '+%Y%m%d-%H%M%S')"
export RUN_OUT="$WORK/service/$RUN_ID"
mkdir -p "$RUN_OUT"
bash "$REPO/src/tools/deployment/run_user.sh" "$WORK" "$DEVICE" \
  python "$REPO/src/tools/validation/guard.py" --out "$WORK/guards/$RUN_ID.json" -- \
  python "$REPO/src/tools/deployment/serve.py" --work "$WORK" --out "$RUN_OUT" --port "$PORT" \
  > "$RUN_OUT/server.log" 2>&1
```

The service runs in the foreground, with logs at `$RUN_OUT/server.log`. Press Ctrl-C to stop it. To restart later, restore your actual `REPO`, `WORK`, `DEVICE`, and `PORT` values in Bash and repeat these commands.

In another terminal, set the port to the value used for deployment. Wait for `/health` to return HTTP 200 before sending a chat request:

```bash
export PORT=18526
curl --noproxy 127.0.0.1 -fsS "http://127.0.0.1:$PORT/health"
curl --noproxy 127.0.0.1 -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "spark",
    "messages": [{"role": "user", "content": "Explain matrix multiplication briefly."}],
    "temperature": 0,
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The reply is in `choices[0].message.content`. This example disables thinking mode for direct answers. For multi-turn conversations, include previous `user` / `assistant` messages in `messages`. To stream the reply, add `"stream": true` to the request and `-N` to curl.

Clients that support a custom OpenAI-compatible Chat Completions endpoint can use:

| Setting | Value |
|---|---|
| API Base URL | `http://127.0.0.1:18526/v1` (match your deployment port) |
| Model | `spark` |
| API Key | Not required by the default service; use `EMPTY` if the client requires a value |

The service listens on `127.0.0.1` on the deployment host. To connect from another machine, run an SSH tunnel on the client machine and use the local URL above. Replace `user@server` with the deployment host's SSH login:

```bash
ssh -N -L 18526:127.0.0.1:18526 user@server
```

## Benchmarks

The following results were measured for this project on both platforms. Fixed 2048 input / 4096 output tokens; three independent services, each warmed up once. Results use median TPOT and exclude time to first token.

| Platform | TPOT (ms/token) | Decode TPS (token/s) |
|---|---:|---:|
| x86_64 + Ascend950PR_957b | 2.909848 | 343.661 |
| ARM64 + Ascend950DT_9581 | 2.963747 | 337.411 |

See [benchmarks](docs/benchmarks.en.md) for test settings, per-round data, and commands.

## Licensing

See [licenses](licenses/) for third-party license texts.
