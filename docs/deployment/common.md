# 公共部署步骤

**中文** | [English](common.en.md) · [平台选择](README.md)

日常使用：完成第 1–4 步后直接执行第 6 步启动服务。第 5 步为可选性能测试；客户端连接方法见[日常使用](../../README.md#日常使用)。

## 1. 创建 uv 环境与部署副本

先在对应平台指南中设置变量。uv 按版本获取独立的 CPython 3.12，并创建 `$WORK/.venv`；脚本会检查开发头文件。随后使用该环境生成部署副本。可将 `SPARK_PYTHON_VERSION` 设为具体的 `3.12.x` 版本。

```bash
bash "$BASE/src/tools/deployment/create_env.sh" "$WORK"
"$WORK/.venv/bin/python" "$BASE/src/tools/deployment/prepare_profile.py" --profile "$PROFILE" --dest "$REPO"
(cd "$REPO" && sha256sum -c DEPLOYMENT_SHA256SUMS)
export DEPLOY="$REPO/src/tools/deployment"
export VALIDATE="$REPO/src/tools/validation"
export NATIVE="$REPO/src/spark_vllm_ascend/native"
run() { bash "$DEPLOY/run_user.sh" "$WORK" "$DEVICE" "$@"; }
```

`run_user.sh` 自动为工作目录分配 `/tmp/spark-ipc-<uid>-<hash>` 通信目录，避免深层工作目录使 Unix socket 路径超过 107 字节。可用 `VLLM_RPC_BASE_PATH` 指定当前用户拥有的短绝对路径，启动时会检查长度和目录权限。启动器也会合并已有的 `NO_PROXY` / `no_proxy`，确保本机请求直连。

## 2. 安装环境与框架

在上一步的 uv 环境中下载并校验 CANN、NNAL、DSL 和平台所需 Triton，安装锁定依赖，从源码编译 vLLM / vLLM-Ascend。uv 将校验过的 pip 安装到该环境，后续依赖安装保留 Ascend/Triton 所需的顺序。原生算子编译、服务启动和测试也使用同一环境。

首次安装和单独恢复 Python 构建均使用 64 路编译并行度：`CMAKE_BUILD_PARALLEL_LEVEL=64`、`MAX_JOBS=64`。

安装器会重新校验并先安装已下载的 DSL / ARM Triton wheel，再解析其他依赖，避免 pip 从索引重复下载相同包。其他依赖仍按锁文件安装；ARM Triton 后端在依赖和源码安装后恢复。

```bash
DEBUG=0 PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=5 \
  bash "$DEPLOY/bootstrap_user.sh" "$WORK" > "$WORK/logs/install.log" 2>&1
run python -m pip check
run python "$DEPLOY/check_runtime.py" --work "$WORK" --profile "$PROFILE" \
  > "$WORK/logs/check-runtime.log" 2>&1
cat "$WORK/logs/check-runtime.log"
```

要求出现 `SPARK_USER_ENVIRONMENT_BUILT`、`pip check` 无冲突和 `RUNTIME_IMPORTS_PASS`。

## 3. 准备模型

成功标志为 `MODEL_PREPARED 290`、`HEAD_CPU_EXACT`。已有 `model/` 或 `head/` 时不要重复准备。下载 Hugging Face 文件受阻时可给 fetch.py 加 `--hf-mirror`，哈希保持不变。

```bash
run python "$DEPLOY/fetch.py" --work "$WORK" --group model > "$WORK/logs/download-model.log" 2>&1
run python "$DEPLOY/prepare_model.py" --work "$WORK" > "$WORK/logs/prepare-model.log" 2>&1
run python "$DEPLOY/prepare_head.py" --work "$WORK" > "$WORK/logs/prepare-head.log" 2>&1
```

## 4. 编译与设备自检

guard 等待目标卡空闲，并在出现外部进程或遥测异常时判定无效。成功标志为 `HEAD_STRICT_PASS`、`SLOT4_STRICT_PASS`。

```bash
run python "$NATIVE/head/build.py" > "$WORK/logs/build-head.log" 2>&1
run python "$NATIVE/head_reference/build.py" > "$WORK/logs/build-head-reference.log" 2>&1
run python "$VALIDATE/guard.py" --out "$WORK/guards/head-01.json" -- \
  python "$VALIDATE/test_head.py" --work "$WORK" > "$WORK/logs/head-01.log" 2>&1
run python "$VALIDATE/guard.py" --out "$WORK/guards/slot4-01.json" -- \
  python "$VALIDATE/test_slot_mapping.py" > "$WORK/logs/slot4-01.log" 2>&1
```

## 5. 三轮性能测试（可选）

测试器依次启动三个独立服务，每轮完整预热一次。此时不要运行下一节的常驻服务。

测速客户端对本机的健康检查、模型查询和流式请求均直连，不使用继承的 HTTP 代理。

```bash
export BENCH_STAMP="$(date '+%Y%m%d-%H%M%S')"
run python "$REPO/src/tools/benchmark/benchmark.py" \
  --work "$WORK" --repo "$REPO" --port "$PORT" --rounds 3 \
  > "$WORK/logs/benchmark-$BENCH_STAMP.log" 2>&1
"$WORK/.venv/bin/python" "$DEPLOY/verify_results.py" --work "$WORK" --repo "$REPO" --device "$DEVICE"
cat "$WORK/benchmark-results.csv"
```

验收要求 `deployment-benchmark-verification.json` 中 `complete=true`。保留 `measurements/`、`service/`、`guards/`；重复测速前另存旧汇总。当前不包含完整模型参考精度验收。

## 6. 单独启动服务

完成第 1–4 步后即可执行本节。服务在前台运行，日志位于 `$RUN_OUT/server.log`，按 Ctrl-C 停止；以后恢复部署变量与 `run` 函数即可重新启动。正式测速时请先停止该服务。

```bash
export RUN_ID="serve-$(date '+%Y%m%d-%H%M%S')"
export RUN_OUT="$WORK/service/$RUN_ID"
mkdir -p "$RUN_OUT"
run python "$VALIDATE/guard.py" --out "$WORK/guards/$RUN_ID.json" -- \
  python "$DEPLOY/serve.py" --work "$WORK" --out "$RUN_OUT" --port "$PORT" \
  > "$RUN_OUT/server.log" 2>&1
```

另开终端，设置相同端口，等待 `/health` 返回 HTTP 200 后发送请求：

```bash
export PORT=18526
curl --noproxy 127.0.0.1 -fsS "http://127.0.0.1:$PORT/health"
curl --noproxy 127.0.0.1 -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark","messages":[{"role":"user","content":"Explain matrix multiplication briefly."}],"temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 失败恢复与复测

- Python 环境创建失败：检查 uv 的下载日志，使用新的工作目录重跑第 1 步；脚本不会覆盖已有 `.venv`。解释器版本与路径记录在 `$WORK/python-environment.json`。
- 下载失败：重跑 `fetch.py --work "$WORK" --group environment`；校验失败时保留文件并检查日志。
- Python 构建失败：确认 CANN/NNAL 安装完整后，运行 `DEBUG=0 bash "$DEPLOY/install_python.sh" "$WORK"`，随后重做运行时检查。该脚本也会恢复 ARM Triton 后端。
- 编译中断后报 `Another process is using this dir`：同样重跑 `install_python.sh`。源码构建会串行保护同一工作目录，并将确认已退出进程留下的 `kernel_meta.lock` 归档到 `$WORK/build-recovery/`。若所属进程仍存在或无法确认状态，会保留锁并报错；先等待该构建结束再恢复。
- 缺少 `libatb.so`：检查 `install-nnal.log` 和配置指定的 NNAL 环境脚本；不能仅根据目录存在认定安装成功。
- 设备自检失败：检查日志，重试时使用新的 guard 文件名；重编 head 后必须重新验证。正式测速失败后整批重测。
- 已有环境复测：使用原部署副本和工作目录，恢复上面的变量与 `run` 函数后执行第 5 步，不重建环境或覆盖已准备权重。
