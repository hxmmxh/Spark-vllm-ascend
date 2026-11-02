<p align="center">
  <img src="docs/spark-vllm-ascend-logo.png" alt="Spark vLLM Ascend" width="720">
</p>

# Spark-vllm-ascend

**中文** | [English](README.en.md)

在 **Ascend 950** 上部署 **Spark 极速推理**，支持 x86_64 和 ARM64 主机，基于 vLLM / vLLM-Ascend 提供安装、服务启动和性能测试流程。

当前支持 **Spark-X2.6-4B Q4_0、单卡、batch=1、BF16 KV**，两种平台的 Decode 吞吐约为 **337–344 token/s**，测试配置和结果见下表。

## 部署

| 平台 | 指南 |
|---|---|
| x86_64 + Ascend950PR_957b | [x86 部署](docs/deployment/x86-950pr.md) |
| ARM64 + Ascend950DT_9581 | [ARM 部署](docs/deployment/arm64-950dt.md) |

```bash
git clone https://github.com/hxmmxh/Spark-vllm-ascend.git
cd Spark-vllm-ascend
```

按对应指南通过 uv 创建 Python 3.12 环境，选择空闲物理 NPU 和独立工作目录。两种平台共用源码，通过配置选择依赖与编译目标。模型、CANN 安装包和编译产物按需获取，不随仓库提供。

[部署要求与版本](docs/deployment/README.md) · [性能结果](docs/benchmarks.md)

## 大模型辅助部署

本项目提供 [Spark 部署 Skill](docs/skills/deploy-spark-vllm-ascend/SKILL.md)。将仓库链接或本地路径连同该 Skill 交给具备文件读取与目标机器终端执行能力的 AI 编程助手，即可让它按流程配置环境、编译、自检并启动服务。

## 日常使用

首次使用先按平台指南完成[公共部署步骤](docs/deployment/common.md)的第 1–4 步，随后即可启动服务；第 5 步性能测试为可选项。在部署终端保留 `REPO`、`WORK`、`DEVICE`、`PORT` 变量并执行：

```bash
export RUN_ID="serve-$(date '+%Y%m%d-%H%M%S')"
export RUN_OUT="$WORK/service/$RUN_ID"
mkdir -p "$RUN_OUT"
bash "$REPO/src/tools/deployment/run_user.sh" "$WORK" "$DEVICE" \
  python "$REPO/src/tools/validation/guard.py" --out "$WORK/guards/$RUN_ID.json" -- \
  python "$REPO/src/tools/deployment/serve.py" --work "$WORK" --out "$RUN_OUT" --port "$PORT" \
  > "$RUN_OUT/server.log" 2>&1
```

服务在前台运行，日志位于 `$RUN_OUT/server.log`，按 Ctrl-C 停止。以后重新启动时，在 Bash 中恢复实际的 `REPO`、`WORK`、`DEVICE`、`PORT` 后重复以上命令即可。

另开终端，将端口设为部署时使用的值。等待 `/health` 返回 HTTP 200 后发送对话请求：

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

回复正文位于 `choices[0].message.content`。示例关闭思考模式以便直接问答；多轮对话在 `messages` 中携带此前的 `user` / `assistant` 消息。需要流式输出时，在请求中加入 `"stream": true` 并给 curl 增加 `-N`。

支持自定义 OpenAI 兼容 Chat Completions 接口的客户端可填写：

| 配置 | 值 |
|---|---|
| API Base URL | `http://127.0.0.1:18526/v1`（端口与部署一致） |
| 模型名 | `spark` |
| API Key | 默认服务不要求；客户端必填时可填 `EMPTY` |

服务监听目标机器的 `127.0.0.1`。从其他机器使用时，可在客户端机器执行 SSH 转发，再连接上述本地地址（将 `user@server` 替换为部署机器的 SSH 登录地址）：

```bash
ssh -N -L 18526:127.0.0.1:18526 user@server
```

## 性能测试

以下为本项目在两种平台上的测试结果。固定 2048 输入 / 4096 输出 token，三个独立服务各预热一次，取三轮 TPOT 中位数，排除首 token 延迟。

| 平台 | TPOT（ms/token） | Decode TPS（token/s） |
|---|---:|---:|
| x86_64 + Ascend950PR_957b | 2.909848 | 343.661 |
| ARM64 + Ascend950DT_9581 | 2.963747 | 337.411 |

测试配置、逐轮数据和测试命令见[性能测试](docs/benchmarks.md)。

## 许可

第三方许可证文本见 [licenses](licenses/)。
