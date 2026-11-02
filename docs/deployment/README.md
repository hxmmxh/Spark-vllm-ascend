# 部署指南

**中文** | [English](README.en.md) · [项目首页](../../README.md)

选择 [ARM + 950DT](arm64-950dt.md) 或 [x86 + 950PR](x86-950pr.md)，设置平台变量，再执行[公共安装步骤](common.md)。

## 环境要求

Linux、可用的 Ascend 驱动与 DCMI v2、uv、GCC/G++、Git、Make、tar、curl；至少预留 40 GiB，下载与构建缓存可能需要更多。流程以普通用户运行，不升级驱动或固件。

[安装 uv](https://docs.astral.sh/uv/getting-started/installation/) 并确保 `uv` 在 `PATH` 中。部署脚本通过 `uv venv --managed-python --python 3.12` 创建独立环境，无需预装系统 Python 或使用特定镜像。Python 开发头文件会在环境创建后检查，GCC/G++ 等构建工具需由宿主机提供。

| 配置 | x86 + 950PR | ARM + 950DT |
|---|---|---|
| 主机架构 | x86_64 | aarch64 |
| SOC_VERSION | ascend950pr_957b | ascend950dt_9581 |
| CANN toolkit / 950 ops | 9.2.0~weekly.20260909.01 | 同左 |
| NNAL / ATB | 9.1.0 | 9.1.0 |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post4 | 同左 |
| cannbot-dsl / triton-ascend | 0.7.1 / 3.2.2 | 同左 |
| 基础 Triton 分发包 | 3.2.0 | 3.5.0 |
| Transformers / NumPy | 5.5.4 / 1.26.4 | 同左 |

两种平台都保持 28 mixed blocks / 56 vector lanes；950DT 的物理核数为 32 AIC / 64 AIV。框架提交、下载地址、哈希和 Python 版本约束以各平台[依赖锁](../../deployment/locks/)为准。

ARM 安装器会在依赖安装结束后恢复锁定的 Ascend Triton 后端，其导入版本显示 3.2.0；用运行时与 slot4 检查确认后端，不能只看版本字符串。x86 NNAL 目前只有官方 HTTPS 对象的 MD5 和大小，下载验证后记录 SHA256。

安装并通过运行时检查、设备自检后，即可按公共步骤第 6 节启动服务，参见[日常使用](../../README.md#日常使用)。性能测试按需执行，配置与结果见[性能测试](../benchmarks.md)。
