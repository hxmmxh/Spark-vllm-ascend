# 性能测试

**中文** | [English](benchmarks.en.md) · [项目首页](../README.md)

2026-10-30（Asia/Shanghai），ARM64 + Ascend950DT_9581；openEuler 24.03 SP4、GCC 12.3.1、Python 3.12.15、驱动 25.6.rc1。

| 轮次 | TPOT（ms/token） | Decode TPS（token/s） | TTFT（ms） |
|---|---:|---:|---:|
| 1 | 2.963747 | 337.411 | 69.386 |
| 2 | 2.946212 | 339.419 | 69.299 |
| 3 | 2.964386 | 337.338 | 69.324 |
| TPOT 中位数 / 对应 TPS | 2.963747 | 337.411 | — |

单卡、batch=1、BF16 KV，固定 2048 输入 / 4096 输出 token。三个独立服务各预热一次，TPOT 为首末含 token ID 的 SSE 事件时间差除以 4095，TPS = 1000 / TPOT(ms)。三轮 guard 通过，预热和测量输出一致。

[测试输入](../src/tools/benchmark/data/prompt.json) · [请求参数](../src/tools/benchmark/data/request.json) · [复测流程](deployment/arm64-950dt.md)

测试命令会在工作目录保存 SSE 原始记录、服务日志和设备检查结果。性能数据适用于上述硬件与请求配置；head / slot4 自检和输出一致性不代表完整模型精度验收。
