# Benchmarks

[中文](benchmarks.md) | **English** · [Project home](../README.en.md)

2026-10-30 (Asia/Shanghai), ARM64 + Ascend950DT_9581; openEuler 24.03 SP4, GCC 12.3.1, Python 3.12.15, driver 25.6.rc1.

| Round | TPOT (ms/token) | Decode TPS (token/s) | TTFT (ms) |
|---|---:|---:|---:|
| 1 | 2.963747 | 337.411 | 69.386 |
| 2 | 2.946212 | 339.419 | 69.299 |
| 3 | 2.964386 | 337.338 | 69.324 |
| Median TPOT / corresponding TPS | 2.963747 | 337.411 | — |

One NPU, batch size 1, BF16 KV, fixed 2048 input / 4096 output tokens. Three independent services are each warmed up once. TPOT is the time between the first and last SSE events containing token IDs, divided by 4095; TPS = 1000 / TPOT(ms). All three guards passed and warmup outputs matched measured outputs.

[Benchmark input](../src/tools/benchmark/data/prompt.json) · [Request parameters](../src/tools/benchmark/data/request.json) · [Repeat the benchmark](deployment/arm64-950dt.en.md)

Benchmark commands save raw SSE records, service logs, and device checks in the work directory. These results apply to the hardware and request settings above. Head / slot4 checks and repeatable outputs do not establish full-model accuracy.
