# Deployment guides

[中文](README.md) | **English** · [Project home](../../README.en.md)

Choose [ARM + 950DT](arm64-950dt.en.md) or [x86 + 950PR](x86-950pr.en.md), set the platform variables, then follow the [shared installation steps](common.en.md).

## Requirements

Linux, a working Ascend driver with DCMI v2, uv, GCC/G++, Git, Make, tar, and curl. Allow at least 40 GiB; downloads and build caches may require more. Installation runs as an ordinary user and does not upgrade drivers or firmware.

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and make sure it is on `PATH`. The deployment script uses `uv venv --managed-python --python 3.12` to create an isolated environment, without requiring system Python or a particular container image. Python development headers are checked after creation; build tools such as GCC/G++ must be available on the host.

| Setting | x86 + 950PR | ARM + 950DT |
|---|---|---|
| Host architecture | x86_64 | aarch64 |
| SOC_VERSION | ascend950pr_957b | ascend950dt_9581 |
| CANN toolkit / 950 ops | 9.2.0~weekly.20260909.01 | Same |
| NNAL / ATB | 9.1.0 | 9.1.0 |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post4 | Same |
| cannbot-dsl / triton-ascend | 0.7.1 / 3.2.2 | Same |
| Base Triton distribution | 3.2.0 | 3.5.0 |
| Transformers / NumPy | 5.5.4 / 1.26.4 | Same |

Both profiles retain 28 mixed blocks / 56 vector lanes; the 950DT has 32 physical AIC / 64 AIV cores. The platform [dependency locks](../../deployment/locks/) specify framework commits, download URLs, hashes, and Python package constraints.

The ARM installer restores the locked Ascend Triton backend after dependency installation; its imported version reports 3.2.0. Verify it with runtime and slot4 checks. For x86 NNAL, the lock currently records the official HTTPS object's MD5 and size; SHA256 is recorded after download verification.

After installation, runtime checks, and device self-checks, start the service with step 6 of the shared instructions. See [everyday use](../../README.en.md#everyday-use) for client connections. Benchmarks are optional; see [benchmarks](../benchmarks.en.md) for settings and results.
