"""Short local IPC paths and loopback proxy exclusions for deployment commands."""
import argparse
import hashlib
import os
import shlex
import stat
from pathlib import Path


def runtime_environment(work, environ=None):
    environ = os.environ if environ is None else environ
    identity = hashlib.sha256(os.fsencode(Path(work).resolve())).hexdigest()[:16]
    rpc = Path(environ.get('VLLM_RPC_BASE_PATH') or f'/tmp/spark-ipc-{os.getuid()}-{identity}')
    # The pinned vLLM appends '/' and a 36-character UUID to this path.
    if not rpc.is_absolute() or len(os.fsencode(rpc / ('0' * 36))) > 107:
        raise ValueError('VLLM_RPC_BASE_PATH must be absolute and fit a UUID socket path within 107 bytes')
    rpc.mkdir(mode=0o700, exist_ok=True)
    info = rpc.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError(f'IPC directory must be owned by this user and not writable by others: {rpc}')
    exclusions = ['127.0.0.1', 'localhost', '::1']
    for key in ['no_proxy', 'NO_PROXY']:
        exclusions.extend(value.strip() for value in environ.get(key, '').split(',') if value.strip())
    bypass = ','.join(dict.fromkeys(exclusions))
    return {'VLLM_RPC_BASE_PATH': str(rpc), 'NO_PROXY': bypass, 'no_proxy': bypass}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    for key, value in runtime_environment(args.work).items():
        print(f'export {key}={shlex.quote(value)}')
