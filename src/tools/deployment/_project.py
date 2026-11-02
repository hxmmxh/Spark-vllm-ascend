"""Read the selected deployment profile without importing runtime dependencies."""
import argparse
import hashlib
import json
import platform
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'src/spark_vllm_ascend'
PROFILES = ('x86-950pr', 'arm64-950dt')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_profile(name, root=ROOT):
    if name not in PROFILES:
        raise ValueError(f'Unsupported profile: {name}')
    path = root / 'deployment/profiles' / f'{name}.json'
    profile = json.loads(path.read_text())
    expected = {
        'x86-950pr': ('x86_64', 'ascend950pr_957b', 'linux/amd64'),
        'arm64-950dt': ('aarch64', 'ascend950dt_9581', 'linux/arm64'),
    }[name]
    if (profile['host_arch'], profile['soc_version']) != expected[:2]:
        raise ValueError('Unsupported host architecture / SOC combination')
    if profile['schema'] != 1 or profile['name'] != name:
        raise ValueError('Invalid deployment profile')
    lock_path = (path.parent / profile['dependency_lock']).resolve()
    if not lock_path.is_relative_to((root / 'deployment/locks').resolve()):
        raise ValueError('Dependency lock must be inside deployment/locks')
    lock = json.loads(lock_path.read_text())
    if lock['schema'] != 2 or lock['platform'] != expected[2]:
        raise ValueError('Dependency lock platform does not match the profile')
    if lock['python_constraints']['triton'] != profile['base_triton']:
        raise ValueError('Triton constraint does not match the profile')
    if profile['kernel_launch'] != {'mixed_blocks': 28, 'vector_lanes': 56}:
        raise ValueError('Only the validated 28 / 56 kernel launch is supported')
    if profile['restore_triton_backend'] and 'triton-ascend' not in lock['downloads']:
        raise ValueError('The Ascend backend workaround needs a locked wheel')
    return profile, lock, path, lock_path


def selected(root=ROOT, check_host=False):
    path = root / 'deployment-profile.json'
    if not path.is_file():
        raise RuntimeError('Prepare a deployment copy with prepare_profile.py first')
    record = json.loads(path.read_text())
    profile, lock, profile_path, lock_path = read_profile(record['profile'], root)
    if (sha256(profile_path) != record['profile_sha256']
            or sha256(lock_path) != record['lock_sha256']):
        raise RuntimeError('Deployment configuration changed; prepare a fresh copy')
    if check_host and platform.machine() != profile['host_arch']:
        raise RuntimeError(f"Expected {profile['host_arch']}, found {platform.machine()}")
    return profile, lock


def environment(profile, lock):
    downloads = lock['downloads']
    return {
        'SPARK_PROFILE_NAME': profile['name'],
        'SPARK_HOST_ARCH': profile['host_arch'],
        'SOC_VERSION': profile['soc_version'],
        'CANNBOTDSL_NPU_ARCH': profile['npu_arch'],
        'SPARK_NNAL_ENV': profile['nnal_env'],
        'SPARK_DSL_WHEEL': downloads['dsl']['filename'],
        'SPARK_CANN_TOOLKIT': downloads['cann_toolkit']['filename'],
        'SPARK_CANN_OPS': downloads['cann_ops']['filename'],
        'SPARK_NNAL_INSTALLER': downloads['nnal']['filename'],
        'SPARK_TRITON_WHEEL': downloads['triton-ascend']['filename']
            if profile['restore_triton_backend'] else '',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shell', action='store_true')
    parser.add_argument('--check-host', action='store_true')
    args = parser.parse_args()
    profile, lock = selected(check_host=args.check_host)
    if args.shell:
        for key, value in environment(profile, lock).items():
            print(f'export {key}={shlex.quote(value)}')
    else:
        print(json.dumps(profile, indent=2))
