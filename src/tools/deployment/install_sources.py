"""Build vLLM and vLLM-Ascend at the locked commits."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from _project import ROOT as root, selected
from build_recovery import archive_stale_kernel_locks, source_build_lock
from public_source import prepare, prepare_catlass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    profile, lock = selected(check_host=True)
    with source_build_lock(args.work):
        archive_stale_kernel_locks(args.work)
        for name in ['vllm', 'vllm-ascend']:
            spec = lock['sources'][name]
            dest = prepare(args.work, name, spec)
            if name == 'vllm-ascend':
                prepare_catlass(args.work, lock['sources']['catlass'], dest)
            env = os.environ.copy()
            if name == 'vllm':
                env['VLLM_TARGET_DEVICE'] = 'empty'
                env.pop('VLLM_VERSION_OVERRIDE', None)
                env['SETUPTOOLS_SCM_PRETEND_VERSION'] = '0.23.0'  # setup adds +empty to distribution metadata.
            else:
                env['SOC_VERSION'] = profile['soc_version']
                env['COMPILE_CUSTOM_KERNELS'] = '1'
                env['SETUPTOOLS_SCM_PRETEND_VERSION'] = '0.23.0'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--index-url', os.environ.get('SPARK_PYPI_INDEX','https://repo.huaweicloud.com/repository/pypi/simple'),
                            '--extra-index-url', 'https://mirrors.huaweicloud.com/ascend/repos/pypi',
                            '-c', str(root / 'deployment/resolved/requirements.txt'), '-e', str(dest)], env=env, check=True)


if __name__ == '__main__':
    main()
