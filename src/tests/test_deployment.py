"""CPU-only checks for profile isolation, package discovery and deployment inputs."""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tools/deployment'))
from _project import environment, read_profile, selected
from prepare_profile import prepare


class DeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='spark-deployment-test-')
        cls.copies = {}
        for name in ['x86-950pr', 'arm64-950dt']:
            dest = Path(cls.temporary.name) / name
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(name, dest)
            cls.copies[name] = dest

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_platform_sources_are_identical(self):
        for path in (ROOT / 'src').rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.suffix in {'.pyc', '.so', '.o'} or any(p.endswith('.egg-info') for p in path.parts):
                continue
            for dest in self.copies.values():
                self.assertEqual(path.read_bytes(), (dest / path.relative_to(ROOT)).read_bytes())

    def test_manifest_covers_current_sources(self):
        for dest in self.copies.values():
            for line in (dest / 'DEPLOYMENT_SHA256SUMS').read_text().splitlines():
                digest, name = line.split(maxsplit=1)
                self.assertEqual(hashlib.sha256((dest / name).read_bytes()).hexdigest(), digest, name)

    def test_installer_inputs_match_each_platform(self):
        for name, dest in self.copies.items():
            profile, lock = selected(dest)
            arch = profile['host_arch']
            exports = environment(profile, lock)
            for key in ['dsl', 'cann_toolkit', 'cann_ops', 'nnal']:
                self.assertIn(arch, lock['downloads'][key]['filename'])
            constraints = (dest / 'deployment/resolved/constraints-python.txt').read_text()
            self.assertIn(f"triton=={profile['base_triton']}\n", constraints)
            for spec in lock['cpu_wheels'].values():
                self.assertIn(arch, spec['links'][0])
                self.assertIn('#sha256=', spec['links'][0])
            self.assertEqual(bool(exports['SPARK_TRITON_WHEEL']), name == 'arm64-950dt')
            self.assertEqual(exports['SOC_VERSION'], profile['soc_version'])

    def test_wrong_host_is_rejected(self):
        with patch('_project.platform.machine', return_value='unsupported'):
            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                selected(self.copies['arm64-950dt'], check_host=True)

    def test_changed_profile_is_rejected(self):
        dest = self.copies['arm64-950dt']
        path = dest / 'deployment/profiles/arm64-950dt.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                selected(dest)
        finally:
            path.write_bytes(original)

    def test_existing_source_copy_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            prepare('arm64-950dt', self.copies['arm64-950dt'])

    def test_install_order_and_launcher_profile(self):
        # Simulate package-manager calls; execute no downloads, builds or NPU work.
        for name, dest in self.copies.items():
            with tempfile.TemporaryDirectory(prefix='spark-installer-test-') as directory:
                work = Path(directory)
                binary = work / '.venv/bin/python'
                binary.parent.mkdir(parents=True)
                binary.write_text(f'''#!{sys.executable}
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if args and args[0].endswith('_project.py'):
    # Other tests check host rejection; this test simulates both supported hosts.
    raise SystemExit(subprocess.call([{sys.executable!r}, *[a for a in args if a != '--check-host']]))
if args and args[0].endswith('runtime_environment.py'):
    raise SystemExit(subprocess.call([{sys.executable!r}, *args]))
with Path(os.environ['INSTALL_TEST_LOG']).open('a') as out:
    out.write(json.dumps({{'args': args, 'soc': os.environ.get('SOC_VERSION'),
                          'atb': os.environ.get('ATB_HOME_PATH'),
                          'work': os.environ.get('SPARK_WORK'),
                          'rpc': os.environ.get('VLLM_RPC_BASE_PATH'),
                          'no_proxy': os.environ.get('no_proxy'),
                          'jobs': os.environ.get('MAX_JOBS'),
                          'cmake_jobs': os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL'),
                          'virtual_env': os.environ.get('VIRTUAL_ENV')}}) + '\\n')
''')
                binary.chmod(0o755)
                competing_python = work / 'sdk/bin/python'
                competing_python.parent.mkdir(parents=True)
                competing_python.write_text('#!/bin/sh\nexit 91\n')
                competing_python.chmod(0o755)
                for relative in ['ascend-toolkit/set_env.sh', 'nnal/atb/9.1.0/atb/set_env.sh']:
                    script = work / 'cann' / relative
                    script.parent.mkdir(parents=True)
                    script.write_text('export SOC_VERSION=incorrect-inherited-target\n'
                                      f'export PATH="{work}/sdk/bin:$PATH"\n'
                                      'export VIRTUAL_ENV=/incorrect/sdk/environment\n'
                                      f'export ATB_HOME_PATH="{work}/cann/nnal/atb"\n')
                log = work / 'commands.jsonl'
                env = {**os.environ, 'INSTALL_TEST_LOG': str(log)}
                env.pop('VLLM_RPC_BASE_PATH', None)
                subprocess.run(['bash', str(dest / 'src/tools/deployment/install_python.sh'), str(work)],
                               env=env, check=True, capture_output=True)
                rows = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertTrue(all(r['soc'] == read_profile(name)[0]['soc_version'] for r in rows))
                self.assertTrue(all(r['atb'] for r in rows))
                self.assertTrue(all(r['virtual_env'] == str(work / '.venv') for r in rows))
                self.assertTrue(all(r['jobs'] == r['cmake_jobs'] == '64' for r in rows))
                local_install = next(i for i, r in enumerate(rows)
                                     if str(work / 'local-wheels.lock.txt') in r['args'])
                dependencies = next(i for i, r in enumerate(rows)
                                    if str(dest / 'deployment/resolved/requirements.txt') in r['args'])
                self.assertLess(local_install, dependencies)
                source_build = next(i for i, r in enumerate(rows) if r['args'][0].endswith('install_sources.py'))
                backend = [i for i, r in enumerate(rows) if any('triton_ascend-' in a for a in r['args'])]
                if name == 'arm64-950dt':
                    self.assertEqual(len(backend), 2)
                    self.assertLess(backend[0], source_build)
                    self.assertGreater(backend[1], source_build)
                    self.assertEqual(rows[backend[-1] + 1]['args'], ['-m', 'pip', 'check'])
                else:
                    self.assertEqual(backend, [])
                subprocess.run(['bash', str(dest / 'src/tools/deployment/run_user.sh'), str(work), '0',
                                'python', '-c', 'launcher smoke test'], env=env, check=True, capture_output=True)
                launched = json.loads(log.read_text().splitlines()[-1])
                self.assertEqual(launched['soc'], read_profile(name)[0]['soc_version'])
                self.assertEqual(launched['work'], str(work))
                self.assertEqual(launched['virtual_env'], str(work / '.venv'))
                self.assertIn('127.0.0.1', launched['no_proxy'].split(','))
                self.assertLessEqual(len(os.fsencode(launched['rpc'])) + 37, 107)
                Path(launched['rpc']).rmdir()

    def test_packaging_excludes_tools_and_tests(self):
        metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertEqual(metadata['tool']['setuptools']['packages']['find']['include'],
                         ['spark_vllm_ascend*'])
        entry = metadata['project']['entry-points']['vllm.general_plugins']['spark2_5']
        self.assertEqual(entry, 'spark_vllm_ascend.plugin:register')
        # Import must remain lazy: no torch, vLLM or NPU initialization.
        code = "import spark_vllm_ascend.plugin as p; import sys; assert callable(p.register); assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True,
                       env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})

    def test_all_executable_sources_are_under_src(self):
        for suffix in ['*.py', '*.sh', '*.cpp', '*.h', '*.asc']:
            for path in ROOT.glob(suffix):
                self.fail(f'Executable source at repository root: {path}')


if __name__ == '__main__':
    unittest.main()
