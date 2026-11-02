"""Check environment creation boundaries and uv's explicit installation target."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CREATE = ROOT / 'src/tools/deployment/create_env.sh'
sys.path.insert(0, str(CREATE.parent))
from prepare_profile import prepare


class EnvironmentSetupTest(unittest.TestCase):
    def test_existing_environment_is_never_cleared(self):
        with tempfile.TemporaryDirectory(prefix='spark-env-test-') as directory:
            work = Path(directory)
            marker = work / '.venv/keep.txt'
            marker.parent.mkdir()
            marker.write_text('existing environment')
            result = subprocess.run(['bash', str(CREATE), str(work)], capture_output=True, text=True,
                                    env={**os.environ, 'SPARK_PYTHON_VERSION': '3.12', 'UV_VENV_CLEAR': '1'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing an existing environment', result.stderr)
            self.assertEqual(marker.read_text(), 'existing environment')

    def test_incompatible_python_version_is_rejected_before_creation(self):
        with tempfile.TemporaryDirectory(prefix='spark-env-test-') as directory:
            result = subprocess.run(['bash', str(CREATE), directory], capture_output=True, text=True,
                                    env={**os.environ, 'SPARK_PYTHON_VERSION': '3.13'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cp312 wheels', result.stderr)
            self.assertFalse((Path(directory) / '.venv').exists())

    def test_uv_failure_does_not_fall_back_to_system_python(self):
        with tempfile.TemporaryDirectory(prefix='spark-env-test-') as directory:
            base = Path(directory)
            binary = base / 'bin/uv'
            binary.parent.mkdir()
            record = base / 'uv.json'
            binary.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
Path({str(record)!r}).write_text(json.dumps(sys.argv[1:]))
raise SystemExit(17)
''')
            binary.chmod(0o755)
            work = base / 'work'
            result = subprocess.run(['bash', str(CREATE), str(work)], capture_output=True, text=True,
                                    env={**os.environ, 'PATH': f'{binary.parent}:{os.environ["PATH"]}',
                                         'SPARK_PYTHON_VERSION': '3.12.15'})
            self.assertEqual(result.returncode, 17)
            args = json.loads(record.read_text())
            self.assertIn('--managed-python', args)
            self.assertEqual(args[args.index('--python') + 1], '3.12.15')
            self.assertEqual(args[-1], str(work / '.venv'))
            self.assertFalse((work / 'python-environment.json').exists())

    def test_bootstrap_seeds_the_prepared_environment_with_uv(self):
        # Only fake downloads/installers run here; no CANN installation or device access.
        with tempfile.TemporaryDirectory(prefix='spark-env-test-') as directory:
            base = Path(directory)
            repo, work = base / 'repo', base / 'work'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare('arm64-950dt', repo)
            binary = work / '.venv/bin/python'
            binary.parent.mkdir(parents=True)
            binary.write_text(f'''#!{sys.executable}
import subprocess, sys
args = sys.argv[1:]
if args and args[0].endswith('_project.py'):
    raise SystemExit(subprocess.call([{sys.executable!r}, *[a for a in args if a != '--check-host']]))
''')
            binary.chmod(0o755)
            uv = base / 'user-bin/uv'
            uv.parent.mkdir()
            record = base / 'uv.json'
            uv.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
Path({str(record)!r}).write_text(json.dumps({{'args': sys.argv[1:], 'venv': os.environ.get('VIRTUAL_ENV')}}))
''')
            uv.chmod(0o755)
            stages = base / 'stages.txt'
            for name in ['install_cann.sh', 'install_python.sh']:
                (repo / 'src/tools/deployment' / name).write_text(
                    f'#!/bin/bash\nprintf "%s\\n" "{name}:$VIRTUAL_ENV:$(command -v python)" >> "{stages}"\n')
            env = {**os.environ, 'PATH': f'{uv.parent}:{os.environ["PATH"]}',
                   'VIRTUAL_ENV': '/unrelated/environment'}
            result = subprocess.run(['bash', str(repo / 'src/tools/deployment/bootstrap_user.sh'), str(work)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(record.read_text())
            self.assertEqual(row['args'][:2], ['pip', 'install'])
            self.assertEqual(row['args'][row['args'].index('--python') + 1], str(binary))
            self.assertIn('--no-index', row['args'])
            self.assertEqual(row['venv'], str(work / '.venv'))
            self.assertEqual(stages.read_text().splitlines(),
                             [f'{name}:{work / ".venv"}:{binary}'
                              for name in ['install_cann.sh', 'install_python.sh']])


if __name__ == '__main__':
    unittest.main()
