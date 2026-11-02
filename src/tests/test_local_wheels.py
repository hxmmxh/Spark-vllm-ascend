"""Local wheel requirements retain locked hashes and reject corrupted cache files."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tools/deployment'))
from local_wheels import prepare_requirements


class LocalWheelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='spark wheel test ')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.wheel = self.work / 'downloads/compiler.whl'
        self.wheel.parent.mkdir()
        self.wheel.write_bytes(b'locked-wheel-fixture')
        self.spec = {'group': 'environment', 'filename': self.wheel.name,
                     'size': self.wheel.stat().st_size,
                     'sha256': hashlib.sha256(self.wheel.read_bytes()).hexdigest()}
        self.lock = {'downloads': {'dsl': self.spec}}

    def test_local_file_url_and_hash_are_used_instead_of_index_name(self):
        self.lock['downloads']['model'] = {'group': 'model', 'filename': 'not-downloaded.gguf'}
        self.lock['downloads']['pip_bootstrap'] = {'group': 'environment', 'filename': 'pip.whl'}
        requirements = prepare_requirements(self.work, self.lock).read_text()
        self.assertEqual(requirements, f'{self.wheel.as_uri()} --hash=sha256:{self.spec["sha256"]}\n')
        self.assertIn('%20', requirements)

    def test_corrupted_or_missing_wheel_stops_before_installation(self):
        self.wheel.write_bytes(b'changed-wheel-bytes')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            prepare_requirements(self.work, self.lock)
        self.assertFalse((self.work / 'local-wheels.lock.txt').exists())
        self.wheel.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'fetch.py'):
            prepare_requirements(self.work, self.lock)


if __name__ == '__main__':
    unittest.main()
