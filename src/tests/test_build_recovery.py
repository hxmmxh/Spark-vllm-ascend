"""Recover dead compiler locks while preserving live or ambiguous owners."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tools/deployment'))
from build_recovery import archive_stale_kernel_locks, source_build_lock


class BuildRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='spark-build-test-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.build = self.work / 'src/vllm-ascend/csrc/build'

    def lock(self, name, pid):
        path = self.build / 'binary/ascend950/gen' / name / 'kernel_meta.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(pid))
        return path

    def test_archives_exited_owner_and_keeps_compiled_outputs(self):
        proc = subprocess.Popen([sys.executable, '-c', 'pass'])
        self.assertEqual(proc.wait(timeout=10), 0)
        lock = self.lock('kernel_meta_test', proc.pid)
        compiled = lock.with_name('completed.o')
        compiled.write_bytes(b'preserve completed compilation')
        unrelated = self.build / 'other/kernel_meta.lock'
        unrelated.parent.mkdir()
        unrelated.write_text(str(proc.pid))
        with source_build_lock(self.work), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(archive_stale_kernel_locks(self.work), 1)
            self.assertEqual(archive_stale_kernel_locks(self.work), 0)
        self.assertFalse(lock.exists())
        archived = list((self.work / 'build-recovery').rglob('kernel_meta.lock'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), str(proc.pid))
        self.assertEqual(compiled.read_bytes(), b'preserve completed compilation')
        self.assertTrue(unrelated.exists())

    def test_live_owner_is_not_moved_or_signalled(self):
        lock = self.lock('kernel_meta_active', os.getpid())
        with self.assertRaisesRegex(RuntimeError, 'still running'):
            archive_stale_kernel_locks(self.work)
        self.assertEqual(lock.read_text(), str(os.getpid()))
        self.assertFalse((self.work / 'build-recovery').exists())

    def test_unverifiable_owner_is_retained(self):
        lock = self.lock('kernel_meta_unknown', os.getpid())
        with patch('build_recovery.os.kill', side_effect=PermissionError), \
                self.assertRaisesRegex(RuntimeError, 'Cannot confirm'):
            archive_stale_kernel_locks(self.work)
        self.assertTrue(lock.exists())
        lock.write_text('incomplete PID')
        with self.assertRaisesRegex(RuntimeError, 'Cannot identify'):
            archive_stale_kernel_locks(self.work)
        self.assertTrue(lock.exists())

    def test_source_build_lock_rejects_overlap_and_releases_on_failure(self):
        with self.assertRaisesRegex(ValueError, 'simulated failure'):
            with source_build_lock(self.work):
                with self.assertRaisesRegex(RuntimeError, 'Another source build'):
                    with source_build_lock(self.work):
                        self.fail('Overlapping build acquired the lock')
                raise ValueError('simulated failure')
        with source_build_lock(self.work):
            pass


if __name__ == '__main__':
    unittest.main()
