"""Serialize source builds and archive CANN locks left by exited compilers."""
import contextlib
import fcntl
import os
import time
from pathlib import Path


@contextlib.contextmanager
def source_build_lock(work):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'framework-build.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another source build is using {work}') from exc
        yield


def archive_stale_kernel_locks(work):
    work = Path(work).resolve()
    build = work / 'src/vllm-ascend/csrc/build'
    candidates = []
    for path in sorted((build / 'binary').rglob('kernel_meta.lock')):
        if path.is_symlink() or not path.resolve().is_relative_to(build):
            raise RuntimeError(f'Unexpected compiler lock path: {path}')
        text = path.read_text().strip()
        if not text.isdecimal() or not 0 < int(text) <= 2**31 - 1:
            raise RuntimeError(f'Cannot identify compiler lock owner: {path}')
        pid = int(text)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            candidates.append(path)
        except PermissionError as exc:
            raise RuntimeError(f'Cannot confirm compiler PID {pid} has exited; retaining {path}') from exc
        else:
            raise RuntimeError(f'Compiler PID {pid} is still running; retaining {path}')
    # Check every owner before moving anything. Never signal a compiler or remove an active lock.
    if candidates:
        archive = work / 'build-recovery' / str(time.time_ns())
        for path in candidates:
            target = archive / path.relative_to(build)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            print('ARCHIVED_STALE_COMPILER_LOCK', str(path), str(target), flush=True)
    return len(candidates)
