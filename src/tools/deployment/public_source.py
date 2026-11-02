"""Fixed-commit official source archives; verify every original file, plus one named patch."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

BEFORE = 'opencv-python-headless >= 4.13.0    # required for video IO'
AFTER = 'opencv-python-headless == 4.11.0.86 # Spark NumPy 1.26-compatible environment'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(work, name, spec):
    """Use only the locked public archive. Existing source requires its own manifest."""
    dest = work / 'src' / name
    record = work / 'source-manifests' / (name + '.json')
    archive_spec = spec['archive']
    archive = work / 'public-archives' / (name + '.tar.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        part = archive.with_suffix('.partial')
        with urllib.request.urlopen(archive_spec['url'], timeout=90) as response, part.open('wb') as out:
            while chunk := response.read(4 << 20):
                out.write(chunk)
        assert part.stat().st_size == archive_spec['size'] and digest(part) == archive_spec['sha256']
        part.replace(archive)
    assert archive.stat().st_size == archive_spec['size'] and digest(archive) == archive_spec['sha256']
    if dest.exists():
        assert record.exists(), f'Existing dependency source lacks its verification manifest: {dest}'
        verify(work, name, spec)
        return dest
    parent = work / 'src'
    parent.mkdir(parents=True, exist_ok=True)
    prefix = name + '-' + spec['commit']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(Path(m.name).parts[0] == prefix and '..' not in Path(m.name).parts for m in members)
        tar.extractall(parent, filter='data')
    (parent / prefix).rename(dest)
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
    patched = {}
    if name == 'vllm':
        path = dest / 'requirements/common.txt'
        original = path.read_text()
        assert original.count(BEFORE) == 1
        path.write_text(original.replace(BEFORE, AFTER))
        patched['requirements/common.txt'] = {'original_sha256': files['requirements/common.txt'], 'patched_sha256': digest(path), 'before': BEFORE, 'after': AFTER}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({'commit': spec['commit'], 'archive': archive_spec, 'files': files, 'patches': patched}, indent=2))
    verify(work, name, spec)
    return dest


def verify(work, name, spec):
    dest = work / 'src' / name
    manifest = json.loads((work / 'source-manifests' / (name + '.json')).read_text())
    assert manifest['commit'] == spec['commit'] and manifest['archive'] == spec['archive']
    for name, expected in manifest['files'].items():
        if name in manifest['patches']:
            patch = manifest['patches'][name]
            assert name == 'requirements/common.txt' and patch['original_sha256'] == expected
            assert patch['before'] == BEFORE and patch['after'] == AFTER
            contents = (dest / name).read_text()
            assert contents.count(AFTER) == 1
            assert hashlib.sha256(contents.replace(AFTER, BEFORE).encode()).hexdigest() == expected
            expected = patch['patched_sha256']
        assert digest(dest / name) == expected, ('Source changed', str(dest / name))
    return {key: manifest[key] for key in ['commit', 'archive', 'patches']}


def prepare_catlass(work, spec, ascend):
    import subprocess
    dest = work / 'src/catlass'
    if not dest.exists():
        subprocess.run(['git', 'init', str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'remote', 'add', 'origin', spec['url']], check=True)
        subprocess.run(['git', '-c', 'http.version=HTTP/1.1', '-C', str(dest), 'fetch', '--depth', '1', 'origin', spec['commit']], check=True, timeout=300)
        subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    verify_catlass(work, spec)
    link = ascend / 'csrc/third_party/catlass'
    if link.is_dir() and not link.is_symlink():
        link.rmdir()  # Only the empty submodule placeholder in the public archive.
    if not link.exists():
        link.symlink_to('../../../catlass', target_is_directory=True)
    assert link.resolve() == dest.resolve()


def verify_catlass(work, spec):
    import subprocess
    dest = work / 'src/catlass'
    assert subprocess.check_output(['git', '-C', str(dest), 'config', '--get', 'remote.origin.url'], text=True).strip() == spec['url']
    assert subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip() == spec['commit']
    assert not subprocess.check_output(['git', '-C', str(dest), 'diff', '--name-only'], text=True)
    assert not (dest / '.git/objects/info/alternates').exists()
    assert (dest / 'include').is_dir()
    return spec
