"""Create an isolated source copy and resolve platform data; never patch code."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from _project import PROFILES, ROOT, read_profile, sha256


def prepare(profile, destination, root=ROOT):
    config, lock, profile_path, lock_path = read_profile(profile, root)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing existing source tree: {destination}')
    for folder in ['src', 'deployment', 'docs', 'licenses']:
        if destination.is_relative_to((root / folder).resolve()):
            raise ValueError('Deployment destination cannot be inside source inputs')
    destination.mkdir(parents=True)
    for folder in ['src', 'deployment', 'docs', 'licenses']:
        shutil.copytree(root / folder, destination / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.o',
                                                     '*.egg-info', 'build', 'dist', 'resolved'))
    for pattern in ['*.md', 'pyproject.toml', '.gitignore']:
        for path in root.glob(pattern):
            shutil.copy2(path, destination / path.name)
    resolved = destination / 'deployment/resolved'
    resolved.mkdir()
    (resolved / 'constraints-python.txt').write_text(''.join(
        f'{name}=={version}\n' for name, version in sorted(lock['python_constraints'].items())))
    cpu = ''.join(f'{name} @ {spec["links"][0]}\n' for name, spec in lock['cpu_wheels'].items())
    (resolved / 'cpu-wheels.lock.txt').write_text(cpu)
    (resolved / 'requirements.txt').write_text(
        (root / 'deployment/requirements.txt').read_text() + '\n' + cpu)
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    record = {'profile': profile, 'profile_sha256': sha256(profile_path),
              'lock_sha256': sha256(lock_path),
              'source_commit': commit.stdout.strip() if commit.returncode == 0 else None,
              'source_manifest': 'DEPLOYMENT_SHA256SUMS',
              'note': 'The manifest records actual copied files, including uncommitted edits.'}
    (destination / 'deployment-profile.json').write_text(json.dumps(record, indent=2) + '\n')
    manifest = [f'{sha256(path)}  {path.relative_to(destination)}'
                for path in sorted(destination.rglob('*')) if path.is_file()]
    (destination / 'DEPLOYMENT_SHA256SUMS').write_text('\n'.join(manifest) + '\n')
    print(json.dumps({'profile': config['name'], 'repository': str(destination),
                      'source_code_patched': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.profile, args.dest)
