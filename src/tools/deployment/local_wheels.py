"""Verify locked environment wheels and install them before index resolution."""
import argparse
import hashlib
from pathlib import Path

from _project import selected


def prepare_requirements(work, lock):
    work = Path(work).resolve()
    requirements = []
    for key, spec in lock['downloads'].items():
        if spec['group'] != 'environment' or not spec['filename'].endswith('.whl') or key == 'pip_bootstrap':
            continue
        path = work / 'downloads' / spec['filename']
        if not path.is_file():
            raise FileNotFoundError(f'{path}: run fetch.py --work {work} --group environment first')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != spec['size'] or digest != spec['sha256']:
            raise ValueError(f'Locked wheel checksum mismatch: {path}')
        requirements.append(f'{path.as_uri()} --hash=sha256:{digest}\n')
    output = work / 'local-wheels.lock.txt'
    output.write_text(''.join(requirements))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    print(prepare_requirements(args.work, selected(check_host=True)[1]))
