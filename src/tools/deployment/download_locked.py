"""Parallel HTTPS ranges; accept artifacts only after pinned size and SHA256 pass."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path


class NetworkDownloadError(RuntimeError):
    """Incomplete ranges can be retried without accepting invalid artifacts."""


def fetch(spec, dest, workers, resolves=(), no_query=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # This CANN CDN serves arbitrary Range requests on its canonical URL, while
    # uncached query variants can return persistent gateway errors. A probe at
    # an otherwise failing offset returned the exact requested canonical range.
    if spec['url'].startswith('https://ascend.devcloud.huaweicloud.com/'):
        no_query = True
    expected_hash = spec['sha256']
    record_hash = expected_hash is None and spec.get('hash_policy') == 'record_after_download'
    assert expected_hash or record_hash, 'Missing pinned hash without an explicit recording policy'
    if dest.exists():
        assert dest.stat().st_size == spec['size']
        expected = expected_hash or dest.with_suffix(dest.suffix + '.sha256').read_text().strip()
        assert hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() == expected
        if spec.get('md5'):
            assert hashlib.file_digest(dest.open('rb'), 'md5').hexdigest() == spec['md5']
        print('ALREADY_VERIFIED', dest, flush=True)
        return
    parts = dest.parent / (dest.name + '.ranges')
    parts.mkdir(exist_ok=True)
    size = 4 << 20

    def chunk(start):
        end = min(start + size, spec['size']) - 1
        path = parts / str(start)
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        temp = path.with_suffix('.resume')
        for attempt in range(12):
            offset = temp.stat().st_size if temp.exists() else 0
            if offset == end-start+1:
                temp.replace(path)
                return path
            tail = path.with_suffix('.tail')
            headers = path.with_suffix('.headers')
            # curl may receive headers and then time out before opening its
            # output. Never append a previous attempt's body in that case.
            tail.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
            # Some mirror caches key on URL without varying on Range. Include
            # the actual resumed interval, and avoid replaying cached 504s.
            url = spec['url'] if no_query else spec['url'] + ('&' if '?' in spec['url'] else '?') + f'spark_range={start+offset}&spark_end={end}&spark_attempt={attempt}'
            command = ['curl', '--fail', '--location', '--silent', '--show-error',
                       '--connect-timeout', '10', '--max-time', '90',
                       '--header', 'Cache-Control: no-cache',
                       '--range', f'{start+offset}-{end}', '--dump-header', str(headers),
                       '--output', str(tail), url]
            for resolve in resolves:
                command.extend(['--resolve', resolve])
            for name, value in spec.get('headers', {}).items():
                command.extend(['--header', f'{name}: {value}'])
            result = subprocess.run(command, capture_output=True, text=True)
            expected = f'content-range: bytes {start+offset}-{end}/{spec["size"]}'
            response_headers = headers.read_text().lower() if headers.exists() else ''
            range_ok = expected in response_headers or (start+offset == 0 and end == spec['size']-1)
            if result.returncode in (0, 18, 28) and tail.exists() and range_ok and 0 < tail.stat().st_size <= end-start+1-offset:
                with temp.open('ab') as stream:
                    stream.write(tail.read_bytes())
            if temp.exists() and temp.stat().st_size == end-start+1:
                temp.replace(path)
                return path
            if attempt == 11:
                raise RuntimeError((dest.name, start, result.returncode, result.stderr))
            time.sleep(2)

    starts = list(range(0, spec['size'], size))
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(chunk, start) for start in starts]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                errors.append(repr(error))
                print('RANGE_ERROR', errors[-1], flush=True)
            if i % 20 == 0 or i == len(starts):
                print('RANGES', dest.name, i, '/', len(starts), flush=True)
    if errors:
        raise NetworkDownloadError(errors)
    output = dest.with_suffix(dest.suffix + '.assembled')
    digest = hashlib.sha256()
    md5 = hashlib.md5() if spec.get('md5') else None
    with output.open('wb') as stream:
        for start in starts:
            data = (parts / str(start)).read_bytes()
            digest.update(data)
            if md5 is not None:
                md5.update(data)
            stream.write(data)
    assert output.stat().st_size == spec['size']
    if md5 is not None:
        assert md5.hexdigest() == spec['md5'], ('Publisher MD5 mismatch', dest.name, md5.hexdigest())
    if dest.suffix == '.whl':
        with zipfile.ZipFile(output) as archive:
            assert archive.testzip() is None, ('Wheel CRC validation failed', dest.name)
    if expected_hash:
        assert digest.hexdigest() == expected_hash, (dest.name, digest.hexdigest())
    else:
        output.with_suffix(dest.suffix + '.sha256').write_text(digest.hexdigest() + '\n')
        output.with_suffix(dest.suffix + '.sha256').replace(dest.with_suffix(dest.suffix + '.sha256'))
    output.replace(dest)
    print('HASH_RECORDED' if record_hash else 'VERIFIED_DOWNLOAD', dest.name, digest.hexdigest(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--resolve', action='append', default=[])
    parser.add_argument('--no-query', action='store_true')
    parser.add_argument('keys', nargs='+')
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    for key in args.keys:
        spec = lock['downloads'][key]
        fetch(spec, args.work / 'downloads' / spec['filename'], args.workers, args.resolve, args.no_query)
