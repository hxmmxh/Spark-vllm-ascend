"""Download only public, hash-locked inputs into a deployment work directory."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from _project import selected

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_large_model(work, key, spec):
    """Finite, checked Range transfers; publish only after the full SHA256 matches."""
    target=work/'downloads'/spec['filename'];part=target.with_name(target.name+'.partial')
    chunk_file=target.with_name(target.name+'.chunk')
    offset=part.stat().st_size if part.exists() else 0
    assert offset<=spec['size'],('Partial file exceeds locked size',key,offset)
    while offset<spec['size']:
        end=min(offset+(64<<20),spec['size'])-1
        expected=f"bytes {offset}-{end}/{spec['size']}"
        for attempt in range(3):
            try:
                url=spec['url']
                if url.startswith('https://hf-mirror.com/'):
                    url += ('&' if '?' in url else '?') + 'download=true&spark_range=' + str(offset)
                request=urllib.request.Request(url,headers={'User-Agent':'Spark-vllm-ascend/1','Range':f'bytes={offset}-{end}','Cache-Control':'no-cache'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    assert response.status==206 and response.headers.get('Content-Range')==expected,('Range not honored',response.status,response.headers.get('Content-Range'),expected)
                    with chunk_file.open('wb') as out:
                        while data:=response.read(8<<20):out.write(data)
                assert chunk_file.stat().st_size==end-offset+1,('Incomplete range',expected,chunk_file.stat().st_size)
                with part.open('ab') as out,chunk_file.open('rb') as source:
                    for data in iter(lambda:source.read(8<<20),b''):out.write(data)
                offset=end+1;chunk_file.unlink();print('VERIFIED_RANGE',key,offset,spec['size'],flush=True)
                break
            except Exception as exc:
                print('RANGE_FAILURE',key,expected,attempt+1,repr(exc),flush=True)
                if attempt==2:raise
                time.sleep(3)
    assert part.stat().st_size==spec['size'] and sha256(part)==spec['sha256'],('Full model checksum mismatch; retain file for investigation',key)
    part.replace(target);print('VERIFIED_DOWNLOAD',key,spec['sha256'],flush=True)
    return key,{'sha256':spec['sha256'],'bytes':spec['size'],'url':spec['url']}


def fetch_one(work, key, spec):
    target = work / 'downloads' / spec['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    if key in {'nnal', 'triton-ascend'}:
        from download_locked import fetch
        fetch(spec, target, workers=8, no_query=True)
        return key, {'sha256': sha256(target), 'bytes': spec['size'], 'url': spec['url']}
    if target.exists():
        if target.stat().st_size != spec['size'] or sha256(target) != spec['sha256']:
            raise RuntimeError(f'Existing file failed checksum: {target}')
        return key, {'sha256': spec['sha256'], 'bytes': spec['size'], 'url': spec['url']}
    if spec['group']=='model' and spec['size']>64<<20:
        return fetch_large_model(work,key,spec)
    part = target.with_name(target.name + '.partial')
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'Spark-vllm-ascend/1'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with urllib.request.urlopen(urllib.request.Request(spec['url'], headers=headers), timeout=90) as response:
                append = offset > 0 and response.status == 206
                if append:
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                with part.open('ab' if append else 'wb') as out:
                    while chunk := response.read(8 << 20):
                        out.write(chunk)
            if part.stat().st_size != spec['size'] or sha256(part) != spec['sha256']:
                raise RuntimeError(f'Download failed checksum: {key}')
            part.replace(target)
            print('VERIFIED_DOWNLOAD', key, spec['sha256'], flush=True)
            return key, {'sha256': spec['sha256'], 'bytes': spec['size'], 'url': spec['url']}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--group', choices=['all', 'environment', 'model'], default='all')
    parser.add_argument('--hf-mirror', action='store_true', help='Use hf-mirror.com transport for Hugging Face inputs; retain identical locked size/SHA256')
    parser.add_argument('--pypi-mirror', action='store_true', help='Use public Huawei PyPI mirror for the locked DSL wheel; preserve size/SHA256')
    args = parser.parse_args()
    lock = selected()[1]
    specs = [(k, v) for k, v in lock['downloads'].items() if args.group == 'all' or v['group'] == args.group]
    if args.hf_mirror:
        specs = [(key, dict(spec, url=spec['url'].replace('https://huggingface.co/', 'https://hf-mirror.com/'))) for key, spec in specs]
    if args.pypi_mirror:
        specs = [(key, dict(spec, url=spec['url'].replace('https://files.pythonhosted.org/packages/', 'https://repo.huaweicloud.com/repository/pypi/packages/'))) for key, spec in specs]
    args.work.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(lambda pair: fetch_one(args.work, *pair), specs))
    (args.work / f'downloads-{args.group}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
