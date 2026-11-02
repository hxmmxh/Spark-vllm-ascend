"""Three independent guarded Spark services, with the README timing protocol."""
import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def local_client():
    # Every benchmark request targets loopback; inherited proxies must not route it away.
    client = requests.Session()
    client.trust_env = False
    return client


def session(args):
    work, repo = args.work.resolve(), args.repo.resolve()
    out = work / 'measurements' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    service_out = work / 'service' / args.run_id
    service_out.mkdir(parents=True, exist_ok=False)
    guard_path = work / 'guards' / (args.run_id + '.json')
    url = f'http://127.0.0.1:{args.port}'

    # Refuse an occupied port before starting a service or sending any request.
    with socket.socket() as probe:
        # Match the server's address reuse policy: a prior service's TIME_WAIT
        # connections must not block the next independent startup.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))

    def read_guard():
        # The repository guard rewrites its JSON file in place on every sample.
        for attempt in range(50):
            try:
                return json.loads(guard_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                if attempt == 49:
                    raise
                time.sleep(.02)

    def check_guard():
        state = read_guard()
        assert state['phase'] == 'running' and not state.get('error'), state
        assert state['pid'] == os.getpid(), 'This controller must be the guard-owned child'
        return state

    # Guard writes its running state immediately after spawning this controller.
    for _ in range(50):
        if read_guard()['phase'] == 'running':
            break
        time.sleep(.1)
    check_guard()
    with local_client() as client, (service_out / 'server.log').open('w') as server_log:
        process = subprocess.Popen(
            [sys.executable, str(repo / 'src/tools/deployment/serve.py'), '--work', str(work),
             '--out', str(service_out), '--port', str(args.port)],
            cwd=repo, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                assert process.poll() is None, f'Service exited {process.returncode}: {service_out}/server.log'
                check_guard()
                try:
                    if client.get(url + '/health', timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Service startup exceeded 1800 seconds')

            prompt = json.loads((repo / 'src/tools/benchmark/data/prompt.json').read_text())
            assert len(prompt) == 2048
            body = json.loads((repo / 'src/tools/benchmark/data/request.json').read_text())
            body['prompt'] = prompt
            save(out / 'request.json', body)
            models = client.get(url + '/v1/models', timeout=10)
            models.raise_for_status()
            save(out / 'models.json', models.json())

            def measure(name):
                ids, events, usage = [], [], None
                started_epoch = time.time()
                start = time.perf_counter()
                with client.post(url + '/v1/completions', json=body,
                                   stream=True, timeout=900) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=8192):
                        now = time.perf_counter()
                        if not line.startswith(b'data: ') or line[6:] == b'[DONE]':
                            continue
                        obj = json.loads(line[6:])
                        assert 'error' not in obj, obj
                        usage = obj.get('usage') or usage
                        for choice in obj.get('choices', []):
                            if choice.get('token_ids'):
                                ids.extend(choice['token_ids'])
                                events.append(dict(time=now, token_ids=choice['token_ids']))
                end = time.perf_counter()
                save(out / (name + '-raw.json'), dict(start=start, end=end,
                     started_epoch=started_epoch, finished_epoch=time.time(),
                     usage=usage, events=events, token_ids=ids))
                assert usage and usage['prompt_tokens'] == 2048, usage
                assert usage['completion_tokens'] == len(ids) == 4096, (usage, len(ids))
                assert len(events) > 1
                assert all(a['time'] <= b['time'] for a, b in zip(events, events[1:]))
                tpot = (events[-1]['time'] - events[0]['time']) * 1000 / 4095
                result = dict(tpot_ms=tpot, tps=1000/tpot,
                              ttft_ms=(events[0]['time']-start)*1000,
                              total_seconds=end-start, output_tokens=len(ids),
                              prompt_tokens=2048, sse_token_events=len(events),
                              output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                save(out / (name + '.json'), result)
                check_guard()
                print(name, json.dumps(result), flush=True)
                return ids, result

            warm, _ = measure('warmup')
            ids, result = measure('measurement')
            assert ids == warm, 'Warmup and measurement token IDs differ'
            save(out / 'guard-after-measurement.json', check_guard())
            log_text = (service_out / 'server.log').read_text()
            required = ['V43_KERNEL_VARIANT cannbot_dsl 0.7.1',
                        'V43_WEIGHTS_PREPACKED 36', 'Q6_HEAD_INSTALLED',
                        'SPARK_SLOT4_ACTIVE', 'V43_PREFILL_NZ_REFRESH',
                        'Graph capturing finished']
            markers = {marker: marker in log_text for marker in required}
            markers['36_layer_executions'] = log_text.count('V43_LAYER_EXECUTE') == 36
            save(out / 'path-validation.json', markers)
            assert all(markers.values()), markers
            save(out / 'result.json', dict(result, deterministic_warmup=True,
                                          runtime_path_valid=True, run_id=args.run_id))
        finally:
            # The repository guard owns this controller's entire process group
            # and terminates it after observing the controller's exit status.
            # Stopping EngineCore here creates a DCMI/PID teardown race while
            # the guard is still sampling and can invalidate a completed run.
            # Leave cleanup to that original guard on both success and failure.
            pass
    print('SERVICE_SESSION_COMPLETE', args.run_id, flush=True)


def run_all(args):
    work = args.work.resolve()
    architecture = platform.machine()
    soc = os.environ.get('SOC_VERSION', 'unspecified-SOC')
    run_prefix = datetime.now().strftime(architecture + '-%Y%m%d-%H%M%S-%f')
    results = []
    for iteration in range(1, args.rounds+1):
        run_id = f'{run_prefix}-r{iteration}'
        guard_path = work / 'guards' / (run_id + '.json')
        command = [sys.executable, str(args.repo.resolve() / 'src/tools/validation/guard.py'),
                   '--out', str(guard_path), '--wait-seconds', '3600', '--',
                   sys.executable, str(Path(__file__).resolve()), '--session',
                   '--repo', str(args.repo.resolve()), '--work', str(work),
                   '--port', str(args.port), '--run-id', run_id]
        log_path = work / 'logs' / (run_id + '.log')
        with log_path.open('w') as log:
            subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        guard = json.loads(guard_path.read_text())
        assert guard['valid'] and guard['phase'] == 'complete', guard.get('error')
        out = work / 'measurements' / run_id
        result = json.loads((out / 'result.json').read_text())
        result['guard_valid'] = True
        results.append(result)
        save(work / 'benchmark-progress.json', results)
        print('ROUND_COMPLETE', json.dumps(result), flush=True)
    median_tpot = statistics.median(x['tpot_ms'] for x in results)
    summary = dict(rounds=results, median_tpot_ms=median_tpot,
                   decode_tps_from_median_tpot=1000/median_tpot,
                   same_outputs_across_rounds=len({x['output_sha256'] for x in results}) == 1,
                   protocol='README 2048 input / 4096 output; B=1; one full warmup per independent service; client SSE TPOT excludes TTFT',
                   scope=f'{architecture} + {soc}; 28-block kernels. Head tested; full-model reference accuracy harness is absent.')
    save(work / 'benchmark-summary.json', summary)
    rows = ['round,tpot_ms,decode_tps,ttft_ms,total_seconds,guard_valid']
    for index, result in enumerate(results, 1):
        rows.append(f"{index},{result['tpot_ms']:.6f},{result['tps']:.3f},{result['ttft_ms']:.3f},{result['total_seconds']:.3f},{result['guard_valid']}")
    (work / 'benchmark-results.csv').write_text('\n'.join(rows) + '\n')
    (work / 'benchmark-results.txt').write_text(
        f'Spark-X2.6-4B Q4_0，{architecture} / {soc} 本机测试\n'
        '输入 2048 token，输出 4096 token，batch=1，BF16 KV。\n'
        '三个独立服务实例，每轮先完整预热一次，再测量一次。\n'
        f'TPOT 中位数：{median_tpot:.6f} ms/token\n'
        f'对应 decode 吞吐：{1000/median_tpot:.3f} token/s（不含首 token 延迟）\n'
        '三轮设备 guard 均通过；预热和测量 token ID 一致。\n'
        '本仓库未包含完整模型精度参考测试，不能据此宣称全模型逐位一致。\n'
        '实际命令、环境和源码哈希见 service/；原始测量见 measurements/，设备检查见 guards/。\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18526)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--session', action='store_true')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    session(args) if args.session else run_all(args)
