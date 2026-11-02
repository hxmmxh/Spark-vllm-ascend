"""Audit three rounds using their raw events, benchmark request and guards."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--device', type=int, required=True)
args = parser.parse_args()
work, repo = args.work.resolve(), args.repo.resolve()
expected = json.loads((repo / 'src/tools/benchmark/data/request.json').read_text())
expected['prompt'] = json.loads((repo / 'src/tools/benchmark/data/prompt.json').read_text())
summary = json.loads((work / 'benchmark-summary.json').read_text())
assert len(summary['rounds']) == 3
sources, hashes, tpots, run_ids = [], [], [], []
for result in summary['rounds']:
    run = result['run_id']
    assert run not in run_ids
    run_ids.append(run)
    directory = work / 'measurements' / run
    guard = json.loads((work / 'guards' / (run + '.json')).read_text())
    assert guard['valid'] and guard['phase'] == 'complete' and guard['returncode'] == 0
    assert guard['physical_device'] == args.device
    assert json.loads((directory / 'request.json').read_text()) == expected
    warm, measured = [json.loads((directory / (name + '-raw.json')).read_text())
                      for name in ['warmup', 'measurement']]
    assert warm['token_ids'] == measured['token_ids']
    for raw in [warm, measured]:
        assert raw['usage']['prompt_tokens'] == 2048
        assert raw['usage']['completion_tokens'] == len(raw['token_ids']) == 4096
        events = raw['events']
        assert len(events) > 1
        assert all(a['time'] <= b['time'] for a, b in zip(events, events[1:]))
        assert [token for event in events for token in event['token_ids']] == raw['token_ids']
    events = measured['events']
    tpot = (events[-1]['time'] - events[0]['time']) * 1000 / 4095
    assert math.isclose(tpot, result['tpot_ms'], abs_tol=1e-12)
    assert math.isclose(1000 / tpot, result['tps'], abs_tol=1e-9)
    tpots.append(tpot)
    output_hash = hashlib.sha256(json.dumps(measured['token_ids']).encode()).hexdigest()
    assert output_hash == result['output_sha256']
    hashes.append(output_hash)
    assert all(json.loads((directory / 'path-validation.json').read_text()).values())
    sources.append(json.loads((work / 'service' / run / 'source-hashes.json').read_text()))
assert len(set(hashes)) == 1
assert all(source == sources[0] for source in sources)
median = statistics.median(tpots)
assert math.isclose(median, summary['median_tpot_ms'], abs_tol=1e-12)
assert math.isclose(1000 / median, summary['decode_tps_from_median_tpot'], abs_tol=1e-9)
head = json.loads((work / 'head-validation.json').read_text())
assert head['complete']
for name, field in [('head/spark_head_q6.so', 'library_sha256'),
                    ('head_reference/spark_head_bf16.so', 'reference_library_sha256')]:
    assert hashlib.sha256((repo / 'src/spark_vllm_ascend/native' / name).read_bytes()).hexdigest() == head[field]
assert json.loads((work / 'slot-mapping-validation.json').read_text())['complete']
for name in ['head-01', 'slot4-01']:
    guard = json.loads((work / 'guards' / (name + '.json')).read_text())
    assert guard['valid'] and guard['physical_device'] == args.device
report = {'complete': True, 'physical_device': args.device, 'run_ids': run_ids,
          'request_matches_protocol': True, 'raw_timing_recomputed': True,
          'warmup_and_measurement_outputs_equal': True, 'all_round_outputs_equal': True,
          'source_hashes_equal_across_rounds': True, 'head_and_slot_checks_passed': True,
          'median_tpot_ms': median, 'decode_tps': 1000 / median,
          'limitation': 'This repository does not include full-model reference accuracy validation.'}
(work / 'deployment-benchmark-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
