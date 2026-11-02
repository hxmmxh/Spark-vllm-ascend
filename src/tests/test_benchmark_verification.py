"""Exercise the relocated result auditor with synthetic records, without an NPU."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class BenchmarkVerificationTest(unittest.TestCase):
    def test_audits_raw_records_and_rejects_request_drift(self):
        with tempfile.TemporaryDirectory(prefix='spark-results-test-') as directory:
            repo = Path(directory) / 'repo'
            work = Path(directory) / 'work'
            request = json.loads((ROOT / 'src/tools/benchmark/data/request.json').read_text())
            prompt = json.loads((ROOT / 'src/tools/benchmark/data/prompt.json').read_text())
            save(repo / 'src/tools/benchmark/data/request.json', request)
            save(repo / 'src/tools/benchmark/data/prompt.json', prompt)
            request['prompt'] = prompt
            ids = list(range(4096))
            output_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            rounds = []
            for index in range(3):
                run = f'synthetic-{index}'
                output = work / 'measurements' / run
                raw = {'usage': {'prompt_tokens': 2048, 'completion_tokens': 4096},
                       'token_ids': ids, 'events': [{'time': 1.0, 'token_ids': ids[:1]},
                                                   {'time': 5.095, 'token_ids': ids[1:]}]}
                tpot = (5.095 - 1.0) * 1000 / 4095
                save(output / 'request.json', request)
                for name in ['warmup', 'measurement']:
                    save(output / f'{name}-raw.json', raw)
                save(output / 'path-validation.json', {'synthetic_path': True})
                save(work / 'guards' / f'{run}.json',
                     {'valid': True, 'phase': 'complete', 'returncode': 0, 'physical_device': 0})
                save(work / 'service' / run / 'source-hashes.json', {'synthetic_source': 'same'})
                rounds.append({'run_id': run, 'tpot_ms': tpot, 'tps': 1000/tpot,
                               'output_sha256': output_hash})
            save(work / 'benchmark-summary.json', {'rounds': rounds, 'median_tpot_ms': tpot,
                                                  'decode_tps_from_median_tpot': 1000/tpot})
            head = {'complete': True}
            for relative, field in [('head/spark_head_q6.so', 'library_sha256'),
                                    ('head_reference/spark_head_bf16.so', 'reference_library_sha256')]:
                path = repo / 'src/spark_vllm_ascend/native' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic test fixture, never loaded')
                head[field] = hashlib.sha256(path.read_bytes()).hexdigest()
            save(work / 'head-validation.json', head)
            save(work / 'slot-mapping-validation.json', {'complete': True})
            for name in ['head-01', 'slot4-01']:
                save(work / 'guards' / f'{name}.json', {'valid': True, 'physical_device': 0})
            command = [sys.executable, str(ROOT / 'src/tools/deployment/verify_results.py'),
                       '--work', str(work), '--repo', str(repo), '--device', '0']
            checked = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            report = json.loads((work / 'deployment-benchmark-verification.json').read_text())
            self.assertTrue(report['complete'])
            self.assertTrue(report['request_matches_protocol'])
            request['seed'] = 0
            save(work / 'measurements/synthetic-0/request.json', request)
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == '__main__':
    unittest.main()
