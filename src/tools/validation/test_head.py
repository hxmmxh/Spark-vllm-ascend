"""Q6_K decoding and GEMV against freshly built, ordered BF16 reference.

The reference retains 40 FMA steps in each of 64 lanes, then the same vector
Reduce/SUM and BF16 rounding. It is NOT a conventional Cube MatMul reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--work', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2] / 'spark_vllm_ascend'
torch.set_num_threads(4)
torch.npu.set_device(0)  # Logical device; physical device selected by the guarded launcher.
fixture = torch.load(a.work / 'head/fixture.pt', weights_only=True, map_location='cpu')
weights, packed = fixture['weights'], fixture['packed']
lib = root / 'native/head/spark_head_q6.so'
ref_lib = root / 'native/head_reference/spark_head_bf16.so'
for f in [lib, ref_lib]:
    torch.ops.load_library(str(f))
op = torch.ops.spark_head_q6s.run
ref = torch.ops.spark_head_bf16.run
out_path = a.work / 'head-validation.json'
result = dict(complete=False, reference='same-order BF16 Vector head, freshly built',
              library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
              reference_library_sha256=hashlib.sha256(ref_lib.read_bytes()).hexdigest(),
              decode=[], gemv=[], graph=[])

def write():
    out_path.write_text(json.dumps(result, indent=2) + '\n')

def equal(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))

def guarded(n):
    store = torch.full((n + 32,), 17, dtype=torch.bfloat16, device='npu')
    return store, store[16:-16]

def guards(store):
    return bool((store[:16] == 17).all() and (store[-16:] == 17).all())

write()
x = torch.ones(2560, dtype=torch.bfloat16, device='npu')
for n in [8, 512, 131072]:
    w = packed[:n // 8].npu()
    s, y = guarded(n * 2560)
    op(x, w, y, n, True)
    torch.npu.synchronize()
    row = dict(rows=n, bitwise_equal=equal(y.cpu(), weights[:n].reshape(-1)), guards=guards(s))
    result['decode'].append(row); write(); print('HEAD_DECODE', row, flush=True)
    assert row['bitwise_equal'] and row['guards'], row
    del s, y, w

# Same uncompressed values in the reference's [tile, K/64, output-row, 64] layout.
n = 131072
w = packed.npu()
rw = weights.view(n // 8, 8, 40, 64).permute(0, 2, 1, 3).contiguous().view(torch.uint8).npu()
s, y = guarded(n)
rs, ry = guarded(n)
saved = []
for i, amp in enumerate([.001, .1, 1, 10, 1000]):
    torch.manual_seed(317 + i)
    xx = (torch.randn(2560) * amp).bfloat16().npu()
    op(xx, w, y, n, False); ref(xx, rw, ry, n, False)
    torch.npu.synchronize()
    row = dict(amplitude=amp, ordered_reference_bitwise_equal=equal(y, ry),
               finite=bool(y.isfinite().all() and ry.isfinite().all()), guards=guards(s) and guards(rs))
    result['gemv'].append(row); write(); print('HEAD_GEMV', row, flush=True)
    assert all(row[k] for k in ['ordered_reference_bitwise_equal', 'finite', 'guards']), row
    saved.append((xx.clone(), y.clone()))

gx = saved[0][0].clone()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    op(gx, w, y, n, False)
for i in [0, 1, 4, 2, 3, 0]:
    gx.copy_(saved[i][0]); graph.replay(); torch.npu.synchronize()
    row = dict(input_index=i, bitwise_equal=equal(y, saved[i][1]), guards=guards(s))
    result['graph'].append(row); write()
    assert row['bitwise_equal'] and row['guards'], row
result['complete'] = True
write()
print('HEAD_STRICT_PASS', flush=True)
