"""Reconstruct the exact BF16 prefill model from the locked Q4_0 GGUF.

No original BF16 checkpoint, cached fixture, requantization, or device needed.
The decode path subsequently pre-packs original GGUF codes in integration/weights.py.
"""
import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path
from _project import selected

import gguf
import torch
from gguf.quants import dequantize
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]
def hf_name(name):
 if name=='token_embd.weight':return 'model.embedding.weight'
 if name=='output_norm.weight':return 'model.norm.weight'
 _,i,rest=name.split('.',2)
 conv={'attn_norm.weight':'input_layernorm.weight','ffn_norm.weight':'post_attention_layernorm.weight','attn_qkv.weight':'self_attn.q_k_v_proj.weight','attn_gate.weight':'self_attn.g_proj.weight','attn_output.weight':'self_attn.out_proj.weight','ffn_gate.weight':'mlp.gate_proj.weight','ffn_up.weight':'mlp.up_proj.weight','ffn_down.weight':'mlp.down_proj.weight'}
 return f'model.layers.{i}.'+conv[rest]
from fetch import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    lock = selected()[1]
    spec = lock['downloads']['gguf']
    source = work / 'downloads' / spec['filename']
    assert source.stat().st_size == spec['size'] and sha256(source) == spec['sha256']
    torch.set_num_threads(4)
    output = work / 'model'
    output.mkdir(exist_ok=False)
    for f in (work / 'downloads/model').iterdir():
        shutil.copy2(f, output / f.name)
    shutil.copy2(ROOT / 'src/spark_vllm_ascend/plugin/chat_template.jinja', output / 'chat_template.jinja')
    (source.parent / 'DOWNLOAD_VERIFIED.json').write_text(json.dumps({'size': spec['size'], 'sha256': spec['sha256'], 'url': spec['url']}))
    reader = gguf.GGUFReader(str(source))
    assert len(reader.tensors) == 290
    batch, sizes, weight_map, tensor_hashes, shards = {}, [], {}, {}, []

    def flush():
        if not batch:
            return
        name = f'quantized-values-{len(shards)+1:03d}.safetensors'
        save_file(batch, str(output / name), metadata={'format': 'pt'})
        restored = load_file(str(output / name))
        for key, tensor in batch.items():
            assert torch.equal(restored[key].view(torch.uint8), tensor.view(torch.uint8))
            weight_map[key] = name
        shards.append({'file': name, 'sha256': sha256(output / name)})
        batch.clear()

    for t in reader.tensors:
        a = dequantize(t.data, t.tensor_type).reshape(tuple(int(v) for v in reversed(t.shape)))
        tensor = torch.from_numpy(a.copy()).bfloat16().contiguous()
        if sum(x.numel()*x.element_size() for x in batch.values()) + tensor.numel()*2 > 512*1024**2:
            flush()
        name = hf_name(t.name)
        assert name not in tensor_hashes
        batch[name] = tensor
        sizes.append(tensor.numel()*2)
        tensor_hashes[name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    flush()
    (output / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': sum(sizes)}, 'weight_map': weight_map}, indent=2))
    (work / 'model-preparation.json').write_text(json.dumps({'source_sha256': spec['sha256'], 'tensor_count': len(weight_map), 'shards': shards, 'tensor_hashes': tensor_hashes, 'method': 'gguf.dequantize -> BF16; source Q4_0 and Q6_K values retained'}, indent=2))
    print('MODEL_PREPARED', len(weight_map), flush=True)


if __name__ == '__main__':
    main()
