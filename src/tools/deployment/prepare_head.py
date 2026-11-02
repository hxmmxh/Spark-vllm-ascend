"""Repack original Q6_K codes and scales without re-quantization."""
from pathlib import Path
import torch,numpy as np,gguf,json,hashlib
from gguf.quants import dequantize
import argparse
a=argparse.ArgumentParser();a.add_argument('--work',type=Path,required=True);args=a.parse_args()
p=args.work.resolve()/'downloads';torch.set_num_threads(4);N,K=131072,2560
out=args.work.resolve()/'head';out.mkdir(exist_ok=False)
r=gguf.GGUFReader(str(p/'Spark-X2.6-4B-Q4_0.gguf'));t=next(t for t in r.tensors if t.name=='token_embd.weight')
assert int(t.tensor_type)==14 and tuple(t.shape)==(K,N)
packed=torch.empty((N//8,22400),dtype=torch.uint8);weights=torch.empty((N,K),dtype=torch.bfloat16)
for start in range(0,N,1024):
 rows=min(1024,N-start);nt=rows//8;raw=t.data.reshape(N,10,210)[start:start+rows].copy();b=raw.reshape(-1,210)
 ql=b[:,:128];qh=b[:,128:192]
 ql=((ql.reshape(-1,2,1,64)>>np.array([0,4],dtype=np.uint8).reshape(1,1,2,1))&15).reshape(-1,8,32)
 qh=((qh.reshape(-1,2,1,32)>>np.array([0,2,4,6],dtype=np.uint8).reshape(1,1,4,1))&3).reshape(-1,8,32)
 q=((ql|(qh<<4)).astype(np.int8)-np.int8(32)).reshape(rows,K)
 sub=raw[:,:,192:208].copy().view(np.int8).reshape(nt,8,40,4).transpose(0,2,1,3).copy()
 ds=np.repeat(raw[:,:,208:].copy().view(np.float16).reshape(nt,8,10),4,axis=-1).transpose(0,2,1).copy()
 scales=ds.astype(np.float32)[:,:,:,None]*sub.astype(np.float32)
 codes=q.reshape(nt,8,40,64).transpose(0,2,1,3).reshape(nt,40,4,2,64).transpose(0,1,2,4,3).copy().reshape(nt,20480).view(np.uint8)
 value=np.concatenate([codes,sub.reshape(nt,-1).view(np.uint8),ds.reshape(nt,-1).view(np.uint8)],axis=1)
 expected=dequantize(b,t.tensor_type).reshape(rows,K)
 decoded=q.reshape(nt,8,40,4,16).transpose(0,2,1,3,4).astype(np.float32)*scales[:,:,:,:,None]
 decoded=decoded.transpose(0,2,1,3,4).copy().reshape(rows,K)
 assert np.array_equal(decoded.view(np.uint32),expected.view(np.uint32)),start
 weights[start:start+rows]=torch.from_numpy(expected.copy()).bfloat16()
 packed[start//8:(start+rows)//8]=torch.from_numpy(value)
print('HEAD_CPU_EXACT',flush=True)
torch.save(dict(packed=packed,weights=weights),out/'fixture.pt')
(out/'fixture.json').write_text(json.dumps(dict(source_sha256=json.loads((p/'DOWNLOAD_VERIFIED.json').read_text())['sha256'],gguf_type='Q6_K',all_fp32_dequant_weights_equal_gguf=True,all_cpu_bf16_weights_exact=True,packed_bytes=packed.numel()),indent=2))
