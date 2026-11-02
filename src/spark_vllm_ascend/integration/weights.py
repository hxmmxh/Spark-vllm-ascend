"""Load GGUF values and losslessly repack Q4 for Spark; host-side code, not a DSL kernel."""
import os,json
from pathlib import Path
import numpy as np
import torch
import gguf
from gguf.quants import dequantize
_reader=None

def reader():
 global _reader
 if _reader is None:
  p=Path(os.environ['SPARK_GGUF_PATH'])
  verified=json.loads(p.with_name('DOWNLOAD_VERIFIED.json').read_text())
  assert verified['sha256']==os.environ['SPARK_EXPECTED_GGUF_SHA']
  assert p.stat().st_size==verified['size']
  _reader=gguf.GGUFReader(str(p))
  assert len(_reader.tensors)==290
 return _reader

def hf_name(name):
 if name=='token_embd.weight':return 'model.embedding.weight'
 if name=='output_norm.weight':return 'model.norm.weight'
 _,i,rest=name.split('.',2)
 conv={'attn_norm.weight':'input_layernorm.weight','ffn_norm.weight':'post_attention_layernorm.weight','attn_qkv.weight':'self_attn.q_k_v_proj.weight','attn_gate.weight':'self_attn.g_proj.weight','attn_output.weight':'self_attn.out_proj.weight','ffn_gate.weight':'mlp.gate_proj.weight','ffn_up.weight':'mlp.up_proj.weight','ffn_down.weight':'mlp.down_proj.weight'}
 return f'model.layers.{i}.'+conv[rest]

def weights():
 for t in reader().tensors:
  a=dequantize(t.data,t.tensor_type).reshape(tuple(int(v) for v in reversed(t.shape)))
  yield hf_name(t.name),torch.from_numpy(a.copy()).bfloat16()
 print('GGUF_WEIGHTS_LOADED 290 Q4_0=216 Q6_K=1 F32=73',flush=True)

def unpack_q4(t):
 assert int(t.tensor_type)==2,t.name
 n,k=tuple(int(v) for v in reversed(t.shape)); b=t.data.reshape(n,k//32,18)
 s=b[:,:,:2].copy().view('<f2').reshape(n,k//32)
 v=b[:,:,2:]
 q=np.concatenate((v&15,v>>4),axis=-1).reshape(n,k).astype(np.int8)-8
 return q,s

def pack(q,s,m):
 n,k=q.shape
 # Existing megakernel layout: tiles, K64, M, 64.
 v=q.reshape(n//m,m,k//64,64).transpose(0,2,1,3).copy().reshape(n//m,m*k)
 scales=s.reshape(n//m,m,k//64,2).transpose(0,2,1,3).copy().reshape(n//m,m*k//32)
 u=(v.astype(np.int16)&15).astype(np.uint8)
 u=u.reshape(n//m,k//64,m//2,2,64)
 codes=(u[:,:,:,0,:]|(u[:,:,:,1,:]<<4)).reshape(n//m,m*k//2)
 scales=scales.reshape(n//m,k//64,m//2,2,2).transpose(0,1,2,4,3).copy().reshape(n//m,m*k//32)
 bf=torch.from_numpy(scales.astype(np.float32)).bfloat16()
 assert np.array_equal(bf.float().numpy(),scales.astype(np.float32)),"Scale is not BF16-exact"
 scale_bytes=bf.view(torch.int16).numpy().copy().view(np.uint8)
 return np.concatenate((codes,scale_bytes),axis=1)

def layer_arrays(i):
 ts={t.name:t for t in reader().tensors}
 qg,sg=unpack_q4(ts[f'blk.{i}.ffn_gate.weight']);qu,su=unpack_q4(ts[f'blk.{i}.ffn_up.weight'])
 q=np.stack((qg,qu)).reshape(2,2560,4,2560).transpose(1,0,2,3).copy().reshape(20480,2560)
 s=np.stack((sg,su)).reshape(2,2560,4,80).transpose(1,0,2,3).copy().reshape(20480,80)
 qd,sd=unpack_q4(ts[f'blk.{i}.ffn_down.weight'])
 return [(pack(q,s,8),q,s,8),(pack(qd,sd,4),qd,sd,4)]

def pack_layer(i,device):
 a,b=layer_arrays(i)
 return torch.from_numpy(a[0]).to(device),torch.from_numpy(b[0]).to(device)

def attention_arrays(i,rd):
 ts={t.name:t for t in reader().tensors};q,s=unpack_q4(ts[f'blk.{i}.attn_qkv.weight'])
 ix=[]
 for h in range(24):
  if h<20:
   for j in range(0,rd//2,4):ix.extend(h*256+t for t in [*range(j,j+4),*range(j+rd//2,j+rd//2+4)])
   ix.extend(range(h*256+rd,(h+1)*256))
  else:ix.extend(range(h*256,(h+1)*256))
 q=q[ix].copy();s=s[ix].copy();qo,so=unpack_q4(ts[f'blk.{i}.attn_output.weight'])
 return [(pack(q,s,8),q,s,8),(pack(qo,so,8),qo,so,8)]

def pack_attention(i,rd,gate,device):
 q,o=attention_arrays(i,rd)
 wq=torch.cat((torch.from_numpy(q[0]).reshape(-1),gate.detach().cpu().contiguous().view(torch.uint8).reshape(-1))).to(device)
 return wq,torch.from_numpy(o[0]).to(device)

def install(model):
 assert os.getenv("SPARK_GGUF_MEGAKERNEL")=="1", "This package supports the Spark deployment path only"
 from spark_vllm_ascend.integration.head import install as install_head
 from spark_vllm_ascend.integration.model import audit
 install_head(model)
 audit(model)
