from pathlib import Path
import hashlib,json,os
import torch,torch_npu
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.distributed import get_tensor_model_parallel_world_size
@torch.library.custom_op('spark_q6_head::linear',mutates_args=())
def linear(x:torch.Tensor,packed:torch.Tensor)->torch.Tensor:
 assert x.shape==(1,2560) and x.dtype==torch.bfloat16 and x.is_contiguous()
 y=x.new_empty((1,131072));torch.ops.spark_head_q6s.run(x,packed,y,131072,False);return y
@linear.register_fake
def fake(x,packed):return x.new_empty((1,131072))
class Q6HeadMethod(UnquantizedEmbeddingMethod):
 def apply(self,layer,x,bias=None):
  if x.shape==(1,2560) and x.dtype==torch.bfloat16 and bias is None:
   return linear(x.contiguous(),layer.q6_head_packed)
  return super().apply(layer,x,bias)
def install(model):
 p=Path(__file__).resolve().parent.parent/'native/head'
 assert get_tensor_model_parallel_world_size()==1
 head=model.lm_head;assert head is model.model.embedding
 assert tuple(head.weight.shape)==(131072,2560) and head.weight.dtype==torch.bfloat16
 validation=json.loads((Path(os.environ['SPARK_WORK'])/'head-validation.json').read_text())
 assert validation['complete'] and len(validation['decode'])==3 and len(validation['gemv'])==5 and len(validation['graph'])==6
 assert all(v['bitwise_equal'] and v['guards'] for v in validation['decode']+validation['graph'])
 assert all(v['ordered_reference_bitwise_equal'] and v['finite'] and v['guards'] for v in validation['gemv'])
 lib=p/'spark_head_q6.so';digest=hashlib.sha256(lib.read_bytes()).hexdigest();assert validation['library_sha256']==digest
 torch.ops.load_library(str(lib))
 fixture=torch.load(Path(os.environ['SPARK_HEAD_FIXTURE']),weights_only=True,map_location='cpu')
 assert torch.equal(head.weight.detach().cpu().view(torch.int16),fixture['weights'].view(torch.int16)),'Tied embedding differs from tested Q6 reconstruction'
 packed=fixture['packed'];assert packed.dtype==torch.uint8 and packed.shape==(16384,22400)
 head.register_buffer('q6_head_packed',packed.to(head.weight.device));head.quant_method=Q6HeadMethod()
 print('Q6_HEAD_INSTALLED',digest,'exact_bf16_weights',True,'packed_bytes',packed.numel(),'decode_aiv',True,flush=True)
