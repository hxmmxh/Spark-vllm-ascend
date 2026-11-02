"""Single batch-one decode kernel and its explicit UB arena; no torch/vLLM dependency."""
import cannbotdsl as cb
from cannbotdsl import dtypes
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.scalar import load_bypass
from cannbotdsl.ops.sync import global_sync_all, vec_sync_all, vec_sync_block_arrive, vec_sync_block_wait, PIPE
from spark_vllm_ascend.kernels.linear import NativePool, NativeLinear, NativeUp
from spark_vllm_ascend.kernels.qkv import FusedQkvStage
from spark_vllm_ascend.kernels.attention import AttentionStage
from spark_vllm_ascend.kernels.vector import gate_prepare, merge_attention
import math
from cannbotdsl import make_buffer, make_channel

class UBArena:
    def __init__(self):
        self.root=cb.UB.view(262144)
        self.cursor=0
        self.high_water=0

    def rewind(self):
        # Caller finishes all old accesses before reusing phase addresses.
        # Manual storage roots do not support channel_rewind; resource identities
        # remain distinct, with alias protection supplied by the phase barriers.
        self.cursor=0

    def view(self,shape,dtype):
        if isinstance(shape,int):shape=(shape,)
        sizes={dtypes.uint8:1,dtypes.bfloat16:2,dtypes.float32:4,dtypes.uint32:4,dtypes.int32:4}
        size=math.prod(shape)*sizes[dtype]
        offset=(self.cursor+31)//32*32
        self.cursor=offset+(size+31)//32*32
        self.high_water=max(self.high_water,self.cursor)
        if self.cursor>253952:raise ValueError(('UB phase exceeds safe capacity',self.cursor))
        return self.root.reinterpret(dtype=dtype,shape=shape,offset=offset)

    def channel(self,shape,dtype,depth=1,kind=None,data_format='nd'):
        if data_format!='nd':raise ValueError(('explicit UB format',data_format))
        return make_channel([self.view(shape,dtype) for _ in range(depth)],kind=kind)

    def buffer(self,shape,dtype):
        return make_buffer(self.view(shape,dtype))

class DecodeLayer:
 def __init__(self,rd):self.rd=rd

 @kernel
 def kernel(self,x,res,wq,wo,w1,w2,gelu,sigmoid,g1,g2,pos,cs,slots,table,kc,vc,nk,nv,
            rawq,qkv,gate,partial,lse,merged,oproj,norm,resmid,rawup,mid,down,rawgate,y,ro,e1,e2):
  lane=get_block_idx()*2+get_subblock_id()
  pv=load_bypass(pos.ptr(0))
  slot=load_bypass(slots.ptr(0))
  arena=UBArena()
  qproj=FusedQkvStage(self.rd,arena)
  qproj.execute_fused(x,wq[0:8847360,],rawq,cs[pv,None].view(self.rd),qkv,kc,vc,nk,nv,slot,lane,56)
  if lane>=40:
   gate_id=lane-40
   wg=wq[8847360:8929280,].view(dtype=dtypes.bfloat16).view(16,2560)
   qproj.gate_from_expanded_input(wg[gate_id,None].view(2560),rawgate[gate_id:gate_id+1,])
  global_sync_all()
  arena.rewind()
  if lane==55:gate_prepare(rawgate,sigmoid,gate,arena)
  attn=AttentionStage(self.rd==64,arena)
  attn.execute(qkv[0:16,None],kc,vc,nk,nv,table,pos,partial,lse)
  # Complete local producers before reusing their UB, as in the native path.
  vec_sync_all()
  arena.rewind()
  pool=NativePool(arena.root)
  o=NativeLinear(4096,16,pool)
  up=NativeUp(pool)
  dn=NativeLinear(10240,8,pool)
  o.prefetch_o(wo,lane)
  vec_sync_block_arrive(PIPE.MTE3,14,mode=0)
  vec_sync_block_wait(PIPE.MTE2,14,mode=0)
  begin=0
  width=5120
  if self.rd==256:
   width=128
   begin=max(0,pv-512)//128*128
  parts=(pv+1-begin+width-1)//width
  if lane<16:
   merge_attention(partial,lse,gate[lane:lane+1,],merged[lane*256:(lane+1)*256,],lane,parts,pool)
  vec_sync_all()
  vec_sync_block_arrive(PIPE.MTE3,15,mode=0)
  vec_sync_block_wait(PIPE.S,15,mode=0)
  pool.prefetch_metadata(res,g1,gelu)
  o.execute_o(merged,wo,oproj,lane,up,w1)
  vec_sync_all()
  vec_sync_block_arrive(PIPE.MTE3,15,mode=0)
  vec_sync_block_wait(PIPE.S,15,mode=0)
  down_slot=up.execute(oproj,w1,resmid,mid,lane,e1,dn,w2)
  vec_sync_all()
  vec_sync_block_arrive(PIPE.MTE3,15,mode=0)
  vec_sync_block_wait(PIPE.S,15,mode=0)
  dn.execute_down(mid,w2,down,lane,down_slot)
  vec_sync_all()
  vec_sync_block_arrive(PIPE.MTE3,15,mode=0)
  vec_sync_block_wait(PIPE.S,15,mode=0)
  if lane==0:pool.final_norm(down,resmid,g2,y,ro,e2)
  vec_sync_all()
  vec_sync_block_arrive(PIPE.MTE3,15,mode=0)
  vec_sync_block_wait(PIPE.S,15,mode=0)

 @host
 def run(self,x,res,wq,wo,w1,w2,gelu,sigmoid,g1,g2,pos,cs,slots,table,kc,vc,nk,nv,
            rawq,qkv,gate,partial,lse,merged,oproj,norm,resmid,rawup,mid,down,rawgate,y,ro,e1,e2):
  self.kernel[28](x,res,wq,wo,w1,w2,gelu,sigmoid,g1,g2,pos,cs,slots,table,kc,vc,nk,nv,
            rawq,qkv,gate,partial,lse,merged,oproj,norm,resmid,rawup,mid,down,rawgate,y,ro,e1,e2)
