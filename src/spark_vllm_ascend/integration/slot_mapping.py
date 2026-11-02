"""Four-group, single-request slot mapping with parallel exact padding.

Semantics match vLLM's Apache-2.0 _compute_slot_mapping_kernel for CP=1.
The full padded output is produced on every call; no persistent-state shortcut.
"""
from vllm.triton_utils import tl,triton
BLOCK=4096
@triton.jit
def _slot_one(N,MAX,Q,P,bt,out,size:tl.constexpr,B:tl.constexpr):
    tile=tl.program_id(0)
    offsets=tile*B+tl.arange(0,B)
    if tile*B>=N:
        tl.store(out+offsets,-1,offsets<MAX)
    else:
        start=tl.load(Q).to(tl.int64);end=tl.load(Q+1).to(tl.int64)
        valid=(offsets>=start)&(offsets<end)&(offsets<N)
        pos=tl.load(P+offsets,mask=valid,other=0)
        block=pos//size
        physical=tl.load(bt+block,mask=valid,other=0).to(tl.int64)
        slots=physical*size+pos-block*size
        values=tl.where(valid,slots,-1)
        tl.store(out+offsets,values,mask=(offsets<MAX)&(valid|(offsets>=N)))

@triton.jit
def slot4_kernel(N,MAX,Q,P,B0,B1,B2,B3,O0,O1,O2,O3,
                 S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,
                 B:tl.constexpr):
    group=tl.program_id(1)
    # Keep each pointer in its own branch; this compiler rejects pointer phis.
    if group==0:_slot_one(N,MAX,Q,P,B0,O0,S0,B)
    elif group==1:_slot_one(N,MAX,Q,P,B1,O1,S1,B)
    elif group==2:_slot_one(N,MAX,Q,P,B2,O2,S2,B)
    else:_slot_one(N,MAX,Q,P,B3,O3,S3,B)

def compute(num_tokens,max_tokens,query_start_loc,positions,tables,outputs,block_sizes,block=BLOCK):
    assert len(tables)==len(outputs)==len(block_sizes)==4
    slot4_kernel[(triton.cdiv(max_tokens,block),4)](num_tokens,max_tokens,query_start_loc,positions,*tables,*outputs,*block_sizes,B=block)

_installed=False

def install(block=BLOCK):
    global _installed
    if _installed:return
    from vllm_ascend.worker.block_table import MultiGroupBlockTable
    original=MultiGroupBlockTable.compute_slot_mapping
    announced=set()
    def fused(self,num_reqs,query_start_loc,positions,positions_compressed_list=None,req_indices_compressed_list=None):
        ts=self.block_tables
        ok=(num_reqs==1 and len(ts)==4 and not positions_compressed_list and not req_indices_compressed_list
            and all(not t.is_mamba_group and t.pcp_world_size*t.dcp_world_size==1 for t in ts)
            and len({t.max_num_batched_tokens for t in ts})==1
            and all(t.block_table.gpu.is_contiguous() and t.slot_mapping.gpu.is_contiguous() for t in ts))
        if not ok:return original(self,num_reqs,query_start_loc,positions,positions_compressed_list,req_indices_compressed_list)
        max_tokens=ts[0].max_num_batched_tokens
        assert positions.shape[0]<=max_tokens and query_start_loc.numel()>=2
        compute(positions.shape[0],max_tokens,query_start_loc,positions,[t.block_table.gpu for t in ts],[t.slot_mapping.gpu for t in ts],[t.block_size for t in ts],block)
        if id(self) not in announced:
            print('SPARK_SLOT4_ACTIVE',len(ts),'block',block,'max_tokens',max_tokens,'block_sizes',[t.block_size for t in ts],flush=True);announced.add(id(self))
    MultiGroupBlockTable.compute_slot_mapping=fused;_installed=True
