"""Spark model integration; native scheduler and standard prefill are retained."""
from pathlib import Path
import torch, torch_npu
import os
from vllm.model_executor.layers.attention.attention import get_attention_context
_layer_indices={};_layers={};_packed={};_nz={};_announced=set();_gelu=None;_sigmoid=None;_last_key=None;_prefill_announced=False


def packed(layer,x):
    global _gelu,_sigmoid
    key=id(layer)
    if key not in _packed:
        if _gelu is None:
            bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(x.device)
            full=torch.nn.functional.gelu(bits,approximate='none')
            _gelu=torch.cat((full[0x3b00:0x4300],full[0xbb00:0xc300])).contiguous()
            _sigmoid=torch.sigmoid(bits.float()).bfloat16().contiguous()
        attn=layer.self_attn;rd=attn.rotary_emb.rotary_dim
        from spark_vllm_ascend.integration.weights import pack_layer,pack_attention
        up,down=pack_layer(_layer_indices[key],x.device)
        wq,wo=pack_attention(_layer_indices[key],rd,attn.g_proj.weight,x.device)
        _packed[key]=(wq,wo,up,down)
    return _packed[key]

def get_nz(kc,vc):
    key=(kc.data_ptr(),vc.data_ptr())
    if key not in _nz:
        b=kc.numel()//(128*1024)
        nk=kc.new_empty((b,4,16,128,16));nv=torch.empty_like(nk)
        _nz[key]=(kc,vc,nk,nv)
        if not torch.npu.is_current_stream_capturing():
            nk.copy_(kc.view(-1,128,4,16,16).permute(0,2,3,1,4))
            nv.copy_(vc.view(-1,128,4,16,16).permute(0,2,3,1,4))
        print('V43_NZ_ALLOC',b,flush=True)
    return _nz[key][2:]

@torch.library.custom_op('spark_v43::refresh_caches',mutates_args=())
def refresh_caches(x:torch.Tensor)->torch.Tensor:
    for kc,vc,nk,nv in _nz.values():
        nk.copy_(kc.view(-1,128,4,16,16).permute(0,2,3,1,4))
        nv.copy_(vc.view(-1,128,4,16,16).permute(0,2,3,1,4))
    return x.clone()
@refresh_caches.register_fake
def refresh_fake(x):return torch.empty_like(x)

@torch.library.custom_op('spark_v43::layer_run',mutates_args=())
def layer_run(x:torch.Tensor,residual:torch.Tensor,positions:torch.Tensor,key:str)->tuple[torch.Tensor,torch.Tensor]:
    layer,nextnorm=_layers[key];attn=layer.self_attn
    metadata,core,cache,_=get_attention_context(key)
    if x.shape[0]!=1 or metadata is None:
        out=attn(positions=positions,hidden_states=x)
        out,residual=layer.post_attention_layernorm(out,residual)
        out=layer.mlp(out)
        out,residual=nextnorm(out,residual)
        if key==_last_key and x.shape[0]>1:
            global _prefill_announced
            for pk,pv,nk,nv in _nz.values():
                nk.copy_(pk.view(-1,128,4,16,16).permute(0,2,3,1,4))
                nv.copy_(pv.view(-1,128,4,16,16).permute(0,2,3,1,4))
            if _nz and not _prefill_announced:
                print('V43_PREFILL_NZ_REFRESH',len(_nz),'tokens',x.shape[0],flush=True);_prefill_announced=True
        return out,residual
    assert x.shape==(1,2560) and x.dtype==torch.bfloat16
    assert metadata.block_tables.shape[0]==1 and metadata.block_tables.numel()>=320
    assert core.impl.sinks is None and not core.impl.enable_c8_quant and not core.impl.enable_hamming_sparse
    rd=attn.rotary_emb.rotary_dim
    assert (rd==64 and core.impl.sliding_window is None) or (rd==256 and core.impl.sliding_window==512)
    kc,vc=cache[0],cache[1]
    nk,nv=get_nz(kc,vc) if rd==64 else (kc,vc)
    wq,wo,w1,w2=packed(layer,x)
    params=[x,residual,wq,wo,w1,w2,_gelu,_sigmoid,layer.post_attention_layernorm.weight,nextnorm.weight,positions,attn.rotary_emb.cos_sin_cache,metadata.slot_mapping,metadata.block_tables]
    y=torch.empty_like(x);ro=torch.empty_like(residual)
    buffers=run_layer(params,kc,vc,y,ro,rd,layer.post_attention_layernorm.variance_epsilon,nextnorm.variance_epsilon,nk=nk,nv=nv)
    # Optional external fixed-token regression probe; disabled by serve.py by default.
    if os.environ.get('CANNBOT_CAPTURE')=='1':
        from dsl_probe import layer as capture
        capture(_layer_indices[id(layer)],x,residual,y,ro,buffers)
    if key not in _announced:
        print('V43_LAYER_EXECUTE',key,'variant','cannbot_dsl','rd',rd,'cache',tuple(kc.shape),'table',tuple(metadata.block_tables.shape),flush=True);_announced.add(key)
    return y,ro
@layer_run.register_fake
def layer_fake(x,residual,positions,key):return torch.empty_like(x),torch.empty_like(residual)

def install(model):
    global _last_key
    import cannbotdsl
    print('V43_KERNEL_VARIANT cannbot_dsl',cannbotdsl.__version__,flush=True)
    for i,layer in enumerate(model.layers):
        _layer_indices[id(layer)]=i
        nextnorm=model.layers[i+1].input_layernorm if i+1<len(model.layers) else model.norm
        _layers[layer.self_attn.core_attention.layer_name]=(layer,nextnorm)
    _last_key=model.layers[-1].self_attn.core_attention.layer_name
    print('V43_REGISTERED_LAYERS',len(_layers),'launch_aic_blocks=28 launch_aiv_lanes=56', 'soc='+os.environ['SOC_VERSION'], 'BF16',flush=True)

    from spark_vllm_ascend.integration.slot_mapping import install as install_slot4
    install_slot4(block=1024)

# Host compilation cache and workspace ABI. Device math is in kernels/.
import cannbotdsl as cb
from cannbotdsl import TensorSpec,dtypes
from spark_vllm_ascend.kernels.layer import DecodeLayer
WORKSPACE_SHAPES=[((6144,),torch.bfloat16),((24,256),torch.bfloat16),((16,),torch.bfloat16),
 ((64,16,256),torch.float32),((64,16,8),torch.float32),((4096,),torch.bfloat16),
 ((2560,),torch.bfloat16),((2560,),torch.bfloat16),((2560,),torch.bfloat16),
 ((20480,),torch.bfloat16),((10240,),torch.bfloat16),((2560,),torch.bfloat16),((16,),torch.bfloat16)]
_CACHE={}
_NZ_TEST_CACHE={}

def local_nz(kc,vc):
 key=(kc.data_ptr(),vc.data_ptr())
 if key not in _NZ_TEST_CACHE:
  nk=kc.view(-1,128,4,16,16).permute(0,2,3,1,4).contiguous()
  nv=vc.view(-1,128,4,16,16).permute(0,2,3,1,4).contiguous()
  _NZ_TEST_CACHE[key]=(kc,vc,nk,nv)
 return _NZ_TEST_CACHE[key][2:]


def run_layer(params,kc,vc,y,ro,rd,e1,e2,nk=None,nv=None):
 x,res,wq,wo,w1,w2,gelu,sigmoid,g1,g2,pos,cs,slots,table=params
 blocks=kc.numel()//(128*1024)
 args=[x.reshape(-1),res.reshape(-1),wq.reshape(-1),wo.reshape(-1),w1.reshape(-1),w2.reshape(-1),gelu,sigmoid,g1,g2,pos,cs,slots.reshape(-1)[:1],table.reshape(-1),kc.view(blocks,128,4,256),vc.view(blocks,128,4,256)]
 if nk is None:
  if rd==64:nk,nv=local_nz(kc,vc)
  else:nk,nv=kc,vc
 args += [nk.view(blocks,4,16,128,16),nv.view(blocks,4,16,128,16)]
 # TensorSpec persists a torch.Tensor ABI; Parameter.detach shares identical storage.
 args=[t.detach() if isinstance(t,torch.nn.Parameter) else t for t in args]
 key=(rd,blocks,cs.shape,table.numel(),slots.dtype,x.device)
 if key not in _CACHE:
  buffers=[(torch.zeros if i in (0,7,9) else torch.empty)(shape,device=x.device,dtype=dtype) for i,(shape,dtype) in enumerate(WORKSPACE_SHAPES)]
  tensors=args+buffers+[y.reshape(-1),ro.reshape(-1)]
  mapping={torch.bfloat16:dtypes.bfloat16,torch.float32:dtypes.float32,torch.uint8:dtypes.uint8,torch.int32:dtypes.int32,torch.int64:dtypes.int64}
  specs=[TensorSpec(tuple(t.shape),mapping[t.dtype]) for t in tensors]+[dtypes.float32,dtypes.float32]
  fn=cb.compile(DecodeLayer(rd).run,*specs)
  _CACHE[key]=(fn,buffers)
  print('CANNBOT_LAYER_COMPILED',rd,'DSL',cb.__version__,flush=True)
 fn,buffers=_CACHE[key]
 fn(*args,*buffers,y.reshape(-1),ro.reshape(-1),e1,e2)
 return buffers


# Runtime configuration and loaded module paths.
from pathlib import Path
import os,sys,json,platform,hashlib
import torch,torch_npu

def audit(model):
 d=Path(os.environ['SPARK_AB_OUT']);d.mkdir(parents=True,exist_ok=True);import vllm
 env_keys=['ASCEND_RT_VISIBLE_DEVICES','SPARK_GGUF_PATH','SPARK_EXPECTED_GGUF_SHA','SPARK_HEAD_FIXTURE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','VLLM_CACHE_ROOT','ASCEND_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','PYTORCH_NPU_ALLOC_CONF','VLLM_USE_V2_MODEL_RUNNER','SOC_VERSION','CANNBOTDSL_CACHE_DIR','CANNBOTDSL_NPU_ARCH','TRITON_CACHE_DIR','ASCEND_HOME_PATH','SPARK_WORK']
 z=dict(pid=os.getpid(),pgid=os.getpgrp(),python=sys.version,torch=torch.__version__,torch_npu=torch_npu.__version__,vllm=vllm.__version__,cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),env={k:os.getenv(k) for k in env_keys},device_name=torch.npu.get_device_name(0),head_shape=list(model.lm_head.weight.shape),head_packed_shape=list(model.lm_head.q6_head_packed.shape))
 z['module_paths']={};z['synthetic_modules']={}
 for name,module in sys.modules.copy().items():
  raw=getattr(module,'__file__',None)
  if not raw or name.split('.')[0] not in ['torch','torch_npu','vllm','vllm_ascend','cannbotdsl','triton','spark_vllm_ascend']:continue
  # These exact PyTorch ModuleType proxies advertise a synthetic __file__.
  # Their implementation modules remain in module_paths and are fully audited.
  proxies={'torch.ops':(torch.ops,'_ops.py','torch._ops','_Ops'),
           'torch.classes':(torch.classes,'_classes.py','torch._classes','_Classes')}
  if name in proxies:
   expected,virtual,owner,cls=proxies[name]
   assert module is expected and raw==virtual and type(module).__module__==owner and type(module).__name__==cls
   source=Path(sys.modules[owner].__file__).resolve();assert source.is_file()
   z['synthetic_modules'][name]={'virtual_file':raw,'implementation_module':owner,'implementation_file':str(source),'implementation_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
  else:z['module_paths'][name]=str(Path(raw).resolve())
 (d/'runtime-audit.json').write_text(json.dumps(z,indent=2));print('AB_RUNTIME_AUDIT',json.dumps(z),flush=True)
