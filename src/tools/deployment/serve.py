"""Explicit Spark server configuration; use via run_user.sh and guard.py."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from _project import selected
ROOT=Path(__file__).resolve().parents[3]

def configuration(work,out,port,capture=False,reference=False):
    work=work.resolve();out=out.resolve();profile,lock=selected()
    assert Path(sys.prefix).resolve()==(work/'.venv').resolve(),'Use the deployment virtual environment'
    assert Path(os.environ['ASCEND_HOME_PATH']).resolve().is_relative_to(work/'cann'),'Use freshly installed CANN'
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SPARK_','NUMERIC_','CANNBOT_')):env.pop(k)
    env.update(SPARK_PHYSICAL_DEVICE=os.environ['SPARK_PHYSICAL_DEVICE'],SPARK_WORK=str(work),
               SPARK_PROFILE_NAME=profile['name'],
               SPARK_V43='1',SPARK2_5_PLUGIN_OVERRIDE='1',
               VLLM_PLUGINS='ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model,spark2_5',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               SPARK_GGUF_PATH=str(work/'downloads'/lock['downloads']['gguf']['filename']),
               SPARK_GGUF_REFERENCE='0',SPARK_GGUF_MEGAKERNEL='1',SPARK_EXPECTED_GGUF_SHA=lock['downloads']['gguf']['sha256'],
               SPARK_HEAD_FIXTURE=str(work/'head/fixture.pt'),SPARK_AB_OUT=str(out),
               CANNBOT_ROOT=str(out),CANNBOT_OUT=str(out),CANNBOT_CAPTURE='1' if capture else '0',
               SPARK_VALIDATE_REFERENCE='1' if reference else '0',SPARK_GATE_OVERLAP='0',SPARK_VECTOR_FUSION='0')
    assert not reference, "Native reference execution belongs to the external validation harness"
    cmd=[str(work/'.venv/bin/vllm'),'serve',str(work/'model'),'--host','127.0.0.1','--port',str(port),
         '--served-model-name','spark','--trust-remote-code','--dtype','bfloat16',
         '--tensor-parallel-size','1','--max-model-len','40960','--max-num-seqs','1',
         '--max-num-batched-tokens','32768','--gpu-memory-utilization','0.5','--block-size','128',
         '--num-gpu-blocks-override','1408','--no-enable-prefix-caching','--enable-chunked-prefill',
         '--compilation-config',json.dumps({'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1]}),
         '--chat-template',str(work/'model/chat_template.jinja')]
    if capture:cmd.append('--enforce-eager')
    out.mkdir(parents=True,exist_ok=True)
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    keys=('SPARK_','CANNBOT_','ASCEND_','ATB_','VLLM_','HF_','TRANSFORMERS_','OMP_','OPENBLAS_','PYTORCH_')
    (out/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(keys) or k in ['PYTHONPATH','NO_PROXY','no_proxy']},indent=2))
    files=[f for f in (ROOT/'src').rglob('*') if f.is_file() and f.suffix in ['.py','.cpp','.h','.asc','.so','.sh']]
    (out/'source-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    return cmd,env

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    cmd,env=configuration(a.work,a.out,a.port)
    os.execve(cmd[0],cmd,env)
