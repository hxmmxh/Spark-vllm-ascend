from pathlib import Path
import torch,torch_npu,subprocess,os,json
r=Path(__file__).resolve().parent
cann=Path(os.environ['ASCEND_HOME_PATH']);cc=cann/'bin/bisheng'
t=Path(torch.__file__).parent;n=Path(torch_npu.__file__).parent
cmds=[
 [str(cc),'-fPIC','--npu-arch=dav-3510','-c','-O3','--cce-res-usage','-x','asc',str(r/'head_q6.asc'),'-o',str(r/'kernel_head_q6.o')],
 ['g++','-std=c++17','-O2','-fPIC','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),'-I'+str(t/'include'),'-I'+str(t/'include/torch/csrc/api/include'),'-I'+str(n/'include'),'-I'+str(cann/'include'),'-c',str(r/'extension.cpp'),'-o',str(r/'extension_head_q6.o')],
 [str(cc),'-shared','-fPIC','--npu-arch=dav-3510',str(r/'kernel_head_q6.o'),str(r/'extension_head_q6.o'),'-L'+str(t/'lib'),'-L'+str(n/'lib'),'-ltorch','-ltorch_cpu','-lc10','-ltorch_npu','-o',str(r/'spark_head_q6.so')]]
for cmd in cmds:
 print('BUILD',cmd,flush=True);subprocess.run(cmd,check=True,cwd=r)
(r/'build_head_q6.json').write_text(json.dumps(cmds,indent=2))
