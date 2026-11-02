"""Read-only DCMI sampling; selected physical device mapped via installed driver API.

Enums/prototypes: /usr/local/Ascend/driver/include/dcmi_interface_api.h.
Unsupported counters are unavailable (rc/value), never silently converted to zero.
No setters, reset, frequency control, or npu-smi calls.
"""
import ctypes as C,time,os
_physical=int(os.environ.get('SPARK_PHYSICAL_DEVICE','0'));assert _physical in range(8)
class Proc(C.Structure):_fields_=[('proc_id',C.c_int),('proc_mem_usage',C.c_ulong)]
_lib=None;_error=None;_dev=None
try:
 _lib=C.CDLL(os.environ.get('SPARK_DCMI_LIBRARY','/usr/local/Ascend/driver/lib64/driver/libdcmi.so'))
 _lib.dcmiv2_init.argtypes=[];_lib.dcmiv2_init.restype=C.c_int
 rc=_lib.dcmiv2_init()
 if rc:raise RuntimeError(('dcmiv2_init',rc))
 f=_lib.dcmiv2_get_dev_id_by_chip_phy_id;f.argtypes=[C.c_int,C.POINTER(C.c_uint)];f.restype=C.c_int
 device=C.c_uint();rc=f(_physical,C.byref(device))
 if rc:raise RuntimeError(('physical device mapping',rc))
 _dev=device.value
 f=_lib.dcmiv2_get_chip_phy_id_by_dev_id;f.argtypes=[C.c_int,C.POINTER(C.c_uint)];f.restype=C.c_int
 physical=C.c_uint();rc=f(_dev,C.byref(physical))
 if rc or physical.value!=_physical:raise RuntimeError(('physical device reverse mapping',rc,physical.value))
 for name in ['dcmiv2_get_device_frequency','dcmiv2_get_device_utilization_rate']:
  f=getattr(_lib,name);f.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_uint)];f.restype=C.c_int
 f=_lib.dcmiv2_get_device_temperature;f.argtypes=[C.c_int,C.POINTER(C.c_int)];f.restype=C.c_int
 f=_lib.dcmiv2_get_device_proc_mem_info;f.argtypes=[C.c_int,C.POINTER(Proc),C.POINTER(C.c_int)];f.restype=C.c_int
except Exception as e:_error=repr(e)
def snapshot():
 out=dict(time=time.time(),physical_device=_physical,loadavg=os.getloadavg(),backend='read-only DCMI v2',available=_error is None)
 if _error:return dict(out,error=_error)
 device=_dev;out.update(logical_device_id=device)
 for name,fn,values in [('utilization','dcmiv2_get_device_utilization_rate',{'aicore':2,'hbm_bandwidth':10,'vectorcore':12,'npu':13}),('frequency','dcmiv2_get_device_frequency',{'hbm':6,'aicore':7,'vectorcore':12})]:
  out[name]={}
  for key,enum in values.items():
   value=C.c_uint();rc=getattr(_lib,fn)(device,enum,C.byref(value));out[name][key]=dict(rc=rc,value=value.value if rc==0 else None)
 value=C.c_int();rc=_lib.dcmiv2_get_device_temperature(device,C.byref(value));out['temperature']=dict(rc=rc,value=value.value if rc==0 else None)
 entries=(Proc*65536)();count=C.c_int(65536);rc=_lib.dcmiv2_get_device_proc_mem_info(device,entries,C.byref(count))
 assert rc!=0 or 0<=count.value<=65536
 out['process_info']=dict(rc=rc,count=count.value if rc==0 else None,processes=[dict(pid=entries[i].proc_id,memory_raw=entries[i].proc_mem_usage) for i in range(count.value)] if rc==0 else None)
 return out
def idle_preflight():
 samples=[];consecutive=0
 for i in range(30):
  s=snapshot();samples.append(s)
  known=[v['value'] for k,v in s.get('utilization',{}).items() if k in ['aicore','vectorcore','npu'] and v['rc']==0]
  procs=s.get('process_info',{})
  if not s['available'] or procs.get('rc')!=0 or not known:raise RuntimeError(('Cannot establish idle device',s))
  foreign=[v for v in (procs.get('processes') or []) if v['pid']!=os.getpid()]
  consecutive=consecutive+1 if not any(v>1 for v in known) and not foreign else 0
  if consecutive>=3:
   return dict(samples=samples,scope='3 consecutive idle samples with no other device process before server start; not proof no other job can start later',known_utilization_available=bool(known),unsupported_counters_are_unavailable=True)
  time.sleep(1)
 raise RuntimeError(('Selected device did not become idle within30s; no benchmark started',samples[-3:]))
if __name__=='__main__':
 import json
 import sys
 print(json.dumps(idle_preflight() if '--idle' in sys.argv else snapshot(),indent=2))
