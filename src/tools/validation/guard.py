"""Wait for idle hardware and guard a newly created child process group.

Never signals unrelated processes. Any uncertainty/foreign device PID invalidates
the run and stops only this guard's child group. An invalid run is not retried.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import psutil
from device_snapshot import snapshot

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--wait-seconds',type=float,default=3600);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
assert cmd and not a.out.exists()
a.out.parent.mkdir(parents=True,exist_ok=True)
state={'command':cmd,'physical_device':int(os.environ['SPARK_PHYSICAL_DEVICE']),'started':time.time(),
       'valid':False,'phase':'waiting_idle','samples':[]}

def save():a.out.write_text(json.dumps(state,indent=2)+'\n')

def sample():
    s=snapshot();s['guard_phase']=state['phase']
    for record in s.get('process_info',{}).get('processes') or []:
        try:
            process=psutil.Process(record['pid'])
            record.update(pgid=os.getpgid(process.pid),created=process.create_time(),name=process.name())
        except (psutil.NoSuchProcess,psutil.AccessDenied,ProcessLookupError,PermissionError) as exc:
            record.update(pgid=None,identity_error=type(exc).__name__)
    state['samples'].append(s);save()
    assert s['available'] and s.get('process_info',{}).get('rc')==0,('unavailable device telemetry',s)
    return s

def is_own(pid,group):
    try:return os.getpgid(pid)==group
    except ProcessLookupError:return False  # Treat an unidentifiable device PID as interference.

child=None
try:
    consecutive=0;end=time.monotonic()+a.wait_seconds
    while time.monotonic()<end:
        s=sample();util=[r['value'] for k,r in s['utilization'].items() if k in ['aicore','vectorcore','npu'] and r['rc']==0]
        assert util,'No supported activity counter'
        clear=not s['process_info']['processes'] and max(util)<=1
        consecutive=consecutive+1 if clear else 0
        if consecutive>=3:break
        time.sleep(2)
    else:raise TimeoutError('Device not idle; no child was started')
    child=subprocess.Popen(cmd,start_new_session=True)
    state.update(phase='running',pid=child.pid,created=psutil.Process(child.pid).create_time());save()
    while True:
        s=sample()
        foreign=[r for r in s['process_info']['processes'] if r.get('pgid')!=child.pid]
        if foreign:raise RuntimeError(('Foreign device process; invalidate own run',foreign))
        rc=child.poll()
        if rc is not None:break
        time.sleep(.5)
    if rc:raise RuntimeError(('Child failed',rc))
    state.update(valid=True,phase='complete',returncode=rc)
except BaseException as exc:
    state.update(valid=False,phase='failed',error=repr(exc));raise
finally:
    if child is not None:
        # This PGID was created by this guard and is never shared with external work.
        for sig in [signal.SIGTERM,signal.SIGKILL]:
            try:os.killpg(child.pid,sig)
            except ProcessLookupError:break
            if sig==signal.SIGTERM:time.sleep(2)
        child.wait()
    state['finished']=time.time();save()
