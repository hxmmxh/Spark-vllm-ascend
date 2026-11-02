"""CPU-side imports and library path checks before guarded NPU validation."""
import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from _project import selected

parser = argparse.ArgumentParser()
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--profile', choices=['x86-950pr', 'arm64-950dt'], required=True)
args = parser.parse_args()
work = args.work.resolve()
profile, lock = selected(check_host=True)
assert args.profile == profile['name']
architecture, soc, base_triton = (profile['host_arch'], profile['soc_version'], profile['base_triton'])
assert platform.machine() == architecture
assert os.environ['SOC_VERSION'] == soc
assert sys.version_info[:2] == (3, 12)
assert Path(sys.prefix).resolve() == work / '.venv'
for key in ['ASCEND_HOME_PATH', 'ATB_HOME_PATH']:
    assert Path(os.environ[key]).resolve().is_relative_to(work / 'cann'), key

import torch
import torch_npu
import triton
import cannbotdsl
from torch_npu.op_plugin.atb import _atb_ops
from vllm.triton_utils import HAS_TRITON
from spark_vllm_ascend.integration.slot_mapping import slot4_kernel

assert _atb_ops.NNAL_EX is None, repr(_atb_ops.NNAL_EX)
assert HAS_TRITON and hasattr(slot4_kernel, '__getitem__'), 'Ascend Triton backend unavailable'
expected = {'torch': '2.10.0+cpu', 'torch-npu': '2.10.0.post4',
            'triton': base_triton, 'triton-ascend': '3.2.2', 'cannbot-dsl': '0.7.1',
            'numpy': '1.26.4', 'transformers': '5.5.4'}
versions = {name: importlib.metadata.version(name) for name in expected}
assert versions == expected, versions
for name in ['vllm', 'vllm-ascend']:
    versions[name] = importlib.metadata.version(name)
    assert versions[name].split('+')[0] == '0.23.0', versions
loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libatb' in line and line.split()[-1].startswith('/')})
assert loaded and all(Path(path).resolve().is_relative_to(work / 'cann') for path in loaded)
record = {'complete': True, 'profile': args.profile, 'python': platform.python_version(),
          'platform': platform.platform(), 'packages': versions,
          'triton_import_version': triton.__version__, 'loaded_atb_libraries': loaded,
          'soc_version': soc, 'physical_device': os.environ['SPARK_PHYSICAL_DEVICE']}
(work / 'deployment-runtime.json').write_text(json.dumps(record, indent=2) + '\n')
print('RUNTIME_IMPORTS_PASS', json.dumps(record, indent=2))
