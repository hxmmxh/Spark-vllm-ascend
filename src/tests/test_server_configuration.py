"""Verify server and device settings without importing an inference runtime."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tools/deployment'))
from _project import read_profile
import serve


class ServerConfigurationTest(unittest.TestCase):
    def test_server_matches_api_request_and_propagates_work_directory(self):
        with tempfile.TemporaryDirectory(prefix='spark-server-test-') as directory:
            work = Path(directory)
            output = work / 'service'
            profile, lock, _, _ = read_profile('arm64-950dt')
            environment = {
                'ASCEND_HOME_PATH': str(work / 'cann/ascend-toolkit'),
                'SPARK_PHYSICAL_DEVICE': '6',
                'SPARK_WORK': '/incorrect/inherited/work',
                'SPARK_PROFILE_NAME': 'incorrect-inherited-profile',
                'SPARK_VALIDATE_REFERENCE': '1',
            }
            with patch.object(serve, 'selected', return_value=(profile, lock)), \
                    patch.object(serve.sys, 'prefix', str(work / '.venv')), \
                    patch.dict(os.environ, environment, clear=True):
                command, env = serve.configuration(work, output, 18526)
            request = json.loads((ROOT / 'src/tools/benchmark/data/request.json').read_text())
            self.assertEqual(command[command.index('--served-model-name') + 1], request['model'])
            self.assertEqual(env['SPARK_WORK'], str(work))
            self.assertEqual(env['SPARK_PROFILE_NAME'], 'arm64-950dt')
            self.assertEqual(env['SPARK_VALIDATE_REFERENCE'], '0')
            self.assertEqual(env['SPARK_PHYSICAL_DEVICE'], '6')
            self.assertEqual(json.loads((output / 'environment.json').read_text())['SPARK_WORK'], str(work))

    def test_custom_dcmi_library_uses_the_configured_path(self):
        spec = importlib.util.spec_from_file_location(
            'spark_device_snapshot_test', ROOT / 'src/tools/validation/device_snapshot.py')
        module = importlib.util.module_from_spec(spec)
        environment = {'SPARK_PHYSICAL_DEVICE': '0', 'SPARK_DCMI_LIBRARY': '/test/driver/libdcmi.so'}
        with patch.dict(os.environ, environment, clear=True), \
                patch('ctypes.CDLL', side_effect=OSError('Test driver is not loaded')) as loader:
            spec.loader.exec_module(module)
        loader.assert_called_once_with('/test/driver/libdcmi.so')
        self.assertFalse(module.snapshot()['available'])


if __name__ == '__main__':
    unittest.main()
