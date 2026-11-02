"""Reproduce long IPC paths and inherited proxies without an NPU."""
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tools/deployment'))
from runtime_environment import runtime_environment

spec = importlib.util.spec_from_file_location('spark_benchmark_test', ROOT / 'src/tools/benchmark/benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_long_work_directory_can_bind_uuid_socket(self):
        with tempfile.TemporaryDirectory(prefix='spark-runtime-test-') as directory:
            work = Path(directory) / ('long-work-directory-' * 8)
            work.mkdir()
            env = runtime_environment(work, {'NO_PROXY': 'internal.example', 'no_proxy': 'another.example'})
            rpc = Path(env['VLLM_RPC_BASE_PATH'])
            self.addCleanup(rpc.rmdir)
            self.assertEqual(runtime_environment(work, env), env)
            self.assertEqual(env['NO_PROXY'], env['no_proxy'])
            self.assertEqual(set(env['no_proxy'].split(',')),
                             {'127.0.0.1', 'localhost', '::1', 'internal.example', 'another.example'})
            path = rpc / ('a' * 36)
            try:
                with socket.socket(socket.AF_UNIX) as sock:
                    sock.bind(str(path))
            finally:
                path.unlink(missing_ok=True)
            self.assertEqual(rpc.stat().st_mode & 0o777, 0o700)

    def test_work_directories_receive_distinct_ipc_paths(self):
        with tempfile.TemporaryDirectory(prefix='spark-runtime-test-') as directory:
            paths = [Path(runtime_environment(Path(directory) / name, {})['VLLM_RPC_BASE_PATH'])
                     for name in ['one', 'two']]
            for path in paths:
                self.addCleanup(path.rmdir)
            self.assertNotEqual(*paths)

    def test_custom_path_checks_byte_length_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory(prefix='spark-rpc-') as directory:
            root = Path(directory)
            for invalid in ['relative/path', '/tmp/' + '界' * 30]:
                with self.assertRaisesRegex(ValueError, '107 bytes'):
                    runtime_environment(root, {'VLLM_RPC_BASE_PATH': invalid})
            target = root / 'private'
            target.mkdir(mode=0o700)
            link = root / 'link'
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'IPC directory'):
                runtime_environment(root, {'VLLM_RPC_BASE_PATH': str(link)})
            self.assertEqual(runtime_environment(root, {'VLLM_RPC_BASE_PATH': str(target)})
                             ['VLLM_RPC_BASE_PATH'], str(target))

    def test_health_models_and_stream_connect_directly_with_broken_proxy(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.path)
                data = json.dumps({'data': []}).encode() if self.path == '/v1/models' else b''
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                received.append(self.path)
                self.rfile.read(int(self.headers['Content-Length']))
                data = b'data: {"choices":[{"token_ids":[7]}]}\n\ndata: [DONE]\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proxies = {'HTTP_PROXY': 'http://127.0.0.1:1', 'HTTPS_PROXY': 'http://127.0.0.1:1',
                   'ALL_PROXY': 'http://127.0.0.1:1', 'NO_PROXY': '', 'no_proxy': ''}
        try:
            with patch.dict(os.environ, proxies, clear=True), benchmark.local_client() as client:
                url = f'http://127.0.0.1:{server.server_port}'
                self.assertEqual(client.get(url + '/health', timeout=2).status_code, 200)
                self.assertEqual(client.get(url + '/v1/models', timeout=2).json(), {'data': []})
                with client.post(url + '/v1/completions', json={}, stream=True, timeout=2) as response:
                    self.assertIn(b'data: [DONE]', list(response.iter_lines()))
            self.assertEqual(received, ['/health', '/v1/models', '/v1/completions'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
