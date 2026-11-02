"""CPU-only structure and Spark device-body contract check; not a numerical test."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'src/spark_vllm_ascend'


def definitions(tree):
    result = {}
    # Python 3.12 adds an empty type_params field to non-generic definitions.
    # Ignore only that empty metadata field so 3.9/3.12 attest the same code.
    for node in ast.walk(tree):
        if getattr(node, 'type_params', None) == []:
            node._fields = tuple(field for field in node._fields if field != 'type_params')

    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = ast.dump(node, include_attributes=False)

    visit(tree.body)
    return result


def main():
    contract = json.loads((ROOT / 'src/tools/validation/kernel_contract.json').read_text())
    trees = {}
    # Only source directories, never a user's virtualenv or downloaded dependencies.
    for folder in ['src']:
        for path in (ROOT / folder).rglob('*.py'):
            text = path.read_text()
            compile(text, str(path), 'exec')
            trees[str(path.relative_to(ROOT))] = ast.parse(text)
    checked = set()
    for row in contract['device_functions']:
        body = definitions(trees[row['path']])[row['function']]
        assert hashlib.sha256(body.encode()).hexdigest() == row['ast_sha256'], row
        checked.add((row['path'], row['function']))
    for row in contract['host_functions']:
        body = definitions(trees[row['path']])[row['function']]
        assert hashlib.sha256(body.encode()).hexdigest() == row['ast_sha256'], row
    for path, digest in contract['file_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    actual = {(path, name) for path, tree in trees.items() if path.startswith('src/spark_vllm_ascend/kernels/')
              for name in definitions(tree)}
    assert actual == checked, ('Device definitions added or removed', actual ^ checked)
    unscoped_modules = {'kernels', 'integration', 'vllm_spark2_5_plugin'}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            else:
                continue
            for module in modules:
                root = module.split('.')[0]
                assert root not in unscoped_modules, (path, module)
                if path.startswith('src/spark_vllm_ascend/kernels/'):
                    assert root not in {'torch', 'torch_npu', 'vllm', 'vllm_ascend', 'integration'}, (path, module)
                    assert not module.startswith('spark_vllm_ascend.integration'), (path, module)
                if root == 'spark_vllm_ascend':
                    target = (ROOT / 'src').joinpath(*module.split('.'))
                    assert target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), (path, module)
    print(json.dumps({'python_files': len(trees), 'checked_device_functions': len(checked),
                      'checked_host_functions': len(contract['host_functions']),
                      'checked_files': len(contract['file_sha256']),
                      'scope': 'syntax/import boundaries/Spark AST only; no device execution'}, indent=2))


if __name__ == '__main__':
    main()
