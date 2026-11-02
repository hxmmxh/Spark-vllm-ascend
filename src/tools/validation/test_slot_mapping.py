"""Check the installed Ascend Triton backend against exact CPU slot mappings."""
import json
import os
from pathlib import Path

import torch
import torch_npu
from spark_vllm_ascend.integration.slot_mapping import compute, slot4_kernel
from vllm.triton_utils import HAS_TRITON

assert HAS_TRITON and hasattr(slot4_kernel, "__getitem__")
torch.npu.set_device(0)
maximum = 32768
sizes = [128, 256, 512, 1024]
tables_cpu = [torch.arange(320, dtype=torch.int32).flip(0) + i * 400
              for i in range(4)]
tables = [table.npu() for table in tables_cpu]
outputs = [torch.full((maximum,), -99, dtype=torch.int64, device="npu")
           for _ in sizes]
records = []


def check(positions, label):
    positions_cpu = positions.cpu()
    for table, size, output in zip(tables_cpu, sizes, outputs):
        expected = torch.full((maximum,), -1, dtype=torch.int64)
        expected[:positions.numel()] = (table[positions_cpu // size].to(torch.int64)
                                        * size + positions_cpu % size)
        assert torch.equal(output.cpu(), expected), label
    records.append({"case": label, "all_four_groups_exact": True,
                    "full_padding_exact": True})
    print("SLOT4_EXACT", label, flush=True)


for count, offset in [(0, 0), (1, 127), (1025, 63), (2048, 3072), (4096, 0)]:
    positions = (torch.arange(count, dtype=torch.int64) + offset).npu()
    query = torch.tensor([0, count], dtype=torch.int32, device="npu")
    compute(count, maximum, query, positions, tables, outputs, sizes, block=1024)
    torch.npu.synchronize()
    check(positions, f"tokens={count},offset={offset}")

positions = torch.tensor([4095], dtype=torch.int64, device="npu")
query = torch.tensor([0, 1], dtype=torch.int32, device="npu")
compute(1, maximum, query, positions, tables, outputs, sizes, block=1024)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    compute(1, maximum, query, positions, tables, outputs, sizes, block=1024)
for offset in [4096, 127, 8192]:
    positions.fill_(offset)
    graph.replay()
    torch.npu.synchronize()
    check(positions, f"graph_position={offset}")

work = Path(os.environ["SPARK_WORK"])
(work / "slot-mapping-validation.json").write_text(
    json.dumps({"complete": True, "kernel_type": type(slot4_kernel).__name__,
                "cases": records}, indent=2))
print("SLOT4_STRICT_PASS", flush=True)
