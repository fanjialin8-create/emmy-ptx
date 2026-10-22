"""Ad-hoc numeric probe for a single elementwise op: compare my PTX against math.*."""
import math
import sys

import numpy as np

import oracle as O
import stress as S

EXPR = sys.argv[1] if len(sys.argv) > 1 else "torch.erf(torch.randn(1,256))"

dump = S.WORK / "dbg"
S.emmy_dump(EXPR, dump)
ir_path = dump / "09_lowering_kernel.json"
cuda = O.json.loads((dump / "10_lowering_cuda.json").read_text())["nodes"]
pf = [n for n in cuda.values() if n["op"] == "CudaOp"][0]["op_fields"]
print("entry:", pf["kernel_name"], "args:", pf["arg_order"])

cubin = O.assemble(S.my_ptx(ir_path), S.WORK / "dbg.mine.cubin")
entry = pf["kernel_name"]
grid, block, smem = S.dims3(pf["grid"]), S.dims3(pf["block"]), int(pf["smem_bytes"])

facts = O.graph_facts(ir_path)
out_names = set(O.json.loads(ir_path.read_text())["outputs"])
args = O.make_args(pf["arg_order"], facts, out_names)

xvals = np.linspace(-3.0, 3.0, 256, dtype=np.float32)
for i, name in enumerate(pf["arg_order"]):
    if name not in out_names:
        args[i] = xvals.reshape(args[i].shape).copy()

O.launch(cubin, entry, grid, block, smem, args)
got = np.asarray([a for n, a in zip(pf["arg_order"], args) if n in out_names][0]).reshape(-1)
want = np.array([math.erf(float(v)) for v in xvals], dtype=np.float32)
print("        x         mine         want")
for i in range(0, 256, 32):
    print(f"{xvals[i]:+9.4f} {got[i]:+12.6f} {want[i]:+12.6f}")
print("max abs err:", float(np.max(np.abs(got - want))))
