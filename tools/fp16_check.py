"""Numeric check for 16-bit kernels: emmy's own CUDA does not compile here, so numpy is the oracle.

Usage: python fp16_check.py "<expr>" "<numpy expression in terms of a, b>"
"""
import sys

import numpy as np

import oracle as O
import stress as S

EXPR = sys.argv[1]
NUMPY = sys.argv[2]
DT = np.float16 if "float16" in EXPR else np.bfloat16 if "bfloat16" in EXPR else np.float32

dump = S.WORK / "fp16"
S.emmy_dump(EXPR, dump)
ir_path = dump / "09_lowering_kernel.json"
cuda = O.json.loads((dump / "10_lowering_cuda.json").read_text())["nodes"]
pf = [n for n in cuda.values() if n["op"] == "CudaOp"][0]["op_fields"]

cubin = O.assemble(S.my_ptx(ir_path), S.WORK / "fp16.mine.cubin")
facts = O.graph_facts(ir_path)
out_names = set(O.json.loads(ir_path.read_text())["outputs"])
args = O.make_args(pf["arg_order"], facts, out_names)

rng = np.random.default_rng(7)
named = {}
for i, name in enumerate(pf["arg_order"]):
    if name in out_names or facts[name][2] is not None:
        continue  # outputs start zeroed, constants keep the value the IR gave them
    arr = rng.standard_normal(args[i].shape).astype(DT)
    args[i] = arr
    named[name] = arr
    if arr.size > 1:
        named.setdefault("x", arr)

O.launch(cubin, pf["kernel_name"], S.dims3(pf["grid"]), S.dims3(pf["block"]),
         int(pf["smem_bytes"]), args)

got = np.asarray([a for n, a in zip(pf["arg_order"], args) if n in out_names][0])
want = np.asarray(eval(NUMPY, {"np": np, DT.__name__: DT, **named}), dtype=DT)
if got.shape != want.shape:
    want = want.reshape(got.shape)
diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
print(f"args={pf['arg_order']} out={got.shape} dtype={DT.__name__}")
print(f"max abs diff = {diff.max():.3e}   (bf16 ulp ~ 8e-3, f16 ulp ~ 1e-3)")
print("first 4:", got.reshape(-1)[:4], "vs", want.reshape(-1)[:4])
