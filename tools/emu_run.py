"""Run a compiled Emmy program from its CUDA dump, without ``emmy run``.

``emmy run`` needs the Rust ``emmy_runtime`` extension and nvcc, neither of which exists on Windows. But the
``--ir cuda`` dump already carries everything a launch needs: each ``CudaOp`` node holds ``kernel_source``,
``kernel_name``, ``arg_order``, ``grid``, ``block`` and ``smem_bytes``, and every buffer carries a shape and dtype.

So this harness compiles each kernel with NVRTC, assembles it with ptxas, allocates the buffers, fills the inputs
from a seeded generator, launches, and reports the outputs. It proves a lowering change end to end on this machine.

Run:  python _setup/emu_run.py <dump-dir|10_lowering_cuda.json> [--seed N]
"""

from __future__ import annotations

import ctypes
import json
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_probe as dp  # noqa: E402
import nvrtc_probe as nv  # noqa: E402

WS = Path(__file__).resolve().parent.parent
F32 = "f32"

#: Host-side values of the last run, for a caller that wants to check the numbers instead of just the exit status.
LAST: dict = {"host": {}, "shapes": {}, "plan": None}


def plan_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        p = p / "10_lowering_cuda.json"
    if not p.exists():
        raise SystemExit(f"no CUDA dump at {p}")
    return p


def topo(nodes: dict) -> list[str]:
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str) -> None:
        mark = state.get(name, 0)
        if mark == 2:
            return
        if mark == 1:
            raise SystemExit(f"cycle at {name}")
        state[name] = 1
        for dep in nodes[name].get("inputs", []):
            if dep in nodes:
                visit(dep)
        state[name] = 2
        order.append(name)

    for name in nodes:
        visit(name)
    return order


def numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def stable_seed(name: str) -> int:
    """A per-buffer seed that does not depend on ``PYTHONHASHSEED``, so reruns compare against themselves."""
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % 1000


def fill(count: int, seed: int) -> list[float]:
    """Deterministic small values: a plain LCG, so a rerun compares against itself."""
    state, out = seed & 0xFFFFFFFF, []
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append((state / 0x3FFFFFFF) - 1.0)
    return out


def build_cubin(kernel_name: str, source: str, arch: str) -> Path:
    lib = nv.load_nvrtc()
    ptx = nv.compile_to_ptx(lib, source, arch)
    ptx_path = WS / "_out" / "emu" / f"{kernel_name}.ptx"
    ptx_path.parent.mkdir(parents=True, exist_ok=True)
    ptx_path.write_text(ptx, encoding="utf-8")
    cubin = ptx_path.with_suffix(".cubin")
    r = subprocess.run(
        [str(nv.PTXAS), f"-arch=sm_{arch}", "-o", str(cubin), str(ptx_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"ptxas failed for {kernel_name}:\n{r.stdout}{r.stderr}")
    return cubin


def main() -> int:
    args = [a for a in sys.argv[1:]]
    seed = 7
    if "--seed" in args:
        i = args.index("--seed")
        seed = int(args[i + 1])
        del args[i : i + 2]
    if not args:
        print(__doc__)
        return 2
    arch = "89"
    if "--arch" in args:
        i = args.index("--arch")
        arch = args[i + 1]
        del args[i : i + 2]
    path = plan_path(args[0])
    plan = json.loads(path.read_text(encoding="utf-8"))
    nodes = plan["nodes"]
    print(f"plan    : {path.name}  nodes={len(nodes)}  outputs={plan.get('outputs')}")

    dp.cuInit(0)
    dev = dp.cuDeviceGet(0)
    dp.cuCtxCreate(dev)
    print(f"device  : {dp.cuDeviceGetName(dev)}")

    devbuf: dict[str, int] = {}
    host: dict[str, list[float]] = {}
    shape_of: dict[str, list[int]] = {}

    def alloc(name: str, n: int) -> int:
        ptr = dp.cuMemAlloc(max(4, n * 4))
        devbuf[name] = ptr
        return ptr

    def upload(name: str, vals: list[float]) -> None:
        blob = struct.pack(f"<{len(vals)}f", *vals)
        dp.cuMemcpyHtoD(devbuf[name], blob)

    for name in topo(nodes):
        node = nodes[name]
        op = node["op"]
        outs = node["outputs"]
        if not outs:
            continue
        spec = outs[0]
        shape = spec.get("shape") or [1]
        dtype = spec.get("dtype", F32)
        if dtype != F32:
            print(f"skip    : {name} dtype={dtype} (harness handles f32 only)")
            continue
        n = numel(shape)
        shape_of[name] = shape
        alloc(name, n)

        if op == "InputOp":
            vals = fill(n, seed + stable_seed(name))
            upload(name, vals)
            host[name] = vals
            print(f"input   : {name} {shape} = {[round(v, 3) for v in vals[:4]]}…")
        elif op == "ConstantOp":
            vals = [1.0] * n
            upload(name, vals)
            host[name] = vals
            print(f"const   : {name} {shape} = ones")
        elif op == "CudaOp":
            f = node["op_fields"]
            kn = f["kernel_name"]
            cubin = build_cubin(kn, f["kernel_source"], arch)
            mod = dp.cuModuleLoadData(cubin.read_bytes())
            fn = dp.cuModuleGetFunction(mod, kn)
            params = [struct.pack("<Q", devbuf[a]) for a in f["arg_order"]]
            grid = tuple(g[0] for g in f["grid"])
            block = tuple(b[0] for b in f["block"])
            print(f"kernel  : {kn}  grid={grid} block={block} smem={f.get('smem_bytes', 0)}")
            dp.cuLaunchKernel(fn, grid, block, params, f.get("smem_bytes", 0) or 0)
            dp.cuCtxSynchronize()
            raw = dp.cuMemcpyDtoH(devbuf[name], n * 4)
            vals = list(struct.unpack(f"<{n}f", raw))
            host[name] = vals
            bad = [v for v in vals if v != v or abs(v) > 1e30]
            print(f"output  : {name} {shape} = {[round(v, 6) for v in vals[:6]]}…")
            print(f"check   : {'FAIL ' + str(len(bad)) + ' bad values' if bad else 'finite'}")
            if bad:
                return 1
        else:
            print(f"skip    : {name} op={op} (no launcher)")

    LAST["host"].clear()
    LAST["host"].update(host)
    LAST["shapes"].clear()
    LAST["shapes"].update(shape_of)
    LAST["plan"] = plan
    print("RESULT: program ran on the local GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
