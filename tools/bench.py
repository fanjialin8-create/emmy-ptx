#!/usr/bin/env python3
"""Benchmark my PTX against emmy's own CUDA for the same problem.

    python tools/bench.py p00_scalar_add r01_array_sum m01_matmul_512 a01_attention

Both sides run on the same device, the same buffers and the same launch
geometry, timed with CUDA events.  The point is to know the ceiling: if my PTX
matches emmy's own codegen the remaining gap is the schedule in the IR, not my
translation, and no amount of instruction tuning will close it.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from localrun import Cuda, NP, assemble, input_array  # noqa: E402
from oracle import nvrtc_ptx  # noqa: E402

WS = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = WS / "hackathon-2026" / "problems"


class Bench(Cuda):
    """Cuda plus timing events and a reusable launcher."""

    def __init__(self) -> None:
        super().__init__()
        L = self.lib
        L.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        L.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        L.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        L.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]

    def load(self, cubin: bytes, entry: str, smem: int):
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(cubin, len(cubin))
        self.check(self.lib.cuModuleLoadData(ctypes.byref(mod), buf))
        fn = ctypes.c_void_p()
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()))
        if smem > 48 * 1024:
            self.check(self.lib.cuFuncSetAttribute(fn, 8, smem))
        self._mods = getattr(self, "_mods", [])
        self._mods.append((mod, buf))
        return fn

    def prepare(self, values: list[int]):
        """Build the ctypes argument array once; the timed body must not allocate."""
        holds = [ctypes.c_uint64(v) for v in values]
        arr = (ctypes.c_void_p * len(holds))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holds])
        return (holds, arr)

    def call(self, fn, grid, block, smem: int, argptr) -> None:
        self.check(self.lib.cuLaunchKernel(fn, grid[0], grid[1], grid[2],
                                           block[0], block[1], block[2], smem, None, argptr, None))

    def time(self, body, warmup: int = 5, iters: int = 50) -> float:
        for _ in range(warmup):
            body()
        self.check(self.lib.cuCtxSynchronize())
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0))
        self.check(self.lib.cuEventCreate(ctypes.byref(stop), 0))
        self.check(self.lib.cuEventRecord(start, None))
        for _ in range(iters):
            body()
        self.check(self.lib.cuEventRecord(stop, None))
        self.check(self.lib.cuEventSynchronize(stop))
        ms = ctypes.c_float()
        self.check(self.lib.cuEventElapsedTime(ctypes.byref(ms), start, stop))
        self.lib.cuEventDestroy_v2(start)
        self.lib.cuEventDestroy_v2(stop)
        return ms.value / iters


def build_kernels(gpu: Bench, pdir: Path, meta: dict):
    """Allocate every argument once; returns a list describing the launch chain."""
    plan = []
    produced: dict[str, np.ndarray] = {}
    keep_dev: dict[str, int] = {}
    for k in meta["kernels"]:
        values, outs = [], []
        for a in k["args"]:
            a = dict(a)
            dt = NP[a["dtype"]]
            shape = tuple(a.get("shape", [1]))
            if a.pop("intermediate", False):
                values.append(keep_dev[a["name"]])
                continue
            arr = np.zeros(shape, dtype=dt) if a.get("output") else input_array(a)
            if a.get("scalar"):
                values.append(int(a["value"]))
                continue
            d = gpu.to_device(arr)
            keep_dev[a["name"]] = d
            if a.get("output"):
                outs.append((a["name"], d, arr.nbytes, dt, shape))
            values.append(d)
        holds, argptr = gpu.prepare(values)
        plan.append({
            "entry": k["entry"], "grid": k["grid"], "block": k["block"],
            "smem": k["smem_bytes"], "values": values, "outs": outs,
            "holds": holds, "argptr": argptr,
        })
    return plan


def fetch(gpu: Bench, plan) -> dict[str, np.ndarray]:
    out = {}
    for k in plan:
        for name, d, nbytes, dt, shape in k["outs"]:
            out[name] = np.array(gpu.from_device(d, nbytes, dt), dtype=dt).reshape(shape)
    return out


def errors(got: dict, expected: dict, rtol: float, atol: float) -> str:
    worst = 0.0
    bad = []
    for name, ref in expected.items():
        if name not in got:
            bad.append(f"{name}:missing")
            continue
        g = got[name].astype(np.float64).reshape(-1)
        r = np.asarray(ref, dtype=np.float64).reshape(-1)
        if g.size != r.size:
            bad.append(f"{name}:size")
            continue
        e = float(np.max(np.abs(g - r))) if g.size else 0.0
        worst = max(worst, e)
        if not np.allclose(g, r, rtol=rtol, atol=atol, equal_nan=True):
            bad.append(name)
    tag = "OK" if not bad else "BAD(" + ",".join(bad) + ")"
    return f"{worst:.3e} {tag}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--compiler", default=str(ROOT / "compile.py"))
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    gpu = Bench()
    print(f"gpu: {gpu.name}  arch: {args.arch}  iters: {args.iters}\n")
    print(f"{'problem':<22}{'mine us':>9}{'emmy us':>9}{'ratio':>8}   errors (mine / emmy)")
    print("-" * 78)

    for name in args.problems:
        pdir = PROBLEMS / name
        meta = json.loads((pdir / "meta.json").read_text())
        expected = dict(np.load(pdir / "expected.npz"))

        mine, theirs = [], []
        mine_err = theirs_err = "-"
        for k in meta["kernels"]:
            ir = next(pdir.glob(f"{k['entry']}.ir.json"))
            r = subprocess.run([sys.executable, args.compiler, str(ir)],
                               capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(f"{name}: compiler failed\n{r.stderr}")
                return 1
            mine.append(gpu.load(assemble(r.stdout, args.arch), k["entry"], k["smem_bytes"]))
            ref_cu = next(pdir.glob(f"{k['entry']}.ref.cu"))
            theirs.append(gpu.load(assemble(nvrtc_ptx(ref_cu.read_text(encoding="utf-8")), args.arch),
                                   k["entry"], k["smem_bytes"]))

        plan = build_kernels(gpu, pdir, meta)

        def run(fns):
            for fn, spec in zip(fns, plan):
                gpu.call(fn, spec["grid"], spec["block"], spec["smem"], spec["argptr"])

        run(mine)
        gpu.check(gpu.lib.cuCtxSynchronize())
        mine_err = errors(fetch(gpu, plan), expected, meta["rtol"], meta["atol"])
        run(theirs)
        gpu.check(gpu.lib.cuCtxSynchronize())
        theirs_err = errors(fetch(gpu, plan), expected, meta["rtol"], meta["atol"])

        t_mine = gpu.time(lambda: run(mine), iters=args.iters) * 1000
        t_theirs = gpu.time(lambda: run(theirs), iters=args.iters) * 1000
        ratio = t_mine / t_theirs if t_theirs else float("nan")
        print(f"{name:<22}{t_mine:>9.2f}{t_theirs:>9.2f}{ratio:>8.2f}x   {mine_err}  /  {theirs_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
