#!/usr/bin/env python3
"""Local stand-in for the hosted runner: compile + launch + compare.

    python tools/localrun.py <problem-id-or-dir> [--compiler ./compile.py]

Needs a real GPU and ptxas on this machine (CUDA toolkit not required).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = WS / "hackathon-2026" / "problems"
PTXAS = WS / ".venv" / "Lib" / "site-packages" / "nvidia" / "cuda_nvcc" / "bin" / "ptxas.exe"

NP = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i32": np.int32, "i64": np.int64}

# --------------------------------------------------------------------- CUDA


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("nvcuda.dll")
        self._proto()
        self.check(self.lib.cuInit(0))
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), 0))
        self.dev = dev.value
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, self.dev))
        self.ctx = ctx
        name = ctypes.create_string_buffer(64)
        self.check(self.lib.cuDeviceGetName(name, 64, self.dev))
        self.name = name.value.decode()
        self.buffers: list[int] = []

    def _proto(self) -> None:
        L = self.lib
        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        L.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        L.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        L.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        L.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        L.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        L.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        L.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        L.cuCtxSynchronize.argtypes = []
        L.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    @staticmethod
    def check(rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"CUDA driver error {rc}")

    def alloc(self, nbytes: int) -> int:
        d = ctypes.c_uint64()
        self.check(self.lib.cuMemAlloc_v2(ctypes.byref(d), nbytes))
        self.buffers.append(d.value)
        return d.value

    def to_device(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        d = self.alloc(arr.nbytes)
        self.check(self.lib.cuMemcpyHtoD_v2(d, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes))
        return d

    def from_device(self, d: int, nbytes: int, dtype) -> list:
        host = np.zeros(nbytes // np.dtype(dtype).itemsize, dtype=dtype)
        self.check(self.lib.cuMemcpyDtoH_v2(host.ctypes.data_as(ctypes.c_void_p), d, nbytes))
        return host.tolist()

    def launch(self, cubin: bytes, entry: str, grid, block, smem: int, arg_values: list[int]) -> None:
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(cubin, len(cubin))
        self.check(self.lib.cuModuleLoadData(ctypes.byref(mod), buf))
        fn = ctypes.c_void_p()
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()))
        if smem > 48 * 1024:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            self.check(self.lib.cuFuncSetAttribute(fn, 8, smem))
        holds = [ctypes.c_uint64(v) for v in arg_values]
        arr = (ctypes.c_void_p * len(holds))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holds])
        rc = self.lib.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                     smem, None, arr, None)
        self.check(rc)
        self.check(self.lib.cuCtxSynchronize())
        self.check(self.lib.cuModuleUnload(mod) if hasattr(self.lib, "cuModuleUnload") else 0)


# ------------------------------------------------------------------ helpers


def assemble(ptx: str, arch: str) -> bytes:
    """Assemble with ptxas. Uses a scratch dir: tempfile cleanup is blocked by the sandbox."""
    td = Path(os.environ.get("LOCALRUN_SCRATCH") or (ROOT / ".localrun"))
    td.mkdir(parents=True, exist_ok=True)
    p = td / "k.ptx"
    c = td / "k.cubin"
    p.write_text(ptx, encoding="ascii")
    if c.exists():
        c.unlink()
    r = subprocess.run([str(PTXAS), f"-arch={arch}", str(p), "-o", str(c)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("ptxas failed:\n" + (r.stderr or r.stdout))
    return c.read_bytes()


def input_array(a: dict) -> np.ndarray:
    dt = NP[a["dtype"]]
    shape = tuple(a["shape"])
    if "data" in a:
        return np.array(a["data"], dtype=dt).reshape(shape)
    if "seed" in a:
        return np.random.default_rng(a["seed"]).standard_normal(shape).astype(dt)
    return np.zeros(shape, dtype=dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--compiler", default=str(ROOT / "compile.py"))
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--keep", action="store_true", help="write the PTX next to the problem")
    args = ap.parse_args()

    pdir = Path(args.problem)
    if not (pdir / "meta.json").exists():
        pdir = PROBLEMS / args.problem
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    gpu = Cuda()
    print(f"gpu: {gpu.name}")

    produced: dict[str, np.ndarray] = {}
    ok = True
    for k in meta["kernels"]:
        ir = next(pdir.glob(f"{k['entry']}.ir.json"))
        r = subprocess.run([sys.executable, args.compiler, str(ir)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{k['entry']}] compiler failed:\n{r.stderr}")
            return 1
        ptx = r.stdout
        if args.keep:
            (pdir / f"{k['entry']}.ptx").write_text(ptx, encoding="ascii")
        cubin = assemble(ptx, args.arch)

        dev_args, arg_values, out_specs = [], [], []
        for a in k["args"]:
            a = dict(a)
            dt = NP[a["dtype"]]
            shape = tuple(a.get("shape", [1]))
            if a.pop("intermediate", False):
                arr = produced[a["name"]]
            elif a.get("output"):
                arr = np.zeros(shape, dtype=dt)
            else:
                arr = input_array(a)
            if a.get("scalar"):
                arg_values.append(int(a["value"]))
                continue
            d = gpu.to_device(arr)
            dev_args.append((a["name"], d, arr.nbytes, dt, a.get("output", False), shape))
            arg_values.append(d)

        gpu.launch(cubin, k["entry"], k["grid"], k["block"], k["smem_bytes"], arg_values)

        for name, d, nbytes, dt, is_out, shape in dev_args:
            if not is_out:
                continue
            vals = gpu.from_device(d, nbytes, dt)
            arr = np.array(vals, dtype=dt).reshape(shape)
            produced[name] = arr
            if name not in expected:
                continue
            ref = expected[name]
            got = arr.astype(np.float32) if dt in (np.float16, np.float32) else arr
            refc = ref.astype(got.dtype)
            close = np.allclose(got, refc, rtol=meta["rtol"], atol=meta["atol"], equal_nan=True)
            err = float(np.max(np.abs(got.astype(np.float64) - refc.astype(np.float64)))) if got.size else 0.0
            print(f"[{k['entry']}] {'PASS' if close else 'FAIL'} {name}: max abs err {err:.3e}")
            if not close:
                ok = False
                flat_g, flat_r = got.reshape(-1), refc.reshape(-1)
                bad = np.nonzero(~np.isclose(flat_g, flat_r, rtol=meta["rtol"], atol=meta["atol"], equal_nan=True))[0][:5]
                for i in bad:
                    print(f"    [{i}] got {flat_g[i]!r} want {flat_r[i]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
