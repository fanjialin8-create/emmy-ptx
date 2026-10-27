"""Run an emmy-generated CUDA kernel as a reference oracle, and diff it against
our PTX output on identical inputs.

The dump from `emmy compile --ir cuda` contains everything needed:
  09_lowering_kernel.json  - the KernelOp IR (shapes, constants, arg order)
  10_lowering_cuda.json    - per-node execution plan (kernel_source, grid, block, smem, arg_order)

We compile the reference `kernel_source` with NVRTC (emmy itself would use nvcc),
assemble our own PTX with ptxas, and launch both through the CUDA driver API.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Arous\Documents\deepseek-harness\default-workspace")
VENV = ROOT / ".venv"
PY = VENV / "Scripts" / "python.exe"
PTXAS = VENV / "Lib" / "site-packages" / "nvidia" / "cuda_nvcc" / "bin" / "ptxas.exe"
NVRTC_DIR = VENV / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin"
EMMY_SRC = ROOT / "emmy-hackathon" / "emmy-main"
SHIM = ROOT / "_shim"

# ---------------------------------------------------------------- driver API

_cuda = None


def cuda():
    global _cuda
    if _cuda is not None:
        return _cuda
    lib = ctypes.WinDLL("nvcuda.dll")
    vp, u64, u32, i32, cstr = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_char_p
    for name, res, args in [
        ("cuInit", i32, [u32]),
        ("cuDeviceGet", i32, [ctypes.POINTER(i32), i32]),
        ("cuCtxCreate_v2", i32, [ctypes.POINTER(vp), u32, i32]),
        ("cuModuleLoadDataEx", i32, [ctypes.POINTER(vp), vp, u32, vp, vp]),
        ("cuModuleGetFunction", i32, [ctypes.POINTER(vp), vp, cstr]),
        ("cuMemAlloc_v2", i32, [ctypes.POINTER(u64), ctypes.c_size_t]),
        ("cuMemcpyHtoD_v2", i32, [u64, vp, ctypes.c_size_t]),
        ("cuMemcpyDtoH_v2", i32, [vp, u64, ctypes.c_size_t]),
        ("cuLaunchKernel", i32, [vp, u32, u32, u32, u32, u32, u32, u32, vp, vp, vp]),
        ("cuFuncSetAttribute", i32, [vp, i32, i32]),
        ("cuCtxSynchronize", i32, []),
        ("cuMemFree_v2", i32, [u64]),
    ]:
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    check(lib.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = vp()
    check(lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    _cuda = (lib, ctx)
    return _cuda


def check(code, what):
    if code != 0:
        raise RuntimeError(f"{what} -> CUDA driver error {code}")


# ---------------------------------------------------------------- compilers


def assemble(ptx_text: str, cubin: Path) -> bytes:
    src = cubin.with_suffix(".ptx")
    src.write_text(ptx_text, encoding="ascii")
    r = subprocess.run(
        [str(PTXAS), "-arch=sm_89", str(src), "-o", str(cubin)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError("ptxas failed:\n" + (r.stderr or r.stdout))
    return cubin.read_bytes()


def nvrtc_ptx(source: str, arch: str = "compute_89") -> str:
    os.environ["PATH"] = str(NVRTC_DIR) + os.pathsep + os.environ.get("PATH", "")
    lib = ctypes.WinDLL(str(NVRTC_DIR / "nvrtc64_120_0.dll"))
    vp, i32, cstr, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t
    lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(vp), cstr, cstr, i32, vp, vp]
    lib.nvrtcCompileProgram.argtypes = [vp, i32, ctypes.POINTER(cstr)]
    lib.nvrtcGetPTXSize.argtypes = [vp, ctypes.POINTER(sz)]
    lib.nvrtcGetPTX.argtypes = [vp, cstr]
    lib.nvrtcGetProgramLogSize.argtypes = [vp, ctypes.POINTER(sz)]
    lib.nvrtcGetProgramLog.argtypes = [vp, cstr]
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]

    prog = vp()
    check(lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"k.cu", 0, None, None),
          "nvrtcCreateProgram")
    opts = [f"--gpu-architecture={arch}", "-default-device", "--std=c++17"]
    arr = (cstr * len(opts))(*[o.encode() for o in opts])
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        n = sz()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value + 1)
        lib.nvrtcGetProgramLog(prog, buf)
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError("nvrtc failed:\n" + buf.value.decode(errors="replace"))
    n = sz()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetPTX(prog, buf)
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.value.decode("ascii", errors="replace")


# ---------------------------------------------------------------- launching

NP_OF = {"f32": np.float32, "f16": np.float16, "bf16": np.float32, "f64": np.float64,
         "i32": np.int32, "i64": np.int64, "u32": np.uint32,
         "bool": np.bool_, "bool8": np.bool_, "i1": np.bool_, "u8": np.uint8, "i8": np.int8}


def launch(cubin: bytes, entry: str, grid, block, smem: int, arrays: list[np.ndarray]) -> None:
    lib, ctx = cuda()
    mod = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(cubin, len(cubin))
    check(lib.cuModuleLoadDataEx(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p), 0, None, None),
          "cuModuleLoadDataEx")
    fn = ctypes.c_void_p()
    check(lib.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()), "cuModuleGetFunction")
    if smem > 48 * 1024:
        check(lib.cuFuncSetAttribute(fn, 8, smem), "cuFuncSetAttribute")

    dev_ptrs, keep = [], []
    for a in arrays:
        a = np.ascontiguousarray(a)
        keep.append(a)
        p = ctypes.c_uint64()
        check(lib.cuMemAlloc_v2(ctypes.byref(p), max(a.nbytes, 1)), "cuMemAlloc")
        check(lib.cuMemcpyHtoD_v2(p, a.ctypes.data_as(ctypes.c_void_p), a.nbytes), "cuMemcpyHtoD")
        dev_ptrs.append(p)

    params = (ctypes.c_void_p * len(dev_ptrs))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in dev_ptrs])
    check(lib.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                             smem, None, params, None), "cuLaunchKernel")
    check(lib.cuCtxSynchronize(), "cuCtxSynchronize")
    for p, a in zip(dev_ptrs, arrays):
        out = np.empty_like(np.ascontiguousarray(a))
        check(lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), p, out.nbytes), "cuMemcpyDtoH")
        a[...] = out
        check(lib.cuMemFree_v2(p), "cuMemFree")


# ---------------------------------------------------------------- graph facts


def graph_facts(ir_path: Path):
    """name -> (shape, dtype, fill) for every value produced in the kernel IR."""
    g = json.loads(Path(ir_path).read_text())
    facts = {}
    for node in g["nodes"].values():
        fields = node.get("op_fields") or {}
        const = node["op"] == "ConstantOp"
        val = fields.get("value")
        for out in node["outputs"] or []:
            facts[out["name"]] = (tuple(out["shape"]), out["dtype"], val if const else None)
    return facts


def make_args(arg_order, facts, outputs=(), seed=1234):
    out = []
    for name in arg_order:
        shape, dtype, fill = facts[name]
        np_dt = NP_OF[dtype]
        n = int(np.prod(shape)) if shape else 1
        if name in outputs:
            arr = np.zeros(n, dtype=np_dt)
        elif isinstance(fill, (list, tuple)):
            arr = np.array(fill, dtype=np_dt).reshape(-1)[:n]
            if arr.size < n:
                arr = np.pad(arr, (0, n - arr.size), mode="edge")
        elif fill is not None:
            arr = np.full(n, fill, dtype=np_dt)
        else:
            rng = np.random.default_rng(zlib.crc32(name.encode()) ^ seed)
            arr = rng.standard_normal(n).astype(np_dt)
        out.append(arr.reshape(shape) if shape else arr.reshape(()))
    return out
