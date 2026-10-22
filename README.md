# emmy-ptx

A PTX backend for [Emmy](https://github.com/cloudrift-ai/emmy): it reads a lowered Emmy kernel IR and writes PTX text.

Emmy's own pipeline schedules the kernel and then renders CUDA C++; this replaces the last step with PTX, so the
`ptxas`-to-`cubin` path is unchanged but the C++ source never exists.

```
python compile.py path/to/kernel.ir.json > out.ptx
```

The only thing used from Emmy is `emmy.compiler.graph.Graph.from_dict`, to read the IR. No Emmy renderer, no
`emmy compile`, no `nvcc`/NVRTC: every instruction below is emitted by hand.

## Shape of the translation

The IR arrives fully scheduled — tiling, vector widths, the shared-memory layout, the `cp.async` pipeline and the
reduction tree are all decided. So this is a one-pass, one-statement-at-a-time translation, not a scheduler:

| IR | PTX |
|---|---|
| kernel args from `arg_order` | `ld.param.u64` + `cvta.to.global.u64` |
| `Tile` | cell id from `gid`, per-axis coordinates (row-major, last axis fastest), `gid >= ncells` guard |
| `Load` / `Write` (width 1/2/4) | `ld.global.v4.f32` / `st.global.v2.f32`, or `.shared` forms |
| `Assign` | one arithmetic instruction per operator; `dtype` comes from the IR, not from inference |
| `Accum` | `name = op(base, value)`, seeded with `op.identity` before the enclosing loop |
| `WarpShuffle` | `shfl.sync.bfly.b32` butterfly over the reduction components |
| `TreeHalve` | static unroll of `if tid < s { smem[tid] += smem[tid + s] }` + `bar.sync` |
| `CpAsyncCopy` | `cp.async.cg.shared.global` (16 B) or `.ca` (4/8 B); destination = stripe base + lane offset |
| `CpAsyncCommit` / `Wait` | `cp.async.commit_group` / `cp.async.wait_group N` |
| `Smem` | `.shared` per symbol, or one `extern .shared` pool with align-16 byte offsets when it exceeds 48 KiB |

Registers are pooled by type (`%r`, `%f`, `%rd`, `%p`) and each emitted value keeps its register for the rest of the
kernel. That wastes registers but never miscompiles, and the IR is already lowered enough that pressure stays bounded.

Math is built from the hardware approximations: `exp` is `ex2.approx.f32` after scaling by log2(e), `relu` is
`max.f32 x, 0f00000000`, and the transcendental operators are composed from `ex2`/`lg2`.

Two semantics are easy to get wrong and worth naming, because both cost real debugging time:

- **Shared-memory base registers are materialised once, in the prologue.** Taking one lazily inside a branch (e.g.
  inside `if lane == 0`) leaves every other thread holding an undefined register; sm_89 happened to tolerate it and
  sm_120 crashed with an illegal address.
- **`cp.async` addresses are `smem_index` plus `lane_index`.** `smem_index` is the stripe base, `lane_index` is the
  lane's own row/column. Emitting only the stripe base lands every lane on the same 16 bytes.

## Testing

Three layers, because "it compiled" and "it is right" are different claims:

| tool | what it checks |
|---|---|
| `tools/localrun.py` | assemble with `ptxas`, launch through `nvcuda.dll` via ctypes, diff against the problem's `expected.npz` |
| `tools/oracle.py`, `tools/stress.py` | differential test: ask Emmy to compile an arbitrary torch expression, keep its own CUDA codegen as the reference, run both kernels on the same data and diff |
| `submit.py` (from the task repo) | the official runner, real hardware, hidden thresholds |

The differential harness is the interesting one: it turns any expression Emmy can lower into a new test case, which is
how the corpus was grown beyond the fifteen public problems (`cases.txt`, 47 shape and operator variants).

## Results

All fifteen public problems pass, verified on both an RTX 4060 (`sm_89`, local) and the official RTX 5090 runner
(`sm_120`). Fused and attention problems are within 1e-5 relative; matmul within 1e-4.

## Limits

- Tensor-core, TMA and swizzled `cp.async` paths are not implemented. No public problem contains them, and the emitter
  raises rather than guess.
- `f16`/`bf16` are supported as storage widths, not as an arithmetic domain.
- One kernel per module; multi-kernel graphs are emitted in order but share no state.

## Windows notes

Emmy itself targets POSIX and wants a Rust toolchain for its runtime extension, so on Windows it is used from source
with a two-line `fcntl` shim (it only ever takes `flock`). Two environment quirks: `EMMY_WORKERS=1`, because the
pipeline forks a process pool and Windows has no `fork` context, and `PYTHONUTF8=1`, because stage dumps contain
characters a GBK code page cannot encode. PTX must be written as ASCII — PowerShell's `>` redirection writes UTF-16
and `ptxas` rejects it on line 1.

## Licence

Emmy is Apache-2.0. This repository is original code; the tools link against NVIDIA's redistributable CUDA libraries
(`ptxas`, NVRTC) rather than against Emmy.
