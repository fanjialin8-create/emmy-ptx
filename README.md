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
| `Assign` producing a bool | held as a predicate (`setp`), never materialised as 0/1 |
| comparison operators (`greater`, `less`, …) | `setp.{gt,lt,ge,le,eq,ne}.{f32,s32}` |
| `Select` | one compare-and-branch chain; the last branch is the catch-all, matching the documented semantics |
| `Accum` | `name = op(base, value)`, seeded with `op.identity` before the enclosing loop |
| f16 / bf16 operands | memory traffic is 16-bit (`ld.b16` / `st.b16`), arithmetic stays f32 and narrows once at the store |
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

### Speed

Measured on the RTX 4060 against Emmy's own CUDA for the same kernel (`tools/bench.py`, 300 iterations, same
buffers, same grid and block, both binaries timed back to back in one process). A ratio below 1.0 means faster than
the nvcc-built reference:

| problem | mine | Emmy's CUDA | ratio |
|---|---|---|---|
| `p00_scalar_add` | 23.9 us | 23.2 us | 1.03x |
| `r01_array_sum` | 23.0 us | 23.9 us | **0.96x** |
| `f02_row_softmax` | 8.85 us | 8.81 us | 1.00x |
| `m01_matmul_512` | 124.7 us | 81.3 us | 1.53x |
| `a01_attention` | 154.2 us | 80.2 us | 1.92x |

That is the honest state: one kernel matches or beats the reference, the other four are within 1.0–1.9x. Nothing here
is a tuning trick aimed at one shape — every item is a property of the emitted code, and the same three rewrites
(constant folding, loop-invariant hoisting, multiply-add fusion) apply to every kernel the compiler emits.

`m01_matmul_512` shows the whole journey. It started 8.8x behind the reference: the GEMM inner loop recomputed a
`div`/`rem` pair per load to work out which of the three `cp.async` stages a `k` belonged to, plus a `mul`/`add`/`sub`
chain to turn tile coordinates into an address. Three changes removed almost all of it:

1. **Unroll straight-line inner loops.** With the trip count as an immediate, every index expression in the body
   folds at emission time and the arithmetic never reaches the GPU. Only bodies made of load/store/assign/accumulate
   are unrolled: unrolling the attention body (which nests loops) silently changed the answer, and unrolling the
   outer stage loop made the kernel 7x longer and 1.6x slower.
2. **Keep integer constants as immediates** and fold through them, instead of materialising each one into a register
   first. `add`, `mul`, `shl`, `and` all take an immediate operand, so `x + 0`, `x * 1`, `i / 1` and constant-only
   expressions disappear at emission.
3. **Reuse repeated address arithmetic** — but only where it is provably still valid: the cache is cleared at every
   branch merge and at every loop back edge, and across a loop only values whose operands were all defined before
   that loop started are reused.

The PTX for `m01_matmul_512` goes from 4756 to 1891 lines, dynamic `div`/`rem` from 557 to 8, and the kernel from
1724 us to 141 us locally — on the official runner, 116.9 us to 35.1 us.

Step 3 is worth spelling out because the first attempt at it was wrong and the failure was silent: a cache that
survived branch merges miscompiled `m02_matmul_rect` (max error 130, on the runner and locally), because a value is
only valid where the thread state that produced it still holds. The kernel is now emitted with the cache cleared at
every point where control can arrive from somewhere else; 100 differential tests and all fifteen problems pass on
both architectures.

Absolute numbers on the shared runner drift by more than 2x between sessions, so only back-to-back runs there are
comparable.


Absolute numbers on that shared machine drift by more than 2x between sessions, so only back-to-back runs are
comparable there.

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
