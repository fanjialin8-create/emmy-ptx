# Kernel IR → CUDA C++ 地图（为「加一条 PTX 发射通路」准备）

> 2026-10-21 只读调研的结论，题目未发时先固化下来。所有行号相对 `emmy-hackathon/emmy-main`。
> 两条前提被纠正：`emmy/compiler/ir/kernel/ARCHITECTURE.md` **不存在**（kernel 方言说明在
> `emmy/compiler/ir/kernel/ir.py:1-23`，方言总览在 `emmy/compiler/ir/ARCHITECTURE.md`）；
> `backend/cuda/runtime.py` **不存在**（执行是 Rust 的 `crates/emmy-runtime`）。

## 管线位置

- 生产者：`pipeline/passes/lowering/kernel/` —— `010_materialize.py` 加 19 个同级规则
  （030_stamp_types, 035_shared_carry, 040_split_invariant_divides, 045_merge_select_loads,
  047_reuse_lane_loads, 050_vectorize_loads, 080_vectorize_stores, 085_fast_exp, 090_guard_reductions,
  095_interleave_loads, 096_pair_ldmatrix_loads, 097_widen_fragment_stores, 100_loopify,
  110_drop_redundant_syncs, 115_spread_one_cell）
- 消费者：`pipeline/passes/lowering/cuda/010_lower_kernelop.py:70` 的 `rewrite()` →
  `:115` 调 `render_kernelop(...)` → `:146` 返回 `CudaOp`
- `pipeline/__init__.py:44-45`：`KERNEL_PASSES = [*TILE_PASSES, "lowering/kernel"]`；
  `CUDA_PASSES = [*KERNEL_PASSES, FINAL_LOWERING_PASS]`；入口 `backend/cuda/backend.py:101-108` `CudaBackend.compile`
- 方言链：frontend → tensor → loop → tile → kernel → cuda（`ir/ARCHITECTURE.md:9-17`）

## Kernel IR 清单（`KernelOp` 在 `ir/kernel/ir.py:2927`；`smem_buffers` :2954，`smem_bytes` :2963）

- 共享/纯计算：Load, Assign, Affine, Accum, Carry, Pre, Init, Let, Write, Select+SelectBranch,
  ZeroPrologue, Loop, StridedLoop, Cond, Call/Subroutine
- 硬件：Tile :182（block_threads/aux_threads/is_static_grid/n_elements）, Smem :78, Sync :138,
  SetMaxNReg :761, TreeHalve :807, WarpShuffle :916, WarpBroadcast :990
- 数据搬运：CpAsyncCopy :504, CpAsyncCommit :589, CpAsyncWait :602, TmaDescriptor :617, TmaLoad :655,
  MbarrierInit :696, MbarrierArriveExpectTx :720, MbarrierArrive :740, MbarrierWait :785
- 张量核：FragLayout :1016, FragmentApply :1105, FragmentRowReduce :1287, RegFragment :1381,
  BlockScaleLoad :1605, LdmatrixLoad :1662, MmaSyncPtx :2008, FragmentPromote :2076, FragmentRepack :2106,
  WgmmaDescriptor :2166, WgmmaMma :2206, WgmmaFence :2277, WgmmaCommit :2290, WgmmaWait :2303, RegStore :2320

## 发射器解剖（`ir/kernel/render.py`，1665 行，字符串拼接，不是表驱动 visitor）

- 唯一公开入口 `render_kernelop(...) -> str` :1416（`__all__` :1665）
- `RenderCtx` :1469，内含 `CudaRenderTarget()` :1470、`_INTRINSIC_TO_CUDA` :1375、`_BUILTIN_TO_CUDA` :1392
- 签名 `:1503-1533`；函数体 `render_body` :1537；lane/warp 声明 :1542；动态 smem 池 :1544-1557；
  paged base :1558-1566；`_GRID_DEPENDENCY` :1573；prelude :1625-1628；头文件 :1629
- **分派是每类一个虚方法**：`Stmt.render(self, ctx) -> list[str]`（`ir/stmt/base.py:851`，基类 :859 抛错）；
  `render_body`（:870）在 :986 处扁平成 `s.render(ctx)`，之前做字面量内联（:892-903）和 EMMY_READABLE 折叠（:917-978）
- 拼写层：`RenderTarget` Protocol（`compiler/render_target.py:24`：type_name :31, literal :39, convert :50,
  bitcast :56, intrinsic :59, has_native_op :68, vector_type :74），由 `CudaRenderTarget`
  （`backend/cuda/render_target.py:148`）实现
- 其它：`op_to_expr`（`stmt/base.py:289`）、`render_index`（:414）、`render_merge_program`（:368）
- prelude 常量：`_TMA` :102, `_CP_ASYNC` :198, `_MMA_SYNC` :575, `_swizzle_prelude` :1281,
  `_WGMMA_PRELUDE` :1310, `_wgmma_wrapper` :1329, `_wgmma_prelude` :1366, `_launch_bounds_for` :1653,
  `_compute_dynamic_smem_offsets` :1633

## 内联 PTX 全清单（PTX 发射器要原样复现的东西）

`render.py`：mbarrier.init :107；fence.proxy.async :114,:131；mbarrier.arrive.expect_tx :115；
mbarrier.arrive :132；mbarrier.try_wait.parity :147；cp.async.bulk.tensor 2d/3d/4d/5d :155,:164,:173,:182；
cp.async.cg/.ca + commit/wait :201,:206,:212,:216,:221；f16 unpack :263；mma.sync.m8n8k4 :509,:522,:534,:543；
ldmatrix x4/x2/x2.trans/x4.trans :578,:584,:593,:603,:609,:618；mma.sync.m16n8k16 f16/bf16/f16acc
:884,:892,:905；e4m3/e5m2 :1049,:1057；block-scaled :1117；wgmma setp :1358；griddepcontrol.wait :1413

`ir/kernel/ir.py`：bar.sync :178,:861；setmaxnreg :781；wgmma.fence :2286；wgmma.commit_group :2299；
wgmma.wait_group :2316。另有 `__cvta_generic_to_shared`（render.py:1312）、`__popc/__popcll`（render_target.py:213）

## 接缝（PTX 后端要动的四个位置）

1. **Lowering**：在 `010_lower_kernelop.py:70` 旁加一条同级规则，产出 `PtxOp`（对标 `CudaOp`，`ir/cuda/ir.py:43`），
   `kernel_source` 里装 PTX
2. **Emitter**：新模块镜像 `render_kernelop` 的签名，吃同一份 `KernelOp.body`。
   **只换 RenderTarget 不够** —— 它只覆盖名字/字面量/转换/内建/向量类型，而 `Stmt.render` 的方法体是 C 语法，
   所以 PTX 通路必须每类自己写一遍 render 走查
3. **Backend**：继承 `Backend`（`backend/base.py:119`），实现唯一的抽象 `compile`（:160），`run`（:164）可选
4. **边界对象是「源码字符串」**：`CudaOp.kernel_source` → `KernelSpec`（`backend/plan.py:118-127`：source,
   binary_key, arch_specific）→ `ExecutionPlan`（`plan_from_graph`, plan.py:214）→ `_compile_kernels`
   （`program.py:85`）→ cubin 路径（`program.py:65-82` → `nvcc.compile_kernel`, `nvcc.py:162`）→ Rust 运行时按路径加载。
   **PTX 后端要么仍产出一个可加载的二进制路径，要么让 plan/runtime 接受 PTX**

## 风险

- `__launch_bounds__`（render.py:1534，经 `_launch_bounds_for` :1653 含 aux_threads）是 C++ 专有 → `.maxntid`
- 静态/动态 smem 以 ptxas 48KB 上限为门（`STATIC_SMEM_CAP` render.py:42）+ `extern __shared__`（:1556）→
  `.shared` / `.extern .shared`；动态 smem 上限属性由运行时（驱动调用）负责
- `const` / `__restrict__`（:1504, :1512）影响 codegen，PTX 无对应拼写
- `effective_flags()`（`nvcc.py:33`）把 FAST_MATH + `EMMY_NVCC_FLAGS` 折进 cubin key，并带 toolkit 摘要
  （:117-126）；PTX 需要自己的 key
- `-arch=sm_NN[a]`（`nvcc.py:154`；`device_arch` :72，`_launchable_arch` :85）—— `a` 后缀是 TMA 和
  block-scaled fp4 必需的，`.target` 必须对得上
- `arch_specific` 是从**源码文本**推断的（`plan.py:211`）→ 应该改成从 IR 推
- `cuda_includes`（:1579）拉 `<cuda_fp16.h>`/`<cuda_fp8.h>`；`__half` / `__nv_fp8_e4m3` 在 PTX 里没有类型对应
- `#pragma unroll` / `EMMY_UNROLL` 是文本级，PTX 无对应物

## 对投标的启示（待题目验证）

- cubin 缓存按 (source, name, arch, toolkit, flags) 内容寻址，PTX 发射器**不能复用**它
- `kernel_attributes`（`program.py:105`）从 cubin 读寄存器/smem 量，任何上报路径今天都是 cubin 形状
- **全量 PTX 后端 = 40+ 个节点类 + 张量核 + TMA/mbarrier 内联 PTX 全重写**，不是七天单人能做完的。
  合理题目形态应当是**限定子集**（例如只要求 elementwise/reduction 走 PTX，张量核仍走 CUDA C++），
  或者要求给现有管线加一条 `--emit ptx` 通路并证明与 NVRTC 产物语义一致。
  题目下来若要求全量重写，必须当天调低可行性判断。
