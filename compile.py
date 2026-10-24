#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX compiler.

    python compile.py path/to/kernel.ir.json > out.ptx

The IR is loaded through emmy's own classes (``Graph.from_dict``); everything
downstream of that -- the PTX text -- is emitted here.
"""
from __future__ import annotations

import json
import struct
import re
import sys
from pathlib import Path

from emmy.compiler.graph import Graph

# --------------------------------------------------------------------------
# dtypes

MUL_RE = re.compile(r"^\tmul\.(lo\.s32|wide\.s32)\s+(%[a-z]+\d+),\s*(%[a-z]+\d+),\s*(-?\d+);")
ADD_RE = re.compile(r"^\t(add\.s32|add\.s64)\s+(%[a-z]+\d+),\s*(%[a-z]+\d+),\s*(%[a-z]+\d+);")
DEST_RE = re.compile(r"^\t[\w.]+\s+(%[a-z]+\d+)\s*,")
SRC_RE = re.compile(r"%[a-z]+\d+")

FLOAT = {"f16": ("f16", 2), "f32": ("f32", 4), "f64": ("f64", 8), "bf16": ("bf16", 2)}
INT = {"int": ("s32", 4), "i32": ("s32", 4), "i64": ("s64", 8), "u32": ("u32", 4), "u64": ("u64", 8)}


def dt_name(dt) -> str:
    n = getattr(dt, "name", None)
    if n is None:
        n = str(dt)
    return {
        "float32": "f32", "float16": "f16", "float64": "f64", "int32": "i32", "int64": "i64",
        "float": "f32", "double": "f64", "half": "f16", "int": "i32",
    }.get(n, n)


def dt_bytes(dt) -> int:
    n = dt_name(dt)
    if n in FLOAT:
        return FLOAT[n][1]
    if n in INT:
        return INT[n][1]
    raise SystemExit(f"unsupported dtype {n!r}")


def dim_int(d) -> int:
    v = getattr(d, "value", d)
    return int(v)


def shape_of(v) -> tuple[int, ...]:
    return tuple(dim_int(x) for x in getattr(v, "shape", ()))


# --------------------------------------------------------------------------
# elementwise ops -> PTX

BINARY_F = {
    "add": "add.f32",
    "subtract": "sub.f32",
    "multiply": "mul.f32",
    "divide": "div.rn.f32",
    "maximum": "max.f32",
    "minimum": "min.f32",
}

# numpy ufunc names that produce a predicate (``setp``), not a value.
COMPARISONS = {
    "greater": "gt",
    "less": "lt",
    "greater_equal": "ge",
    "less_equal": "le",
    "equal": "eq",
    "not_equal": "ne",
}

EXPR_I = {
    "+": "add.s32",
    "-": "sub.s32",
    "*": "mul.lo.s32",
    "/": "div.s32",
    "%": "rem.s32",
    "&": "and.b32",
    "|": "or.b32",
    "^": "xor.b32",
    "<<": "shl.b32",
    ">>": "shr.s32",
}

EXPR_F = {
    "+": "add.f32",
    "-": "sub.f32",
    "*": "mul.f32",
    "/": "div.rn.f32",
}

BINARY_I = {
    "add": "add.s32",
    "subtract": "sub.s32",
    "multiply": "mul.lo.s32",
    "divide": "div.s32",
    "modulo": "rem.s32",
    "maximum": "max.s32",
    "minimum": "min.s32",
}

ARITH_OP = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "rem",
    "&": "and",
    "|": "or",
    "<<": "shl",
    ">>": "shr",
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "modulo": "rem",
}

UNARY_F = {
    "relu": None,  # handled specially: max(x, 0)
    "neg": "neg.f32",
    "negative": "neg.f32",
    "absolute": "abs.f32",
    "abs": "abs.f32",
    "fabs": "abs.f32",
    "sqrt": "sqrt.rn.f32",
    "rsqrt": "rsqrt.approx.f32",
    "reciprocal": "rcp.rn.f32",
    "sin": "sin.approx.f32",
    "cos": "cos.approx.f32",
    "floor": "cvt.rmi.f32.f32",
    "ceil": "cvt.rpi.f32.f32",
    "round": "cvt.rni.f32.f32",
    "rint": "cvt.rni.f32.f32",
    "trunc": "cvt.rzi.f32.f32",
}

LW = 1.4426950408889634  # log2(e)
LN2 = 0.6931471805599453


class Emitter:
    def __init__(self, graph: Graph, kernel_node, prefix: str = ""):
        self.g = graph
        self.node = kernel_node
        self.op = kernel_node.op
        self.prefix = prefix  # namespaces module-level shared symbols in multi-kernel modules
        self.n_r = self.n_f = self.n_p = self.n_rd = self.n_rs = 0
        self.body: list[str] = []
        self.syms: dict[str, list[str]] = {}  # name -> [reg, ptx type]
        self.ptr: dict[str, str] = {}  # buffer -> u64 base register
        self.addr_of_sym: dict[str, str] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self.smem: dict[str, tuple[tuple[int, ...], str, int]] = {}
        self.labels = 0
        self.params: list[tuple[str, str]] = []  # (name, dtype)
        # redundant-arithmetic elimination: value key -> register, register -> loop deps
        self.cse: dict[tuple, str] = {}
        self.dep: dict[str, frozenset[str]] = {}
        # loop-invariant reuse: one cache per enclosing loop, plus the seq of the last
        # register defined before that loop started
        self.reg_seq: dict[str, int] = {}
        self.seq_n: int = 0
        self.mutated: set[str] = set()
        self.loop_cache: list[dict] = []
        self.loop_floor: list[int] = []
        # per-loop stack of body indices holding arithmetic that can move in front of the loop,
        # and the finished (header index, indices) lifts waiting to be applied
        self.hoist: list[list[tuple[int, str]]] = []
        self.lifts: list[tuple[int, list[int]]] = []
        self.defined: dict[str, int] = {}
        # registers known to be non-negative, and registers holding a known constant
        self.nn: dict[str, bool] = {}
        self.konst: dict[str, int] = {}

    # ---- register bank ---------------------------------------------------
    def emit(self, line: str) -> None:
        self.body.append(line if line.endswith(":") else "\t" + line)
        hit = DEST_RE.match(self.body[-1])
        if hit is not None:
            self.defined[hit.group(1)] = len(self.body) - 1

    def r(self) -> str:
        self.n_r += 1
        return f"%r{self.n_r}"

    def f(self) -> str:
        self.n_f += 1
        return f"%f{self.n_f}"

    def p(self) -> str:
        self.n_p += 1
        return f"%p{self.n_p}"

    def rd(self) -> str:
        self.n_rd += 1
        return f"%rd{self.n_rd}"

    def rs(self) -> str:
        """16-bit register: half/bfloat16 values travel through it on the way to and from memory."""
        self.n_rs += 1
        return f"%rs{self.n_rs}"

    def label(self, tag: str = "") -> str:
        self.labels += 1
        return f"L{self.labels}{tag}"

    # ---- integer arithmetic: folding, immediates, common subexpressions ---
    @staticmethod
    def _imm(x) -> int | None:
        """The integer value of an operand, or None if it lives in a register."""
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            try:
                return int(x, 0)
            except ValueError:
                return None
        return None

    def deps(self, x) -> frozenset[str]:
        return self.dep.get(x, frozenset()) if isinstance(x, str) else frozenset()

    def prune(self, carried: set[str]) -> None:
        """Drop cached values that depended on a loop counter (they change every iteration).

        Values that do *not* depend on the counter survive the back edge: they were computed
        once during the first iteration and their register still holds that value, which is
        exactly loop-invariant code motion for free.
        """
        self.cse = {k: v for k, v in self.cse.items() if not (self.dep.get(v, frozenset()) & carried)}

    def arith(self, op: str, a, b, t: str = "s32"):
        """Emit one 32-bit integer op.  Returns a register name, or an int when folded."""
        ia, ib = self._imm(a), self._imm(b)
        if ia is not None and ib is not None:
            folded = {
                "add": lambda: ia + ib,
                "sub": lambda: ia - ib,
                "mul": lambda: ia * ib,
                "div": (lambda: int(ia / ib)) if ib else None,
                "rem": (lambda: ia - int(ia / ib) * ib) if ib else None,
                "and": lambda: ia & ib,
                "or": lambda: ia | ib,
                "shl": lambda: ia << ib,
                "shr": lambda: ia >> ib,
            }.get(op)
            if folded is not None:
                return folded()
        if op == "add":
            if ia == 0:
                return b
            if ib == 0:
                return a
            if ia is not None:  # commutative: keep the immediate on the right
                a, b = b, a
        elif op == "mul":
            if ia == 0 or ib == 0:
                return 0
            if ia == 1:
                return b
            if ib == 1:
                return a
            if ia is not None:
                a, b = b, a
        elif op == "sub" and ib == 0:
            return a
        elif op in ("div", "rem") and ib == 1:
            return a if op == "div" else 0

        if self._imm(a) is not None:  # a must be a register for the remaining ops
            a = self.force(a)
        # Reuse is only sound for values that cannot change inside the loop we are in: both
        # operands must have been defined before this loop started and neither may be mutated
        # in place (loop counters are).  Anything else is recomputed, because a cache that
        # ignores this silent-miscompiles a GEMM.
        key = (op,) + tuple(sorted((str(a), str(b)))) if op in ("add", "mul", "and", "or") else (op, a, b)
        hit = self.cse.get(key)
        if hit is not None:
            return hit
        loop_key = None
        if self.loop_cache and self.loop_floor and op in ("add", "sub", "mul", "div", "rem", "and", "or", "shl", "shr"):
            floor = self.loop_floor[-1]
            if self._invariant(a, floor) and self._invariant(b, floor):
                loop_key = key
                hit = self.loop_cache[-1].get(loop_key)
                if hit is not None:
                    return hit
        suffix = {"add": "add.s32", "sub": "sub.s32", "mul": "mul.lo.s32", "div": "div.s32",
                  "rem": "rem.s32", "and": "and.b32", "or": "or.b32",
                  "shl": "shl.b32", "shr": "shr.s32"}[op]
        out = self.p() if t == "pred" else self.r()
        self.emit(f"{suffix} {out}, {a}, {b};")
        self.dep[out] = self.deps(a) | self.deps(b)
        self.seq_n += 1
        self.reg_seq[out] = self.seq_n
        self.cse[key] = out
        if loop_key is not None and self.loop_cache:
            self.loop_cache[-1][loop_key] = out
        if self.hoist:
            self.hoist[-1].append((len(self.body) - 1, out))
        return out

    def _loop_enter(self, counter: str) -> None:
        """Start an invariance scope: only values defined before here may be reused."""
        self.loop_cache.append({})
        self.loop_floor.append(self.seq_n + 1)
        self.hoist.append([])
        self.mutated.add(counter)

    def _hoist_out(self, mark: int) -> None:
        """Queue the counter-independent arithmetic recorded since ``mark`` for lifting.

        ``dep`` records which loop counters a value was built from, so an empty set means the
        value cannot change while this loop runs.  Deciding that at the end - rather than while
        the body is being emitted - lets a constant that was itself lifted make its consumers
        liftable too.
        """
        scope = self.hoist.pop()
        cand = {i: reg for i, reg in scope if not self.dep.get(reg)}
        lifted: set[int] = set()
        for i, reg in sorted(cand.items()):
            if self._operands_live(i, mark, lifted):
                lifted.add(i)
        if lifted:
            self.lifts.append((mark, sorted(lifted)))

    def _operands_live(self, index: int, mark: int, lifted: set) -> bool:
        """True when every source register of ``self.body[index]`` is available at ``mark``."""
        srcs = SRC_RE.findall(self.body[index])
        for s in srcs[1:] if srcs else []:
            home = self.defined.get(s)
            if home is None:
                return False  # defined by a non-arithmetic emitter: leave it alone
            if home < mark or home in lifted:
                continue
            return False
        return True

    def _fuse_madd(self) -> None:
        """Rewrite ``t = a * k`` + ``d = t + b`` as ``mad d, a, k, b`` when ``t`` has one use.

        Both forms are exact for 32-bit wraparound, so this only removes an instruction; the
        temporary has to be single-use or the rewrite would duplicate work.
        """
        uses: dict[str, int] = {}
        defs: dict[str, int] = {}
        for i, ln in enumerate(self.body):
            hit = MUL_RE.match(ln)
            if hit is not None:
                defs[hit.group(2)] = i
            regs = SRC_RE.findall(ln)
            head = DEST_RE.match(ln)
            if head is not None and regs and regs[0] == head.group(1):
                regs = regs[1:]  # its own destination is a definition, not a use
            for reg in regs:
                uses[reg] = uses.get(reg, 0) + 1
        dead: set[int] = set()
        for i, ln in enumerate(self.body):
            hit = ADD_RE.match(ln)
            if hit is None:
                continue
            dst, srcs = hit.group(2), hit.groups()[2:]
            for j, reg in enumerate(srcs):
                home = defs.get(reg)
                if home is None or uses.get(reg) != 1:
                    continue
                # A multiply sitting outside a loop still dominates uses inside it, but one
                # inside a conditional does not dominate what follows the branch, so only the
                # region-closing labels block the rewrite.
                if any(ln2.endswith(("_else:", "_end:", "_endif:")) for ln2 in self.body[home:i]):
                    continue
                mk = MUL_RE.match(self.body[home])
                width, _dst, lhs, imm = mk.group(1), mk.group(2), mk.group(3), mk.group(4)
                if width != ("wide.s32" if ".s64" in ln else "lo.s32"):
                    continue
                other = srcs[1 - j]
                if other == reg:
                    continue
                self.body[i] = f"\tmad.{width} {dst}, {lhs}, {imm}, {other};"
                dead.add(home)
                break
        if dead:
            self.body = [ln for i, ln in enumerate(self.body) if i not in dead]

    def _apply_lifts(self) -> None:
        """Apply every lift, largest body index first so earlier indices stay valid."""
        if not self.lifts:
            return
        for mark, idxs in sorted(self.lifts, key=lambda item: -item[0]):
            lifted = [self.body[i] for i in idxs]
            drop = set(idxs)
            self.body = [ln for i, ln in enumerate(self.body) if i not in drop]
            self.body[mark:mark] = lifted
        self.lifts.clear()

    def _loop_exit(self) -> None:
        self.loop_cache.pop()
        self.loop_floor.pop()
        # the back edge mutates the counter, so nothing cached inside the body survives it
        self.cse.clear()

    def _invariant(self, x, floor: int) -> bool:
        """True when the value is loop-invariant for the loop whose entry seq is ``floor``."""
        if self._imm(x) is not None:
            return True
        if x in self.mutated:
            return False
        return self.reg_seq.get(x, 1 << 30) < floor

    def force(self, x) -> str:
        """Materialise an immediate into a register (only when an instruction needs one)."""
        v = self._imm(x)
        if v is None:
            return x
        key = ("const", v)
        hit = self.cse.get(key)
        if hit is not None:
            return hit
        out = self.r()
        self.emit(f"mov.s32 {out}, {v};")
        self.cse[key] = out
        self.dep[out] = frozenset()
        self.seq_n += 1
        self.reg_seq[out] = self.seq_n
        if self.hoist:
            self.hoist[-1].append((len(self.body) - 1, out))
        return out

    # ---- pool of registers for a dtype -----------------------------------
    def new_of(self, t: str) -> str:
        if t in ("f16", "f32", "f64", "bf16"):
            return self.f()
        if t in ("s32", "u32"):
            return self.r()
        if t in ("s64", "u64"):
            return self.rd()
        if t == "pred":
            return self.p()
        raise SystemExit(f"no register class for {t}")

    # ---- setup -----------------------------------------------------------
    def prepare(self) -> None:
        """Collect shapes, dtypes and the parameter list."""
        for name, node in self.g.nodes.items():
            for out in getattr(node, "outputs", ()) or ():
                nm = getattr(out, "name", None)
                if nm is None:
                    continue
                self.shapes[nm] = shape_of(out)
                self.dtypes[nm] = dt_name(out.dtype)
        order = list(self.node.inputs) + [v.name for v in self.node.outputs]
        meta = self.meta()
        if meta is not None:
            order = list(meta.get("arg_order", order))
        self.params = [(n, "u64") for n in order]
        self.collect_smem(self.op.body)

    def collect_smem(self, body) -> None:
        """Find every static shared buffer up front, so bases can be hoisted."""
        for st in body or ():
            if type(st).__name__ == "Smem":
                self.smem[st.name] = (
                    tuple(dim_int(x) for x in st.extents),
                    dt_name(st.dtype),
                    getattr(st, "align", 0) or 0,
                )
            for attr in ("body", "else_body"):
                sub = getattr(st, attr, None)
                if sub:
                    self.collect_smem(sub)

    def smem_layout(self):
        """Ordered (sym, n_elems, ptx_t, align, byte_offset) plus the total byte size."""
        out = []
        off = 0
        for sym, (ext, t, align) in self.smem.items():
            n = 1
            for e in ext:
                n *= e
            al = align or dt_bytes_for(t)
            off = (off + al - 1) // al * al
            ptx_t = "f32" if t == "f32" else {"f16": "f16", "f64": "f64"}.get(t, "s32")
            out.append((sym, n, ptx_t, al, off))
            off += n * dt_bytes_for(t)
        return out, off

    def sym_ref(self, sym: str) -> str:
        """Module-level name of a shared symbol (namespaced when one module holds several kernels)."""
        return f"{self.prefix}{sym}"

    def meta(self):
        p = getattr(self, "ir_path", None)
        if p is None:
            return None
        mpath = Path(p).parent / "meta.json"
        if not mpath.exists():
            return None
        try:
            meta = json.loads(mpath.read_text())
        except Exception:
            return None
        for k in meta.get("kernels", []):
            if k.get("entry") == getattr(self.op, "name", None):
                return k
        return None

    # ---- prologue --------------------------------------------------------
    def prologue(self) -> None:
        for i, (name, _t) in enumerate(self.params, start=1):
            base = self.rd()
            self.addr_of_sym[name] = base
            self.emit(f"ld.param.u64 {base}, [p_{name}];")
            self.emit(f"cvta.to.global.u64 {base}, {base};")
        tid = self.r()
        self.emit(f"mov.u32 {tid}, %tid.x;")
        ctaid = self.r()
        self.emit(f"mov.u32 {ctaid}, %ctaid.x;")
        ntid = self.r()
        self.emit(f"mov.u32 {ntid}, %ntid.x;")
        gid = self.r()
        self.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid};")
        self.gid = gid
        self.tid = tid
        lane = self.r()
        self.emit(f"and.b32 {lane}, {tid}, 31;")
        warp = self.r()
        self.emit(f"shr.u32 {warp}, {tid}, 5;")
        self.syms["lane"] = [lane, "s32"]
        self.syms["warp"] = [warp, "s32"]
        # Shared bases are hoisted: a base first materialised inside a divergent branch would
        # be undefined on the threads that skipped it.  Buffers above the 48 KiB static cap
        # live in one dynamic pool, exactly like the extern __shared__ block of the reference.
        layout, total = self.smem_layout()
        self.dyn_smem = total > 48 * 1024
        for sym, _n, _t, _al, off in layout:
            base = self.r()
            self.addr_of_sym["__smem_" + sym] = base
            if not self.dyn_smem:
                self.emit(f"mov.u32 {base}, {self.sym_ref(sym)};")
            else:
                self.emit(f"mov.u32 {base}, {self.sym_ref('_smem_pool')};")
                if off:
                    self.emit(f"add.u32 {base}, {base}, {off};")

    # ---- address helpers -------------------------------------------------
    def as_int(self, e) -> str:
        """An integer operand: an immediate when the expression is constant, else a register."""
        if isinstance(e, int) and not isinstance(e, bool):
            return str(e)
        if type(e).__name__ == "Dim":
            return str(dim_int(e))
        if isinstance(e, str):
            if e.startswith("%"):
                return e
            if e in self.syms:
                return self.syms[e][0]
        reg, t = self.expr(e)
        return reg

    def linear(self, buf: str, index) -> str:
        """Row-major element offset of a coordinate list."""
        shape = self.shapes[buf] if buf in self.shapes else tuple(dim_int(x) for x in self.smem[buf][0])
        parts = list(index)
        if not parts:
            return "0"
        if len(parts) == len(shape) and len(shape) > 1:
            acc = self.as_int(parts[0])
            for i in range(1, len(parts)):
                if shape[i] == 1 and i == len(parts) - 1:
                    continue
                acc = self.arith("mul", acc, shape[i])
                acc = self.arith("add", acc, self.as_int(parts[i]))
            return acc
        if len(parts) == 1:
            return self.as_int(parts[0])
        # fall back: flat sum assuming the leading index is already linear
        acc = self.as_int(parts[0])
        for p in parts[1:]:
            acc = self.arith("mul", acc, shape[-1])
            acc = self.arith("add", acc, self.as_int(p))
        return acc

    def byte_offset(self, lin: str, nbytes: int) -> str:
        v = self._imm(lin)
        if v is not None:
            return str(v * nbytes)
        if nbytes == 1:
            return lin
        key = ("widen", lin, nbytes)
        hit = self.cse.get(key)
        if hit is not None:
            return hit
        off = self.rd()
        self.emit(f"mul.wide.s32 {off}, {lin}, {nbytes};")
        self.cse[key] = off
        self.dep[off] = self.deps(lin)
        return off

    def global_addr(self, buf: str, index) -> str:
        lin = self.linear(buf, index)
        nbytes = dt_bytes_for(self.dtypes.get(buf, "f32"))
        base = self.addr_of_sym[buf]
        v = self._imm(lin)
        if v is not None:
            if v == 0:
                return base
            key = ("gaddr", base, v * nbytes)
            hit = self.cse.get(key)
            if hit is not None:
                return hit
            a = self.rd()
            self.emit(f"add.s64 {a}, {base}, {v * nbytes};")
            self.cse[key] = a
            self.dep[a] = frozenset()
            return a
        key = ("gaddr", base, lin, nbytes)
        hit = self.cse.get(key)
        if hit is not None:
            return hit
        a = self.rd()
        if nbytes == 1:
            off = self.byte_offset(lin, nbytes)
            self.emit(f"add.s64 {a}, {base}, {off};")
        else:
            # one instruction: 64-bit address = 32-bit index * element size + base
            self.emit(f"mad.wide.s32 {a}, {lin}, {nbytes}, {base};")
        self.cse[key] = a
        self.dep[a] = self.deps(lin)
        return a

    def shared_addr(self, buf: str, index) -> str:
        ext, t, _align = self.smem[buf]
        lin = self.linear(buf, index)
        nbytes = dt_bytes_for(t)
        off = lin if nbytes == 1 else self.arith("mul", lin, nbytes)
        return self.arith("add", self.shared_base(buf), off)

    def mul_const(self, reg: str, k: int) -> str:
        return self.arith("mul", reg, k)

    def shared_base(self, buf: str) -> str:
        """32-bit shared-space base of a static shared symbol (like the worked example)."""
        key = "__smem_" + buf
        if key not in self.addr_of_sym:
            reg = self.r()
            self.emit(f"mov.u32 {reg}, {self.sym_ref(buf)};")
            self.addr_of_sym[key] = reg
        return self.addr_of_sym[key]

    # ---- expressions -----------------------------------------------------
    def expr(self, e):
        """Return (register, ptx type)."""
        cls = type(e).__name__
        if cls == "Var":
            name = e.name
            if name in self.syms:
                return self.syms[name]
            raise SystemExit(f"unknown variable {name!r}")
        if cls == "Literal":
            return self.literal(e)
        if cls == "Dim":
            return [str(dim_int(e)), "s32"]
        if isinstance(e, int):
            return [str(e), "s32"]
        if isinstance(e, float):
            return self.f_literal(e)
        if cls == "BinaryExpr":
            return self.binary(e)
        if cls == "TernaryExpr":
            return self.ternary(e)
        if cls == "FlatIndex":
            buf = getattr(e, "buffer", None)
            lin = self.linear(buf, e.index)
            return [lin if self._imm(lin) is None else self.force(lin), "s32"]
        if cls == "CastExpr":
            reg, t = self.expr(e.expr)
            want = dt_name(e.dtype)
            return [self.cast(reg, t, want), want]
        if cls == "Builtin":
            nm = getattr(e, "name", "")
            if nm in ("threadIdx.x", "tid.x"):
                return [self.tid, "s32"]
            if nm == "blockIdx.x":
                return [self.syms.get("__ctaid", [self.ctaid_reg])[0], "s32"]
            raise SystemExit(f"unsupported builtin {nm!r}")
        if cls == "FuncCallExpr":
            return self.func_call(e)
        raise SystemExit(f"unsupported expression {cls}")

    def ctaid_reg(self):  # pragma: no cover - defensive
        return self.gid

    def literal(self, e):
        v = e.value
        t = dt_name(getattr(e, "dtype", "int"))
        if t in FLOAT:
            reg = self.f()
            bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
            self.emit(f"mov.b32 {reg}, 0f{bits:08X};")
            return [reg, FLOAT[t][0]]
        reg = self.force(int(v))
        return [reg, "s32"]

    def is_float(self, t: str) -> bool:
        return t in ("f16", "f32", "f64")

    def f_literal(self, v: float):
        reg = self.f()
        bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        self.emit(f"mov.b32 {reg}, 0f{bits:08X};")
        return [reg, "f32"]

    def binary(self, e):
        op = e.op
        lt = self.expr(e.left)
        rt = self.expr(e.right)
        if self.is_float(lt[1]) or self.is_float(rt[1]):
            t = lt[1] if self.is_float(lt[1]) else rt[1]
            if op in EXPR_F:
                out = self.f()
                self.emit(f"{EXPR_F[op]} {out}, {self.as_type(lt, t)}, {self.as_type(rt, t)};")
                return [out, t]
            if op in BINARY_F:
                out = self.f()
                self.emit(f"{BINARY_F[op]} {out}, {self.as_type(lt, t)}, {self.as_type(rt, t)};")
                return [out, t]
            if op in ("==", "!=", "<", "<=", ">", ">="):
                pr = self.p()
                m = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op]
                self.emit(f"setp.{m}.f32 {pr}, {self.as_type(lt, t)}, {self.as_type(rt, t)};")
                return [pr, "pred"]
            raise SystemExit(f"unsupported float binary op {op!r}")
        if op in EXPR_I:
            aop = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "rem", "&": "and",
                   "|": "or", "<<": "shl", ">>": "shr"}.get(op)
            if aop is None:
                out = self.r()
                self.emit(f"{EXPR_I[op]} {out}, {lt[0]}, {rt[0]};")
                return [out, "s32"]
            r = self.arith(aop, lt[0], rt[0])
            if self._imm(r) is not None:
                r = self.force(r)
            return [r, "s32"]
        aop = ARITH_OP.get(op)
        if aop is not None:
            r = self.arith(aop, lt[0], rt[0])
            if self._imm(r) is not None:
                r = self.force(r)
            return [r, "s32"]
        if op in BINARY_I:
            out = self.r()
            self.emit(f"{BINARY_I[op]} {out}, {lt[0]}, {rt[0]};")
            return [out, "s32"]
        if op in ("==", "!=", "<", "<=", ">", ">="):
            pr = self.p()
            m = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op]
            self.emit(f"setp.{m}.s32 {pr}, {lt[0]}, {rt[0]};")
            return [pr, "pred"]
        if op in ("&&", "and"):
            pr = self.p()
            self.emit(f"and.pred {pr}, {lt[0]}, {rt[0]};")
            return [pr, "pred"]
        if op in ("||", "or"):
            pr = self.p()
            self.emit(f"or.pred {pr}, {lt[0]}, {rt[0]};")
            return [pr, "pred"]
        raise SystemExit(f"unsupported int binary op {op!r}")

    def as_type(self, val, want: str) -> str:
        reg, t = val
        if t == want:
            return reg
        return self.cast(reg, t, want)

    def cast(self, reg: str, have: str, want: str) -> str:
        if have == want:
            return reg
        # a predicate bound to a bool-typed SSA name stays a predicate: every consumer
        # (``where``/``Select``/``Cond``) wants it that way, and materialising 0/1 would
        # only add a selp per use.
        if have == "pred" and want in ("bool", "u8", "s8", "s32", "u32"):
            return reg
        if have in ("bool", "u8") and want == "pred":
            pr = self.p()
            self.emit(f"setp.ne.s32 {pr}, {reg}, 0;")
            return pr
        # 16-bit floats stay in f32 registers here; memory stores narrow them once,
        # so rounding to f16/bf16 mid-expression would only lose accuracy.
        if want in ("f16", "bf16") and self.is_float(have):
            return reg
        if self.is_float(have) and self.is_float(want):
            out = self.new_of(want)
            self.emit(f"cvt.{want}.{have} {out}, {reg};")
            return out
        if have in ("s32", "u32") and want in ("s32", "u32", "s64", "u64"):
            out = self.new_of(want)
            self.emit(f"cvt.{want}.{have} {out}, {reg};")
            return out
        raise SystemExit(f"unsupported cast {have} -> {want}")

    def ternary(self, e):
        cond = self.expr(e.cond)
        a = self.expr(e.if_true)
        b = self.expr(e.if_false)
        t = a[1] if self.is_float(a[1]) else b[1]
        out = self.new_of(t)
        ins = "selp.f32" if self.is_float(t) else "selp.b32"
        self.emit(f"{ins} {out}, {self.as_type(a, t)}, {self.as_type(b, t)}, {cond[0]};")
        return [out, t]

    def func_call(self, e):
        nm = getattr(e, "name", "")
        args = [self.expr(a) for a in getattr(e, "args", ())]
        return self.elementwise(nm, args)

    # ---- elementwise -----------------------------------------------------
    def elementwise(self, name: str, args):
        # the first operand is often the predicate of ``where``; the value type comes
        # from the data operands.
        t = next((a[1] for a in args if a[1] not in ("pred", "bool")),
                 args[0][1] if args else "f32")
        if name in COMPARISONS and len(args) == 2:
            pr = self.p()
            sfx = "f32" if self.is_float(t) else "s32"
            lt, rt = args[0], args[1]
            self.emit(f"setp.{COMPARISONS[name]}.{sfx} {pr}, {self.as_type(lt, t)}, {self.as_type(rt, t)};")
            return [pr, "pred"]
        if name in ("logical_and", "logical_or") and len(args) == 2:
            pr = self.p()
            lo = args[0][0] if args[0][1] == "pred" else self._pred(args[0][0], args[0][1])
            ro = args[1][0] if args[1][1] == "pred" else self._pred(args[1][0], args[1][1])
            self.emit(f"{'and' if name == 'logical_and' else 'or'}.pred {pr}, {lo}, {ro};")
            return [pr, "pred"]
        if name == "logical_not" and len(args) == 1:
            pr = self.p()
            lo = args[0][0] if args[0][1] == "pred" else self._pred(args[0][0], args[0][1])
            self.emit(f"not.pred {pr}, {lo};")
            return [pr, "pred"]
        if name in ("isnan", "isinf") and len(args) == 1:
            pr = self.p()
            self.emit(f"testp.{'nan' if name == 'isnan' else 'inf'}.f32 {pr}, {args[0][0]};")
            return [pr, "pred"]
        if name in ("copy", "bitcast") and len(args) == 1:
            out = self.new_of(t)
            self.emit(f"mov.b32 {out}, {args[0][0]};")
            return [out, t]
        if not self.is_float(t):
            if name in BINARY_I and len(args) == 2:
                out = self.r()
                self.emit(f"{BINARY_I[name]} {out}, {args[0][0]}, {args[1][0]};")
                return [out, "s32"]
            raise SystemExit(f"unsupported int elementwise {name!r}")
        a = [x[0] for x in args]
        if name in BINARY_F and len(a) == 2:
            out = self.f()
            self.emit(f"{BINARY_F[name]} {out}, {a[0]}, {a[1]};")
            return [out, t]
        if name == "relu" and len(a) == 1:
            return [self._bin("max.f32", a[0], self._const(0.0)), t]
        if name in UNARY_F and UNARY_F[name] and len(a) == 1:
            out = self.f()
            self.emit(f"{UNARY_F[name]} {out}, {a[0]};")
            return [out, t]
        if name in ("where", "select") and len(a) == 3:
            cond = args[0]
            pr = cond[0] if cond[1] == "pred" else self._pred(cond[0], cond[1])
            out = self._selp(a[1], a[2], pr)
            return [out, t]
        if len(a) == 1:
            one = {
                "exp": self._exp,
                "exp_fast": self._exp,
                "exp2": self._ex2,
                "log": lambda x: self._bin("mul.f32", self._lg2(x), self._const(LN2)),
                "ln": lambda x: self._bin("mul.f32", self._lg2(x), self._const(LN2)),
                "log2": self._lg2,
                "log10": lambda x: self._bin("mul.f32", self._lg2(x), self._const(0.30102999566398120)),
                "tanh": self._tanh,
                "sigmoid": self._sigmoid,
                "silu": self._silu,
                "softplus": self._softplus,
                "gelu": self._gelu,
                "gelu_tanh": self._gelu_tanh,
                "erf": self._erf,
                "sign": self._sign,
            }.get(name)
            if one is not None:
                return [one(a[0]), t]
        if len(a) == 2:
            two = {
                "power": self._pow,
                "pow": self._pow,
                "mod": self._fmod,
                "remainder": self._fmod,
                "fmod": self._fmod,
                "floor_divide": self._floor_div,
            }.get(name)
            if two is not None:
                return [two(a[0], a[1]), t]
        raise SystemExit(f"unsupported elementwise op {name!r}")

    # ---- instruction helpers --------------------------------------------
    def _const(self, v: float) -> str:
        reg = self.f()
        bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        self.emit(f"mov.b32 {reg}, 0f{bits:08X};")
        return reg

    def _bin(self, op: str, x: str, y: str) -> str:
        out = self.f()
        self.emit(f"{op} {out}, {x}, {y};")
        return out

    def _fma(self, x: str, y: str, z: str) -> str:
        out = self.f()
        self.emit(f"fma.rn.f32 {out}, {x}, {y}, {z};")
        return out

    def _pred(self, reg: str, t: str) -> str:
        """A predicate valid for ``reg`` when the IR hands us a 0/1 value instead."""
        if t == "pred":
            return reg
        pr = self.p()
        sfx = "f32" if self.is_float(t) else "s32"
        self.emit(f"setp.ne.{sfx} {pr}, {reg}, 0;")
        return pr

    def _ex2(self, x: str) -> str:
        return self._un("ex2.approx.f32", x)

    def _un(self, op: str, x: str) -> str:
        out = self.f()
        self.emit(f"{op} {out}, {x};")
        return out

    def _lg2(self, x: str) -> str:
        return self._un("lg2.approx.f32", x)

    def _exp(self, x: str) -> str:
        return self._un("ex2.approx.f32", self._bin("mul.f32", x, self._const(LW)))

    def _rcp(self, x: str) -> str:
        return self._un("rcp.rn.f32", x)

    def _abs(self, x: str) -> str:
        return self._un("abs.f32", x)

    def _sigmoid(self, x: str) -> str:
        # 1 / (1 + 2^(-x*log2(e)))
        neg = self._bin("mul.f32", x, self._const(-LW))
        return self._rcp(self._bin("add.f32", self._ex2(neg), self._const(1.0)))

    def _tanh(self, x: str) -> str:
        # 1 - 2 / (2^(2x*log2(e)) + 1)
        e2 = self._ex2(self._bin("mul.f32", x, self._const(2.0 * LW)))
        den = self._bin("add.f32", e2, self._const(1.0))
        q = self._bin("div.rn.f32", self._const(2.0), den)
        return self._bin("sub.f32", self._const(1.0), q)

    def _erf(self, x: str) -> str:
        """Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, which is under an f32 ulp at 1.0."""
        ax = self._abs(x)
        t = self._rcp(self._fma(self._const(0.3275911), ax, self._const(1.0)))
        poly = self._const(1.061405429)
        for c in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
            poly = self._fma(poly, t, self._const(c))
        # poly is sum(a_i * t^(i-1)); the series needs one more factor of t
        series = self._bin("mul.f32", poly, t)
        ax2 = self._bin("mul.f32", ax, ax)
        tail = self._bin("mul.f32", series,
                         self._exp(self._bin("mul.f32", ax2, self._const(-1.0))))
        y = self._bin("sub.f32", self._const(1.0), tail)
        # restore the sign bit without a branch: erf(0) keeps +0
        out = self.f()
        neg = self.r()
        self.emit(f"and.b32 {neg}, {x}, 0f80000000;")
        self.emit(f"or.b32 {out}, {y}, {neg};")
        return out

    def _silu(self, x: str) -> str:
        return self._bin("mul.f32", x, self._sigmoid(x))

    def _softplus(self, x: str) -> str:
        # max(x,0) + log(1 + exp(-|x|))
        ax = self._abs(x)
        e = self._exp(self._bin("mul.f32", ax, self._const(-1.0)))
        lg = self._bin("mul.f32", self._lg2(self._bin("add.f32", e, self._const(1.0))), self._const(LN2))
        return self._bin("add.f32", self._bin("max.f32", x, self._const(0.0)), lg)

    def _gelu(self, x: str) -> str:
        inner = self._erf(self._bin("mul.f32", x, self._const(0.7071067811865476)))
        return self._bin("mul.f32", self._bin("mul.f32", x, self._const(0.5)),
                         self._bin("add.f32", inner, self._const(1.0)))

    def _gelu_tanh(self, x: str) -> str:
        x3 = self._bin("mul.f32", self._bin("mul.f32", x, x), x)
        inner = self._bin("mul.f32", self._const(0.7978845608028654),
                          self._bin("add.f32", x, self._bin("mul.f32", self._const(0.044715), x3)))
        return self._bin("mul.f32", self._bin("mul.f32", x, self._const(0.5)),
                         self._bin("add.f32", self._tanh(inner), self._const(1.0)))

    def _pow(self, x: str, y: str) -> str:
        return self._ex2(self._bin("mul.f32", y, self._lg2(x)))

    def _fmod(self, x: str, y: str) -> str:
        q = self._bin("div.rn.f32", x, y)
        tr = self._un("cvt.rzi.f32.f32", q)
        return self._bin("sub.f32", x, self._bin("mul.f32", tr, y))

    def _floor_div(self, x: str, y: str) -> str:
        return self._un("cvt.rmi.f32.f32", self._bin("div.rn.f32", x, y))

    def _sign(self, x: str) -> str:
        pos = self.p()
        nz = self.p()
        self.emit(f"setp.gt.f32 {pos}, {x}, 0f00000000;")
        self.emit(f"setp.ne.f32 {nz}, {x}, 0f00000000;")
        one = self._selp(self._const(1.0), self._const(-1.0), pos)
        return self._selp(one, self._const(0.0), nz)

    def _selp(self, x: str, y: str, pr: str) -> str:
        out = self.f()
        self.emit(f"selp.b32 {out}, {x}, {y}, {pr};")
        return out

    def transcendental(self, name: str, x: str) -> str:
        """Kept for callers that bypass :meth:`elementwise`."""
        table = {"exp": self._exp, "exp2": self._ex2, "log": lambda v: self._bin("mul.f32", self._lg2(v), self._const(LN2)),
                 "ln": lambda v: self._bin("mul.f32", self._lg2(v), self._const(LN2)),
                 "log2": self._lg2, "tanh": self._tanh, "sigmoid": self._sigmoid}
        if name not in table:
            raise SystemExit(f"unsupported transcendental {name!r}")
        return table[name](x)

    # ---- guard -----------------------------------------------------------
    def guard(self, n_cells: int, end_label: str) -> None:
        pr = self.p()
        self.emit(f"setp.lt.s32 {pr}, {self.gid}, {n_cells};")
        self.emit(f"@!{pr} bra {end_label};")
        # Threads that fail the guard never execute anything below, so a cached expression is
        # only reusable inside this region: keep the CSE scoped to it.
        self.cse.clear()
        if self.loop_cache:
            self.loop_cache[-1].clear()

    # ---- statements ------------------------------------------------------
    def stmts(self, body) -> None:
        for st in body:
            self.stmt(st)

    def stmt(self, st) -> None:
        fn = getattr(self, "st_" + type(st).__name__, None)
        if fn is None:
            raise SystemExit(f"unsupported statement {type(st).__name__}")
        fn(st)

    def st_Tile(self, st) -> None:
        axes = list(st.axes)
        extents = [dim_int(a.extent) for a in axes]
        total = 1
        for e in extents:
            total *= e
        end = self.label("_end")
        self.guard(total, end)
        # decode this thread's coordinate, row major with the last axis fastest
        rest = self.gid
        for i, ax in enumerate(axes):
            inner = 1
            for e in extents[i + 1:]:
                inner *= e
            if i == len(axes) - 1:
                reg = rest
            else:
                q = self.arith("div", rest, inner)
                reg = q
                rest = self.arith("sub", rest, self.arith("mul", q, inner))
            self.syms[ax.name] = [reg, "s32"]
        self.stmts(st.body)
        self.emit(f"{end}:")

    def st_Load(self, st) -> None:
        names = list(st.names)
        buf = st.input
        n = len(names)
        if getattr(st, "dtype", None) is not None:
            t = dt_name(st.dtype)
        elif buf in self.smem:
            t = self.smem[buf][1]
        else:
            t = self.dtypes.get(buf, "f32")
        ptx_t = FLOAT[t][0] if t in FLOAT else (INT[t][0] if t in INT else t)
        shared = buf in self.smem
        if shared:
            addr = self.shared_addr(buf, st.index)
        else:
            addr = self.global_addr(buf, st.index)
        space = "shared" if shared else "global"
        if t in ("f16", "bf16"):
            # every value we compute with is f32; 16-bit storage is widened on the way in
            for i, nm in enumerate(names):
                self.syms[nm] = [self.load_half(space, self.step(addr, i * 2), t), "f32"]
            return
        if n == 1:
            out = self.new_of(ptx_t)
            self.emit(f"ld.{space}.{ptx_t} {out}, [{addr}];")
            self.syms[names[0]] = [out, ptx_t]
            return
        if n not in (2, 4):
            raise SystemExit(f"unsupported vector load width {n}")
        regs = [self.new_of(ptx_t) for _ in names]
        self.emit(f"ld.{space}.v{n}.{ptx_t} {{{', '.join(regs)}}}, [{addr}];")
        for nm, reg in zip(names, regs):
            self.syms[nm] = [reg, ptx_t]

    def st_Assign(self, st) -> None:
        impl = st.op
        name = getattr(impl, "name", None) or str(impl)
        args = [self.value(a) for a in st.args]
        want = dt_name(st.dtype) if getattr(st, "dtype", None) is not None else None
        if want in ("bool", "bool8", "i1"):
            # a boolean SSA name is held as a predicate, never as a 0/1 register
            reg, got = self.elementwise(name, args)
            self.syms[st.name] = [self._pred(reg, got), "pred"]
            return
        ptx_t = None if want is None else (FLOAT[want][0] if want in FLOAT else (INT[want][0] if want in INT else want))
        reg, got = self.elementwise(name, args)
        if ptx_t is None:
            self.syms[st.name] = [reg, got]
            return
        if ptx_t in ("f16", "bf16") and self.is_float(got):
            # the value stays f32 in registers; the store narrows it exactly once
            self.syms[st.name] = [reg, "f32"]
            return
        if got != ptx_t and not (self.is_float(got) and self.is_float(ptx_t)):
            reg = self.cast(reg, got, ptx_t)
        self.syms[st.name] = [reg, ptx_t]

    def value(self, a):
        """An operand that may be an SSA name or an expression object."""
        if isinstance(a, str):
            if a in self.syms:
                return self.syms[a]
            if a in ("lane", "warp"):
                return self.syms[a]
            raise SystemExit(f"unknown SSA name {a!r}")
        return self.expr(a)

    # ---- 16-bit float traffic -------------------------------------------
    def step(self, addr: str, delta: int) -> str:
        """``addr`` shifted by ``delta`` bytes (zero means no instruction)."""
        if not delta:
            return addr
        if addr.startswith("%rd"):
            out = self.rd()
            self.emit(f"add.s64 {out}, {addr}, {delta};")
            return out
        return self.arith("add", addr, delta)

    def load_half(self, space: str, addr: str, t: str) -> str:
        rs = self.rs()
        self.emit(f"ld.{space}.b16 {rs}, [{addr}];")
        out = self.f()
        self.emit(f"{'cvt.f32.f16' if t == 'f16' else 'cvt.f32.bf16'} {out}, {rs};")
        return out

    def store_half(self, space: str, addr: str, reg: str, t: str) -> None:
        rs = self.rs()
        self.emit(f"{'cvt.rn.f16.f32' if t == 'f16' else 'cvt.rn.bf16.f32'} {rs}, {reg};")
        self.emit(f"st.{space}.b16 [{addr}], {rs};")

    def st_Let(self, st) -> None:
        t = dt_name(st.dtype)
        ptx_t = FLOAT[t][0] if t in FLOAT else ("s32" if t in INT else t)
        val = st.value
        if type(val).__name__ == "FlatIndex":
            buf = getattr(val, "buffer", None)
            lin = self.linear(buf, val.index)
            self.syms[st.name] = [lin, "s32"]
            return
        reg, got = self.expr(val)
        if got != ptx_t:
            reg = self.cast(reg, got, ptx_t)
        self.syms[st.name] = [reg, ptx_t]

    def st_Init(self, st) -> None:
        t = dt_name(st.dtype)
        ptx_t = FLOAT[t][0] if t in FLOAT else "s32"
        ident = st.identity
        reg = self.new_of(ptx_t)
        if self.is_float(ptx_t):
            bits = struct.unpack("<I", struct.pack("<f", float(ident)))[0]
            self.emit(f"mov.b32 {reg}, 0f{bits:08X};")
        else:
            self.emit(f"mov.s32 {reg}, {int(ident)};")
        self.syms[st.name] = [reg, ptx_t]

    def accum_identity(self, st):
        """The seed is always the combine's neutral element; ``base`` only redirects the fold."""
        op = st.op
        if getattr(op, "has_identity", False) and getattr(op, "identity", None) is not None:
            return op.identity
        name = getattr(op, "name", "")
        return {"add": 0.0, "multiply": 1.0, "maximum": float("-inf"), "minimum": float("inf")}.get(name, 0.0)

    def hoist_accums(self, body) -> None:
        """Accumulators live across iterations: seed them before the loop."""
        for name, st in self.collect_accums(body):
            if name in self.syms:
                continue
            t = dt_name(st.dtype)
            ptx_t = FLOAT[t][0] if t in FLOAT else "s32"
            reg = self.new_of(ptx_t)
            ident = self.accum_identity(st)
            if self.is_float(ptx_t):
                bits = struct.unpack("<I", struct.pack("<f", float(ident)))[0]
                self.emit(f"mov.b32 {reg}, 0f{bits:08X};")
            else:
                self.emit(f"mov.s32 {reg}, {int(ident)};")
            self.syms[name] = [reg, ptx_t]

    def collect_accums(self, body, out=None):
        if out is None:
            out = []
        for st in body:
            cls = type(st).__name__
            if cls == "Accum":
                out.append((st.name, st))
            elif cls in ("StridedLoop", "Loop"):
                continue  # hoisted by the nested loop itself
            else:
                for kid in getattr(st, "body", ()) or ():
                    self.collect_accums([kid], out)
                for kid in getattr(st, "else_body", ()) or ():
                    self.collect_accums([kid], out)
        return out

    def st_Accum(self, st) -> None:
        impl = st.op
        name = getattr(impl, "name", "") or str(impl)
        acc, t = self.syms[st.name]
        val = self.value(st.value)
        left = acc
        base = getattr(st, "base", None)
        if base is not None and base != st.name:
            left = self.value(base)[0]
        if self.is_float(t):
            if name not in BINARY_F:
                raise SystemExit(f"unsupported accumulator op {name!r}")
            self.emit(f"{BINARY_F[name]} {acc}, {left}, {val[0]};")
        else:
            if name not in BINARY_I:
                raise SystemExit(f"unsupported int accumulator op {name!r}")
            self.emit(f"{BINARY_I[name]} {acc}, {left}, {val[0]};")

    def st_Write(self, st) -> None:
        out = st.output
        values = list(st.values)
        n = len(values)
        t = dt_name(st.value_dtype)
        ptx_t = FLOAT[t][0] if t in FLOAT else (INT[t][0] if t in INT else t)
        regs = [self.syms[v][0] for v in values]
        shared = out in self.smem
        addr = self.shared_addr(out, st.index) if shared else self.global_addr(out, st.index)
        space = "shared" if shared else "global"
        if t in ("f16", "bf16"):
            if getattr(st, "atomic", False):
                raise SystemExit("atomic 16-bit write unsupported")
            for i, reg in enumerate(regs):
                self.store_half(space, self.step(addr, i * 2), reg, t)
            return
        if getattr(st, "atomic", False):
            if n != 1:
                raise SystemExit("atomic vector write unsupported")
            self.emit(f"red.{space}.add.{ptx_t} [{addr}], {regs[0]};")
            return
        if n == 1:
            self.emit(f"st.{space}.{ptx_t} [{addr}], {regs[0]};")
        elif n in (2, 4):
            self.emit(f"st.{space}.v{n}.{ptx_t} [{addr}], {{{', '.join(regs)}}};")
        else:
            raise SystemExit(f"unsupported write width {n}")

    def st_Smem(self, st) -> None:
        ext = tuple(dim_int(x) for x in st.extents)
        t = dt_name(st.dtype)
        align = getattr(st, "align", 0) or 0
        self.smem[st.name] = (ext, t, align)

    def st_Sync(self, st) -> None:
        warp = getattr(st, "warp", False)
        count = getattr(st, "count", None)
        bid = getattr(st, "barrier_id", 0) or 0
        if warp:
            self.emit("bar.warp.sync 0xffffffff;")
        elif count:
            self.emit(f"bar.sync {bid}, {count};")
        else:
            self.emit(f"bar.sync {bid};")

    def st_CpAsyncCopy(self, st) -> None:
        swz = str(getattr(st, "swizzle", "NONE") or "NONE")
        if swz.upper() != "NONE":
            raise SystemExit(f"unsupported cp.async swizzle {swz}")
        if getattr(st, "valid", None) is not None:
            raise SystemExit("predicated cp.async unsupported")
        # The destination is the stripe base plus the lane's own (row, col) inside it.
        lin = self.linear(st.smem, st.smem_index)
        lane = getattr(st, "lane_index", None)
        if lane is not None:
            cols = self.smem[st.smem][0][-1]
            l0, l1 = self.as_int(lane[0]), self.as_int(lane[1])
            lin = self.arith("add", lin, self.arith("add", self.arith("mul", l0, cols), l1))
        off = self.mul_const(lin, dt_bytes_for(self.smem[st.smem][1]))
        saddr = self.arith("add", self.shared_base(st.smem), off)
        gaddr = self.global_addr(st.src, st.src_index)
        nbytes = int(st.nbytes)
        if nbytes == 16:
            self.emit(f"cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
        elif nbytes in (4, 8):
            self.emit(f"cp.async.ca.shared.global [{saddr}], [{gaddr}], {nbytes};")
        else:
            raise SystemExit(f"unsupported cp.async size {nbytes}")

    def st_CpAsyncCommit(self, st) -> None:
        self.emit("cp.async.commit_group;")

    def st_CpAsyncWait(self, st) -> None:
        grp = int(getattr(st, "group", 0) or 0)
        self.emit(f"cp.async.wait_group {grp};")

    def st_Cond(self, st) -> None:
        pr, _ = self.expr(st.cond)
        other = self.label("_else")
        end = self.label("_endif")
        self.emit(f"@!{pr} bra {other};")
        self.stmts(st.body)
        if getattr(st, "else_body", None):
            self.emit(f"bra {end};")
            self.emit(f"{other}:")
            self.stmts(st.else_body)
            self.emit(f"{end}:")
        else:
            self.emit(f"{other}:")
        # a value computed on one side of a branch is not computed on the other
        self.cse.clear()
        if self.loop_cache:
            self.loop_cache[-1].clear()

    def st_Select(self, st) -> None:
        """Coord-predicated value binding; the last branch is the catch-all."""
        brs = st.branches
        ends = self.label("_selend")
        exits = [self.label("_selnext") for _ in brs]
        dst = None
        dst_t = "f32"
        for i, br in enumerate(brs):
            if i:
                self.emit(f"{exits[i]}:")
            if i + 1 < len(brs):
                pr, pt = self.expr(br.select)
                if pt != "pred":
                    pr = self._pred(pr, pt)
                self.emit(f"@!{pr} bra {exits[i + 1]};")
            val = self.value(br.value)
            if dst is None:
                dst, dst_t = self.new_of(val[1]), val[1]
            self.emit(f"mov.b32 {dst}, {val[0]};")
            if i + 1 < len(brs):
                self.emit(f"bra {ends};")
        self.emit(f"{ends}:")
        self.syms[st.name] = [dst, dst_t]
        self.cse.clear()
        if self.loop_cache:
            self.loop_cache[-1].clear()

    def _flat(self, body) -> bool:
        """True when every statement is straight-line: no nested loop, branch or barrier.

        Only such bodies are unrolled.  A loop that carries a nested loop unrolls into
        thousands of live temporaries and ptxas then shuffles them badly, which is both
        slower and (as measured on the attention kernel) numerically wrong.
        """
        allowed = {"Load", "Write", "Assign", "Accum", "Let", "Init", "CpAsyncCopy",
                   "CpAsyncCommit", "CpAsyncWait"}
        return all(type(s).__name__ in allowed for s in body)

    def _trip(self, st):
        """(first, step, count) when a strided loop has constant bounds, else None."""
        try:
            first = dim_int(st.start) if not isinstance(st.start, int) else st.start
            step_i = dim_int(st.step) if not isinstance(st.step, int) else st.step
            last = st.end if getattr(st, "end", None) is not None else st.axis.extent
            last = dim_int(last) if not isinstance(last, int) else last
        except Exception:
            return None
        if step_i <= 0 or last <= first:
            return None
        count = (last - first + step_i - 1) // step_i
        if count <= 0:
            return None
        return first, step_i, count

    def st_StridedLoop(self, st) -> None:
        self.hoist_accums(st.body)
        # A short constant-trip outer loop is worth unrolling: its counter feeds the
        # shared-memory stage rotation, which then folds to a constant offset instead of
        # running div/rem in the middle of the GEMM inner loop.
        bounds = self._trip(st)
        if bounds is not None and bounds[2] <= 1:  # outer unrolling measured slower: off
            first, step_i, count = bounds
            for i in range(count):
                self.syms[st.axis.name] = [str(first + i * step_i), "s32"]
                self.stmts(st.body)
            return
        top = self.label("_top")
        end = self.label("_lend")
        start, _ = self.expr(st.start)
        cur = self.r()
        self.emit(f"mov.s32 {cur}, {start};")
        self.dep[cur] = frozenset({cur})
        self._loop_enter(cur)
        mark = len(self.body)
        self.emit(f"{top}:")
        lim = self.as_int(st.end if getattr(st, "end", None) is not None else st.axis.extent)
        pr = self.p()
        self.emit(f"setp.ge.s32 {pr}, {cur}, {lim};")
        self.emit(f"@{pr} bra {end};")
        self.syms[st.axis.name] = [cur, "s32"]
        self.stmts(st.body)
        step = self.as_int(st.step)
        # values that do not depend on the counter stay valid across the back edge
        self.prune({cur})
        self._loop_exit()
        self._hoist_out(mark)
        self.emit(f"add.s32 {cur}, {cur}, {step};")
        self.emit(f"bra {top};")
        self.emit(f"{end}:")

    def st_Loop(self, st) -> None:
        self.hoist_accums(st.body)
        lim_v = st.end if getattr(st, "end", None) is not None else st.axis.extent
        lim_i = dim_int(lim_v) if not isinstance(lim_v, int) else lim_v
        if getattr(st, "unroll", False) and lim_i and 1 < lim_i <= 128 and self._flat(st.body):
            # Unroll with the counter as an immediate: every index expression in the body
            # folds to a constant, so the div/rem/mul chains never reach the GPU.
            for i in range(lim_i):
                self.syms[st.axis.name] = [str(i), "s32"]
                self.stmts(st.body)
            return
        end = self.label("_lend")
        cur = self.r()
        self.emit(f"mov.s32 {cur}, 0;")
        self.dep[cur] = frozenset({cur})
        self._loop_enter(cur)
        mark = len(self.body)
        top = self.label("_top")
        self.emit(f"{top}:")
        lim = self.as_int(st.end if getattr(st, "end", None) is not None else st.axis.extent)
        pr = self.p()
        self.emit(f"setp.ge.s32 {pr}, {cur}, {lim};")
        self.emit(f"@{pr} bra {end};")
        self.syms[st.axis.name] = [cur, "s32"]
        self.stmts(st.body)
        self.prune({cur})
        self._loop_exit()
        self._hoist_out(mark)
        self.emit(f"add.s32 {cur}, {cur}, 1;")
        self.emit(f"bra {top};")
        self.emit(f"{end}:")

    def merge(self, st, bindings) -> None:
        """Bind the carrier's state/state_b names, then run its combine program."""
        for name, reg, t in bindings:
            self.syms[name] = [reg, t]
        for a in getattr(st, "combine_states", ()) or ():
            self.st_Assign(a)

    def st_WarpShuffle(self, st) -> None:
        states = tuple(st.state)
        others = tuple(getattr(st, "state_b", ()) or ())
        length = int(getattr(st, "length", 32) or 32)
        off = length // 2
        while off >= 1:
            bindings = []
            for i, name in enumerate(states):
                acc, t = self.syms[name]
                tmp = self.r()
                got = self.r()
                other = self.new_of(t)
                self.emit(f"mov.b32 {tmp}, {acc};")
                self.emit(f"shfl.sync.bfly.b32 {got}|{self.p()}, {tmp}, {off}, 0x1f, 0xffffffff;")
                self.emit(f"mov.b32 {other}, {got};")
                bindings.append((others[i] if i < len(others) else f"{name}__o", other, t))
            self.merge(st, bindings)
            off //= 2

    def st_TreeHalve(self, st) -> None:
        bufs = tuple(getattr(st, "bufs", ()))
        states = tuple(st.state)
        others = tuple(getattr(st, "state_b", ()) or ())
        length = int(getattr(st, "length", 0))
        tid = getattr(st, "tid_var", "warp")
        treg = self.syms[tid][0] if tid in self.syms else self.syms["warp"][0]
        t = self.syms[states[0]][1]
        ptx_t = "f32" if self.is_float(t) else "s32"
        bar = getattr(st, "barrier_id", 0) or 0
        cnt = getattr(st, "barrier_count", None)
        s = length // 2
        while s >= 1:
            pr = self.p()
            self.emit(f"setp.lt.s32 {pr}, {treg}, {s};")
            skip = self.label("_tskip")
            self.emit(f"@!{pr} bra {skip};")
            bindings, slots = [], []
            for i, (buf, name) in enumerate(zip(bufs, states)):
                a_addr = self.shared_addr(buf, [treg])
                b_addr = self.shared_addr(buf, [self.plus(treg, s)])
                a = self.new_of(t)
                b = self.new_of(t)
                self.emit(f"ld.shared.{ptx_t} {a}, [{a_addr}];")
                self.emit(f"ld.shared.{ptx_t} {b}, [{b_addr}];")
                bindings.append((name, a, t))
                bindings.append((others[i] if i < len(others) else f"{name}__o", b, t))
                slots.append((a_addr, name))
            self.merge(st, bindings)
            for a_addr, name in slots:
                self.emit(f"st.shared.{ptx_t} [{a_addr}], {self.syms[name][0]};")
            self.emit(f"{skip}:")
            self.emit(f"bar.sync {bar}, {cnt};" if cnt else f"bar.sync {bar};")
            s //= 2
        for buf, name in zip(bufs, states):
            base = self.shared_base(buf)
            out = self.new_of(t)
            self.emit(f"ld.shared.{ptx_t} {out}, [{base}];")
            self.syms[name] = [out, t]

    def plus(self, reg: str, k: int) -> str:
        return self.arith("add", reg, k)

    # ---- program ---------------------------------------------------------
    def build(self, ir_path: str) -> dict:
        """Emit one kernel and return what the module writer needs."""
        self.ir_path = ir_path
        self.prepare()
        self.prologue()
        self.stmts(self.op.body)
        self._apply_lifts()
        self._fuse_madd()
        layout, total = self.smem_layout()
        return {
            "name": self.op.name,
            "params": list(self.params),
            "body": list(self.body),
            "layout": layout,
            "total": total,
            "regs": (self.n_p, self.n_r, self.n_f, self.n_rd, self.n_rs),
            "pool": total > 48 * 1024,
            "prefix": self.prefix,
        }


# Shared memory is addressed with 32-bit shared offsets, exactly like the worked example in
# examples/k_sum_1_reduce.ptx.  Anything above the 48 KiB static cap comes out of one dynamic
# pool instead, and a module holding several kernels gives each of them its own symbols.
def shared_decls(built: list[dict]) -> list[str]:
    out = []
    for b in built:
        if b["pool"]:
            out.append(f".extern .shared .align 16 .b8 {b['prefix']}_smem_pool[];")
        else:
            for sym, n, ptx_t, al, _off in b["layout"]:
                out.append(f".shared .align {al} .{ptx_t} {b['prefix']}{sym}[{n}];")
    return out


def entry_text(b: dict) -> list[str]:
    n_p, n_r, n_f, n_rd, n_rs = b["regs"]
    args = ",\n".join(f"\t.param .u64 p_{n}" for n, _t in b["params"])
    out = [f".visible .entry {b['name']}(\n{args}\n)", "{"]
    if n_p:
        out.append(f"\t.reg .pred %p<{n_p + 1}>;")
    if n_r:
        out.append(f"\t.reg .b32 %r<{n_r + 1}>;")
    if n_f:
        out.append(f"\t.reg .f32 %f<{n_f + 1}>;")
    if n_rd:
        out.append(f"\t.reg .b64 %rd<{n_rd + 1}>;")
    if n_rs:
        out.append(f"\t.reg .b16 %rs<{n_rs + 1}>;")
    out.extend(b["body"])
    out.append("\tret;")
    out.append("}")
    return out


def module_text(built: list[dict]) -> str:
    lines = [".version 8.0", ".target sm_89", ".address_size 64", ""]
    decls = shared_decls(built)
    lines.extend(decls)
    if decls:
        lines.append("")
    for b in built:
        lines.extend(entry_text(b))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def dt_bytes_for(t: str) -> int:
    if t in FLOAT:
        return FLOAT[t][1]
    if t in INT:
        return INT[t][1]
    raise SystemExit(f"unknown dtype {t!r}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.exit("usage: python compile.py path/to/kernel.ir.json > out.ptx")
    ir_path = argv[1]
    graph = Graph.from_dict(json.loads(Path(ir_path).read_text()))
    kernels = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if not kernels:
        sys.exit("no KernelOp in this graph")
    multi = len(kernels) > 1
    built = [Emitter(graph, n, prefix=f"{n.op.name}__" if multi else "").build(ir_path) for n in kernels]
    sys.stdout.write(module_text(built))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


