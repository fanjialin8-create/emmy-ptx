"""Dump a compact structural summary of every open problem's Kernel IR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph

WS = Path(r"C:\Users\Arous\Documents\deepseek-harness\default-workspace")
PROBLEMS = WS / "hackathon-2026" / "problems"


def attrs(obj, names):
    out = []
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v in (None, (), [], "", 0, False, "NONE"):
                continue
            out.append(f"{n}={v!r}")
    return ", ".join(out)


def walk(st, depth=0, out=None):
    if out is None:
        out = []
    pad = "  " * depth
    name = type(st).__name__
    detail = attrs(st, ("name", "input", "output", "op", "args", "dtype", "shape", "role"))
    if name in ("Tile", "StridedLoop", "Loop", "Cond"):
        detail = attrs(st, ("axes", "axis", "start", "step", "end", "unroll", "cond", "block_threads", "aux_threads"))
    elif name == "Write":
        detail = attrs(st, ("output", "values", "atomic", "value_dtype", "index"))
    elif name == "Load":
        detail = attrs(st, ("names", "input", "index", "dtype"))
    elif name == "Accum":
        detail = attrs(st, ("name", "op", "value", "axes", "base", "dtype"))
    elif name == "WarpShuffle":
        detail = attrs(st, ("state", "state_b", "combine_states", "length", "dtype"))
    elif name == "TreeHalve":
        detail = attrs(st, ("bufs", "state", "state_b", "combine_states", "length", "tid_var", "barrier_id", "barrier_count", "inner"))
    elif name == "Smem":
        detail = attrs(st, ("name", "extents", "dtype", "align"))
    elif name == "Sync":
        detail = attrs(st, ("barrier_id", "count", "warp"))
    out.append(f"{pad}{name}({detail})")
    for kid in getattr(st, "body", ()) or ():
        walk(kid, depth + 1, out)
    else_body = getattr(st, "else_body", ()) or ()
    for kid in else_body:
        walk(kid, depth + 1, out)
    return out


def summary(path: Path) -> list[str]:
    g = Graph.from_dict(json.loads(path.read_text()))
    lines = [f"== {path.parent.name}  ({path.name})"]
    consts = {}
    for name, node in g.nodes.items():
        if type(node.op).__name__ == "ConstantOp":
            consts[name] = getattr(node.op, "value", None)
    lines.append(f"  graph inputs={list(g.inputs)} outputs={list(g.outputs)}")
    for name, node in g.nodes.items():
        kind = type(node.op).__name__
        outs = [(o.name, tuple(o.shape), o.dtype) for o in node.outputs]
        if kind == "InputOp":
            lines.append(f"  input {name}: shape={tuple(node.outputs[0].shape)} dtype={node.outputs[0].dtype}")
        elif kind == "ConstantOp":
            v = getattr(node.op, "value", None)
            lines.append(f"  const {name}: value={v!r} shape={tuple(node.outputs[0].shape)} dtype={node.outputs[0].dtype}")
        elif kind == "KernelOp":
            op = node.op
            lines.append(f"  kernel entry={op.name!r} args={list(node.inputs)}+{[o.name for o in node.outputs]}"
                         f" zero_delegated={getattr(op, 'zero_delegated', None)} serial={getattr(op, 'serial', None)}")
            for st in op.body:
                lines.append("  body:")
                lines.extend("    " + l for l in walk(st))
        else:
            lines.append(f"  node {name}: {kind} outs={outs}")
    return lines


def main() -> None:
    targets = sys.argv[1:] or [p.name for p in sorted(PROBLEMS.iterdir()) if p.is_dir()]
    for t in targets:
        d = PROBLEMS / t
        for ir in sorted(d.glob("*.ir.json")):
            print("\n".join(summary(ir)))


if __name__ == "__main__":
    main()
