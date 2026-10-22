"""Survey: which IR node kinds, elementwise ops and shapes the 15 open problems actually use."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from emmy.compiler.graph import Graph

WS = Path(r"C:\Users\Arous\Documents\deepseek-harness\default-workspace")
PROBLEMS = WS / "hackathon-2026" / "problems"

stmts: Counter = Counter()
ops: Counter = Counter()
exprs: Counter = Counter()
per_problem: dict[str, set[str]] = {}


def walk(node):
    stmts[type(node).__name__] += 1
    if type(node).__name__ == "Assign":
        impl = node.op
        nm = getattr(impl, "name", None) or getattr(impl, "op", None) or repr(impl)
        ops[str(nm)] += 1
    for kid in getattr(node, "body", ()) or ():
        walk(kid)
    for kid in getattr(node, "else_body", ()) or ():
        walk(kid)


def walk_expr(e):
    if e is None:
        return
    tn = type(e).__name__
    exprs[tn] += 1
    for attr in ("left", "right", "if_true", "if_false", "cond"):
        v = getattr(e, attr, None)
        if v is not None and hasattr(v, "__class__") and tn != "Literal":
            walk_expr(v)
    for v in getattr(e, "args", ()) or ():
        walk_expr(v)


for d in sorted(PROBLEMS.iterdir()):
    if not d.is_dir():
        continue
    names = set()
    for ir in d.glob("*.ir.json"):
        g = Graph.from_dict(json.loads(ir.read_text()))
        for name, node in g.nodes.items():
            if type(node.op).__name__ == "KernelOp":
                for st in node.op.body:
                    walk(st)
                    names.add(type(st).__name__)
                    for sub in st.__dict__.values():
                        if hasattr(sub, "__class__") and type(sub).__name__.endswith("Expr"):
                            walk_expr(sub)
    per_problem[d.name] = names

print("STATEMENTS")
for k, v in stmts.most_common():
    print(f"  {k:<20} {v}")
print("ASSIGN OPS")
for k, v in ops.most_common():
    print(f"  {k:<20} {v}")
print("EXPRS")
for k, v in exprs.most_common():
    print(f"  {k:<20} {v}")
print("PER PROBLEM")
for k, v in per_problem.items():
    print(f"  {k:<22} {sorted(v)}")

# attribute names of one ElementwiseImpl
for d in sorted(PROBLEMS.iterdir()):
    for ir in d.glob("*.ir.json"):
        g = Graph.from_dict(json.loads(ir.read_text()))
        for node in g.nodes.values():
            if type(node.op).__name__ == "KernelOp":
                for st in node.op.body:
                    for child in getattr(st, "body", ()) or ():
                        if type(child).__name__ == "Assign":
                            impl = child.op
                            print("ElementwiseImpl dir:", [a for a in dir(impl) if not a.startswith('_')])
                            print("  dict:", getattr(impl, "__dict__", None))
                            print("  repr:", repr(impl))
                            raise SystemExit
