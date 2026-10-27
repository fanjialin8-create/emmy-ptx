"""Hidden-set stress test: diff our PTX against emmy's own CUDA codegen.

For every expression we ask emmy for a kernel-stage IR (our input) plus the CUDA
source it would have compiled itself (reference oracle), then run both on the
same data and compare.

usage:  python tools/stress.py --cases cases.txt
        python tools/stress.py -c "nn.LayerNorm(256)(torch.randn(2,8,256))"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle as O  # noqa: E402

WORK = O.ROOT / "_out" / "stress"
COMPILE = O.ROOT / "_proj" / "emmy-ptx" / "compile.py"


ENV = dict(os.environ)
ENV["PYTHONPATH"] = f"{O.SHIM};{O.EMMY_SRC}"
ENV["PYTHONUTF8"] = "1"
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["EMMY_WORKERS"] = "1"
ENV["NO_PROXY"] = "*"


def emmy_dump(expr: str, dump: Path):
    env = dict(ENV)
    return subprocess.run(
        [str(O.PY), "-m", "emmy.emmy", "compile", "-c", expr, "--ir", "cuda",
         "--target", "sm_89", "--dump-dir", str(dump)],
        cwd=str(O.EMMY_SRC), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def my_ptx(ir: Path) -> str:
    r = subprocess.run([str(O.PY), str(COMPILE), str(ir)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-800:])
    return r.stdout


def last_line(text: str) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return (lines[-1] if lines else "")[:400]


def dims3(v):
    return [int(x[0]) if isinstance(x, (list, tuple)) else int(x) for x in v]


def run_case(name: str, expr: str) -> tuple[str, str]:
    dump = WORK / name
    if dump.exists():
        shutil.rmtree(dump, ignore_errors=True)
    dump.parent.mkdir(parents=True, exist_ok=True)
    r = emmy_dump(expr, dump)
    ir_path = dump / "09_lowering_kernel.json"
    if not ir_path.exists():
        return "DUMP-FAIL", (r.stdout + r.stderr).strip().splitlines()[-1][:300]

    plan_path = dump / "10_lowering_cuda.json"
    if not plan_path.exists():
        return "DUMP-FAIL", "no cuda plan: " + last_line((r.stdout + r.stderr).strip()[-400:])
    plan = json.loads(plan_path.read_text())
    cuda = [n for n in plan["nodes"].values() if n["op"] == "CudaOp"]
    if not cuda:
        return "NO-KERNEL", "no CudaOp in the plan"

    ir = json.loads(ir_path.read_text())
    facts = O.graph_facts(ir_path)
    outputs = set(ir["outputs"])
    plans = [n["op_fields"] for n in cuda]
    missing = sorted({a for p in plans for a in p["arg_order"] if a not in facts})
    if missing:
        return "NO-FACTS", "unknown args " + ",".join(missing)

    try:
        mine = O.assemble(my_ptx(ir_path), WORK / f"{name}.mine.cubin")
    except RuntimeError as exc:
        return "PTX-FAIL", last_line(str(exc))

    refs = []
    for i, pf in enumerate(plans):
        try:
            refs.append(O.assemble(O.nvrtc_ptx(pf["kernel_source"]), WORK / f"{name}.ref{i}.cubin"))
        except RuntimeError as exc:
            return "REF-FAIL", last_line(str(exc))

    # One buffer per name, shared by every kernel in the plan: the kernels chain through memory.
    names = sorted({a for p in plans for a in p["arg_order"]})
    pool_mine = dict(zip(names, O.make_args(names, facts, outputs)))
    pool_ref = {k: v.copy() for k, v in pool_mine.items()}

    try:
        for pf, ref in zip(plans, refs):
            order = pf["arg_order"]
            for zeroed in pf.get("zero_outputs") or []:
                if zeroed in pool_mine:
                    pool_mine[zeroed][...] = 0
                    pool_ref[zeroed][...] = 0
            grid, block, smem = dims3(pf["grid"]), dims3(pf["block"]), int(pf["smem_bytes"])
            O.launch(mine, pf["kernel_name"], grid, block, smem, [pool_mine[a] for a in order])
            O.launch(ref, pf["kernel_name"], grid, block, smem, [pool_ref[a] for a in order])
    except RuntimeError as exc:
        return "RUN-FAIL", last_line(str(exc))

    worst, detail = 0.0, []
    for nm in names:
        if nm not in outputs:
            continue
        a = np.asarray(pool_mine[nm], dtype=np.float64)
        b = np.asarray(pool_ref[nm], dtype=np.float64)
        if a.shape != b.shape:
            return "SHAPE", f"{nm}: {a.shape} vs {b.shape}"
        if a.size == 0:
            continue
        scale = np.maximum(np.abs(b), 1e-3)
        rel = float(np.max(np.abs(a - b) / scale))
        worst = max(worst, rel)
        detail.append(f"{nm}: max_rel={rel:.3e}")
    ok = worst < 5e-3
    tag = "" if len(plans) == 1 else f"[{len(plans)} kernels] "
    return ("PASS" if ok else "MISMATCH"), tag + f"worst_rel={worst:.3e} " + " ".join(detail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--code", action="append", default=[])
    ap.add_argument("--cases", type=Path)
    args = ap.parse_args()

    cases: list[tuple[str, str]] = [(f"c{i:02d}", e) for i, e in enumerate(args.code)]
    if args.cases:
        for line in args.cases.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, expr = line.partition("|")
            cases.append((name.strip(), expr.strip() or name.strip()))
    if not cases:
        print("no cases")
        return 2

    fails = []
    for i, (name, expr) in enumerate(cases, 1):
        status, info = run_case(name, expr)
        print(f"[{status}] {name}: {expr}\n         {info}", flush=True)
        if status != "PASS":
            fails.append((name, status, info))
    print(f"\n{len(cases) - len(fails)}/{len(cases)} passed", flush=True)
    if fails:
        print("failing:", ", ".join(f"{n}({s})" for n, s, _ in fails), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
