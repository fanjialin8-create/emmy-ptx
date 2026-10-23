"""Compile all 15 public problems with two different emmy versions and diff the PTX.

The task repo pins emmy-ml==0.3.11 while local development uses the 0.3.22 source tree.
If the wire format or attribute names moved between those releases the compiler would
break on the grader, so verify: same IR in, byte-identical PTX out.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WS = Path(r"C:\Users\Arous\Documents\deepseek-harness\default-workspace")
PROBLEMS = WS / "hackathon-2026" / "problems"
COMPILER = WS / "_proj" / "emmy-ptx" / "compile.py"
SHIM = WS / "_shim"
V311 = WS / "_src" / "emmy_ml_0.3.11"
V322 = WS / "emmy-hackathon" / "emmy-main"
PY = WS / ".venv" / "Scripts" / "python.exe"


def run(ir: Path, emmy_root: Path) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SHIM), str(emmy_root)])
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(
        [str(PY), str(COMPILER), str(ir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    irs = sorted(PROBLEMS.glob("*/*.ir.json"))
    print(f"problems: {len(irs)}")
    bad = 0
    for ir in irs:
        ra, outa, erra = run(ir, V322)
        rb, outb, errb = run(ir, V311)
        name = ir.parent.name
        if ra != 0 or rb != 0:
            print(f"[ERROR] {name}: 0.3.22 rc={ra} 0.3.11 rc={rb}")
            print("   0.3.11 stderr tail:", (errb or "").strip().splitlines()[-1:] or "")
            bad += 1
            continue
        same = outa == outb
        lines = outa.count("\n")
        print(f"[{'SAME' if same else 'DIFF'}] {name}: {lines} lines")
        if not same:
            bad += 1
            a, b = outa.splitlines(), outb.splitlines()
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"   first diff at line {i + 1}:\n     0.3.22: {x}\n     0.3.11: {y}")
                    break
            if len(a) != len(b):
                print(f"   line count differs: {len(a)} vs {len(b)}")
    print(f"\n{bad} problem(s) failed or differed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
