"""Check the tiny RMSNorm program numerically: Emmy's kernel on the GPU against numpy on the host.

The harness already proves a kernel launches. This proves the numbers are right, which is what a lowering or
emission change actually has to preserve.

Run:  python _setup/verify_tiny.py [dump-dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emu_run  # noqa: E402

WS = Path(__file__).resolve().parent.parent
EPS = 1e-6


def main() -> int:
    dump = sys.argv[1] if len(sys.argv) > 1 else str(WS / "_out" / "tiny1")
    argv = sys.argv
    sys.argv = ["emu_run", dump]
    rc = emu_run.main()
    sys.argv = argv
    if rc != 0:
        return rc

    host = emu_run.LAST["host"]
    plan = emu_run.LAST["plan"]
    x = np.asarray(host["x"], dtype=np.float32)
    w = np.asarray(host["p_weight"], dtype=np.float32)
    got = np.asarray(host["rms_norm"], dtype=np.float32)

    desired_shape = tuple(plan["nodes"]["rms_norm"]["outputs"][0]["shape"])
    x = x.reshape(desired_shape)
    w = w.reshape(-1)
    got = got.reshape(desired_shape)

    # nn.RMSNorm normalizes over the last dimension and scales by the weight.
    xf = x.astype(np.float64)
    ms = (xf * xf).mean(axis=-1, keepdims=True)
    want = (xf / np.sqrt(ms + EPS) * w.astype(np.float64)).astype(np.float32)

    diff = np.abs(got - want)
    scale = np.maximum(np.abs(want), 1e-6)
    rel = diff / scale
    print(f"\ncompare : shape={got.shape}  max_abs={diff.max():.3e}  max_rel={rel.max():.3e}")
    ok = float(rel.max()) < 2e-4
    print("RESULT: " + ("numbers match numpy" if ok else "MISMATCH"))
    if not ok:
        for i in np.argsort(rel.ravel())[-4:]:
            print(f"        idx {i}: got {got.ravel()[i]:.6f} want {want.ravel()[i]:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
