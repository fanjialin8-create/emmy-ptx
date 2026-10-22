"""Reload a stage dump into live objects, then report which kernel-IR statement classes it uses.

Emmy ships a wire codec (``emmy/compiler/wire.py``) and ``Graph.from_dict``, and the `--ir <stage>` dumps are written
through it. If a dump round-trips, a PTX emitter prototype can be driven in-process from real IR instead of from text.

Run:  python _setup/probe_dump_reload.py [dump.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "_shim"))
sys.path.insert(0, str(WS / "emmy-hackathon" / "emmy-main"))

from emmy.compiler.graph import Graph  # noqa: E402

INTERESTING = (
    "Load", "Store", "Assign", "Accum", "Init", "Let", "Write", "Cond", "Sync", "Smem", "Tile",
    "Loop", "StridedLoop", "WarpShuffle", "WarpBroadcast", "TreeHalve", "Pre", "Carry", "ZeroPrologue",
    "Call", "Subroutine", "Select", "Affine", "SetMaxNReg", "RegStore", "RegFragment", "FragmentApply",
    "MmaSyncPtx", "LdmatrixLoad", "CpAsyncCopy", "TmaLoad", "MbarrierWait",
)


def walk(obj, seen: dict[str, int], depth: int = 0) -> None:
    name = type(obj).__name__
    if name.startswith(INTERESTING):
        seen[name] = seen.get(name, 0) + 1
    if depth > 14:
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            walk(item, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            walk(item, seen, depth + 1)
        return
    for key, item in vars(obj).items() if hasattr(obj, "__dict__") else ():
        if key.startswith("_"):
            continue
        walk(item, seen, depth + 1)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else WS / "_out" / "tiny1" / "09_lowering_kernel.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"file    : {path.name} ({path.stat().st_size} bytes)  top keys: {sorted(data)[:8]}")

    graph = Graph.from_dict(data)
    print(f"graph   : ok  nodes={len(graph.nodes)}  inputs={getattr(graph, 'inputs', None)}")

    seen: dict[str, int] = {}
    for name, node in graph.nodes.items():
        walk(node, seen)
    total = sum(seen.values())
    print(f"classes : {len(seen)} distinct, {total} instances")
    for name, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
