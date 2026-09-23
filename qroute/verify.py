"""Check stored routings with the organisers' scorer.

    python -m qroute.verify results/*.json

Each file holds {"benchmark", "placement", "routed", ...}. Nothing here is
trusted: the benchmark program, hardware graph and scorer all come from the
vendored starter kit.
"""
from __future__ import annotations

import json
import sys

from starter_kit.benchmarks import BENCHMARKS
from starter_kit.hardware import build_hardware_graph
from starter_kit.scorer import score_summary


def verify(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    placement = {int(k): v for k, v in d["placement"].items()}
    routed = [tuple(op) for op in d["routed"]]
    return score_summary(BENCHMARKS[d["benchmark"]], build_hardware_graph(), placement, routed)


def main():
    ok = True
    for path in sys.argv[1:]:
        s = verify(path)
        ok &= s["valid"]
        print(f"{path}: valid={s['valid']} ({s['message']})  swaps={s['swap_count']}  "
              f"depth={s['depth']}  score={s['score']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
