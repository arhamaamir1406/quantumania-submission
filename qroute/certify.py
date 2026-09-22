"""Certify optimality with the relaxed exact model (see exact.py).

    python -m qroute.certify ladder_trotter 6.5
    python -m qroute.certify qaoa_random 11.5

To prove that no routing scores below the incumbent `S`, it suffices to show
that no solution scores `<= S - 0.5` (scores are multiples of 0.5). Such a
solution has at least `s0` SWAPs (the analytical floor) and depth `d` in
`[d0, 2 * (S - 0.5 - s0)]`, where `d0` is the logical ASAP depth. The relaxed
model admits every valid routed program, so it is solved once per depth slice
with the depth fixed and the objective capped at `S - 0.5`: if every slice is
INFEASIBLE, `S` is optimal. Slicing keeps each model small -- the swap budget
shrinks as the depth grows.
"""
from __future__ import annotations

import sys
import time

from starter_kit.benchmarks import BENCHMARKS
from starter_kit.hardware import build_hardware_graph

from .bounds import asap_depth, swap_lower_bound
from .exact import solve_exact


def certify(program, hw, incumbent: float, time_limit: float = 600.0, verbose: bool = True):
    """Returns (proved: bool, per-slice statuses)."""
    s0, _ = swap_lower_bound(program, hw)
    d0 = asap_depth(program)
    target = incumbent - 0.5
    d_max = int(2 * (target - s0))
    slices = []
    for d in range(d0, d_max + 1):
        t = time.monotonic()
        r = solve_exact(program, hw, T=d, strict=False, time_limit=time_limit,
                        upper=target, fix_depth=d, lazy=False)
        slices.append((d, r["status"]))
        if verbose:
            print(f"  depth {d:2d}: {r['status']:10s} {time.monotonic() - t:6.1f}s", flush=True)
        if r["status"] != "INFEASIBLE":
            return False, slices
    return True, slices


def main():
    name, incumbent = sys.argv[1], float(sys.argv[2])
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    hw = build_hardware_graph()
    print(f"{name}: is {incumbent} optimal? (no solution <= {incumbent - 0.5})")
    ok, _ = certify(BENCHMARKS[name], hw, incumbent, limit)
    print(f"{name}: {'PROVED OPTIMAL' if ok else 'not proved'} at {incumbent}")


if __name__ == "__main__":
    main()
