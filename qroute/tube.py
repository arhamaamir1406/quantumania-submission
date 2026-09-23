"""Tube LNS and the long-budget solver built on it.

Large instances are out of reach of the exact model as a whole (`dense_random`,
40 gates: no improvement from whole-instance CP-SAT in 15 minutes), and
windows pinned to the incumbent at both ends turn out to be locally optimal
already. The *tube* is a different neighbourhood: keep the whole instance, but
allow every gate only within `r` layers of where the incumbent runs it. SWAPs
stay completely free. Because the neighbourhood spans the entire horizon,
CP-SAT can trade SWAPs for depth globally -- the kind of move the move-set
search cannot express -- while the model stays small enough to solve.

On `dense_random`, portfolio + tube rounds went 40.0 -> 39.0 -> 38.5 -> 36.0
(24 SWAPs, depth 24) in ~11 minutes, below the best the portfolio or window
LNS had ever produced (36.5).

The solver is the watermark model (watermark.py), which encodes the scorer's
order rule exactly, so every tube solution can be emitted as-is.
"""
from __future__ import annotations

import os
import time

import networkx as nx

from starter_kit.scorer import score_summary

from .bounds import score_lower_bound
from .ir import materialize


def tube_lns(program, hw: nx.Graph, placement, routed, deadline: float,
             radii=(2, 4, 3), round_limit: float = 150.0, stall_limit: int = 3,
             horizon_slack: int = 3, workers: int | None = None, seed: int = 0,
             verbose: bool = False):
    """Improve (placement, routed) by tube rounds until `stall_limit` rounds
    in a row find nothing, or the deadline. Returns (score, placement, routed),
    never worse than the input."""
    from .watermark import layered_from_routed, solve_exact

    workers = workers or max(1, os.cpu_count() or 1)
    swap_floor = score_lower_bound(program, hw)[1]
    best = score_summary(program, hw, placement, routed)["score"]
    stall, rnd = 0, 0
    while stall < stall_limit and deadline - time.monotonic() > 5.0:
        r_ = radii[rnd % len(radii)]
        depth = len(layered_from_routed(program, hw, placement, routed)[0])
        r = solve_exact(program, hw, hint=(placement, routed), upper=best,
                        horizon=depth + horizon_slack, tube=r_, swap_floor=swap_floor,
                        time_limit=min(round_limit, deadline - time.monotonic()),
                        workers=workers, seed=seed * 1009 + rnd)
        rnd += 1
        if r.ops is not None and r.score < best:
            cand = materialize(program, r.placement, r.ops)
            s = score_summary(program, hw, r.placement, cand)
            if s["valid"] and s["score"] < best:        # the scorer has the last word
                best, placement, routed, stall = s["score"], r.placement, cand, 0
                if verbose:
                    print(f"    [tube r={r_}] {best:.1f} (swaps {s['swap_count']}, "
                          f"depth {s['depth']})", flush=True)
                continue
        stall += 1
        if verbose:
            print(f"    [tube r={r_}] no gain ({r.status})", flush=True)
    return best, placement, routed


def solve_long(program, hw: nx.Graph, budget: float, restart_budget: float = 60.0,
               exact_max_gates: int = 24, verbose: bool = False, **portfolio_kw):
    """Portfolio restarts, each followed by tube LNS, within one budget.

    Small instances (the exact model already runs whole inside the portfolio)
    get a single portfolio run. Each restart uses a different seed, because the
    tube plateaus per starting solution and the portfolio's incumbents differ
    run to run. Returns (placement, routed).
    """
    from .portfolio import solve as portfolio_solve

    t0 = time.monotonic()
    deadline = t0 + budget
    n2q = sum(1 for op in program if op[0] == "2Q")
    floor = score_lower_bound(program, hw)[0]
    best = None
    restart = 0
    while True:
        left = deadline - time.monotonic()
        if best is not None and left < restart_budget + 30:
            break
        rb = max(1.0, min(restart_budget, left))
        pl, rt = portfolio_solve(program, hw, budget=rb, seed=0xC0FFEE + restart,
                                 verbose=False, **portfolio_kw)
        sc = score_summary(program, hw, pl, rt)["score"]
        if verbose:
            print(f"  [restart {restart}] portfolio {sc:.1f}", flush=True)
        if best is None or sc < best[0]:
            best = (sc, pl, rt)
        if n2q <= exact_max_gates or sc <= floor:
            break
        sc, pl, rt = tube_lns(program, hw, pl, rt, deadline, seed=restart, verbose=verbose)
        if sc < best[0]:
            best = (sc, pl, rt)
        restart += 1
    return best[1], best[2]
