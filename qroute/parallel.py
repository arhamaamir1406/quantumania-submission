"""Best-of-K portfolio runs in parallel, inside one wall-clock budget.

One portfolio run does not use a 24-core machine well: its policy stage is
largely serial work on the GPU. Splitting the cores K ways costs a run almost
nothing (dense_random, 60 s: 36.22 mean on all cores vs 36.28 on 6), and the
best of K independent runs is both better and far steadier (best-of-3 mean
35.75, worst 36.5, vs single runs ranging up to 38.5; best-of-4 vs best-of-3
on 24 cores, 8 trials: 35.56 vs 36.00, worst 36.0 vs 37.5, no losses).

Runs are separate spawned processes (not a Pool: pool workers are daemonic
and cannot start the placement search's own process pool). Any failure falls
back to one ordinary run, so this can never do worse than not using it.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import networkx as nx

# A run needs a handful of cores for its placement search to stay effective.
MIN_CORES_PER_RUN = 6
MAX_RUNS = 4
SPAWN_OVERHEAD = 3.0     # seconds for a spawned run to import and load the policy


def plan_runs(cores: int | None = None) -> int:
    cores = cores or os.cpu_count() or 1
    return max(1, min(MAX_RUNS, cores // MIN_CORES_PER_RUN))


def _child(program, hw_edges, hw_nodes, budget, seed, workers, q, kw):
    try:
        import networkx as _nx
        from .portfolio import solve
        hw = _nx.Graph()
        hw.add_nodes_from(hw_nodes)
        hw.add_edges_from(hw_edges)
        pl, rt = solve(program, hw, budget=budget, seed=seed, workers=workers, **kw)
        q.put((seed, pl, rt))
    except Exception:                      # reported as a missing result
        q.put((seed, None, None))


def solve_parallel(program: list[tuple], hw: nx.Graph, budget: float,
                   runs: int | None = None, base_seed: int = 0xC0FFEE, **portfolio_kw):
    """Best valid result of `runs` parallel portfolio runs. Returns
    (placement, routed), or None if the parallel path could not be used."""
    from starter_kit.scorer import score_summary

    runs = runs or plan_runs()
    if runs < 2 or budget <= 2 * SPAWN_OVERHEAD + 5:
        return None
    workers = max(1, (os.cpu_count() or runs) // runs)
    t0 = time.monotonic()
    child_budget = budget - SPAWN_OVERHEAD
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_child,
                         args=(program, list(hw.edges), list(hw.nodes), child_budget,
                               base_seed + 7919 * k, workers, q, portfolio_kw))
             for k in range(runs)]
    try:
        for p in procs:
            p.start()
        results = []
        for _ in procs:
            left = budget + 30 - (time.monotonic() - t0)   # generous: never hang
            try:
                results.append(q.get(timeout=max(1.0, left)))
            except Exception:
                break
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    best = None
    for _, pl, rt in results:
        if pl is None:
            continue
        s = score_summary(program, hw, pl, rt)
        if s["valid"] and (best is None or s["score"] < best[0]):
            best = (s["score"], pl, rt)
    return None if best is None else (best[1], best[2])
