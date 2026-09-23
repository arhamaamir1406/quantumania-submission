"""Placement search: sample many starting placements, then improve the best.

Routing from a fixed placement saturates fast -- beam widths from 50 to 1200
return the same score, and even a random value function matches the learned
one on most benchmarks. The starting placement is what moves the score: across
a few hundred jittered constructive placements the best one beats the old
portfolio by 0.5-5.5 points per instance. Good placements are rare (2-15% of
samples beat the old result), so we need volume, and a narrow beam is cheap
enough (1-60 ms) to be the placement's fitness function.

Two stages, both parallel across a process pool (fork, or spawn on Windows):

1. **Sampling** -- jittered constructive placements at a spread of jitter
   levels, each scored by a narrow beam.
2. **Iterated local search** from the best few distinct placements. The
   neighbourhood swaps two logicals or moves one to a free physical qubit,
   restricted to physical qubits within distance 2 of where the logical sits
   (the hardware has max degree 3, so this stays ~9 moves per logical).
   First-improvement over shuffled batches sized to the worker count (so a
   serial run takes many cheap steps, a parallel one wide ones); on a local optimum, perturb with a few random swaps and
   continue from the incumbent.

Beam width is non-monotone in quality here: a width-16 beam often beats a
width-100 one, and scoring more placements at width 16 beat scoring fewer at
(16, 48) on every worker count tried. The final wide beam in the portfolio
still runs from the best placements found.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import time

import networkx as nx

from .mdp import Problem
from .placement import constructive_placement
from .search import beam_search

INF = float("inf")
EVAL_WIDTHS = (16,)

# Worker-process state, set by _init. With fork the Problem is inherited, not
# re-pickled per task.
_P: Problem | None = None


def _init(program, hw):
    global _P
    _P = Problem(program, hw)


def _evaluate(problem: Problem, placement: dict[int, int], widths=None):
    """Best (cost, ops) over the beam widths, or (INF, None)."""
    widths = EVAL_WIDTHS if widths is None else widths
    best_cost, best_ops = INF, None
    for w in widths:
        s = beam_search(problem, problem.initial(placement), width=w, incumbent=best_cost)
        if s is not None and s.cost < best_cost:
            best_cost, best_ops = s.cost, s.ops()
    return best_cost, best_ops


def _task(placement):
    return _evaluate(_P, placement)



def _spawn_pool(workers: int, program, hw):
    """A spawn-context pool that never re-imports the caller's __main__.

    Spawned workers normally re-import the parent's __main__. From a REPL,
    stdin or notebook there is nothing to import, and from a caller's script
    without an `if __name__ == "__main__"` guard the import re-runs that
    script; either way each worker dies during bootstrap and the pool respawns
    it forever. The workers only need this module, which they import by name,
    so __main__ is swapped for an empty module while they start.
    """
    import sys
    import types
    real = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        return mp.get_context("spawn").Pool(workers, _init, (program, hw))
    finally:
        if real is not None:
            sys.modules["__main__"] = real


class Evaluator:
    """Scores placements in a process pool when possible, serially otherwise.

    fork where available (Linux); spawn otherwise (Windows, where there is no
    fork -- without it the whole placement search ran on one core there).
    Spawn workers cost ~1 s to start, so they are skipped for short budgets.
    """

    def __init__(self, program, hw, workers: int | None = None, allow_spawn: bool = True):
        self.problem = Problem(program, hw)
        self.pool = None
        if workers is None:
            workers = max(1, (os.cpu_count() or 1) - 1)
        if workers > 1:
            try:
                if "fork" in mp.get_all_start_methods():
                    self.pool = mp.get_context("fork").Pool(workers, _init, (program, hw))
                elif allow_spawn:
                    self.pool = _spawn_pool(workers, program, hw)
            except (OSError, ValueError, RuntimeError):
                self.pool = None
        # Neighbour batch size for local search: one round of the pool, or a
        # handful of evaluations when serial.
        self.batch = 2 * self.pool._processes if self.pool is not None else 6
        self.cache: dict[tuple, tuple] = {}

    def __call__(self, placements: list[dict[int, int]]) -> list[tuple]:
        keys = [tuple(sorted(pl.items())) for pl in placements]
        todo = [(k, pl) for k, pl in zip(keys, placements) if k not in self.cache]
        uniq = dict(todo)
        if uniq:
            items = list(uniq.items())
            if self.pool is not None:
                results = self.pool.map(_task, [pl for _, pl in items],
                                        chunksize=max(1, len(items) // (4 * self.pool._processes)))
            else:
                results = [_evaluate(self.problem, pl) for _, pl in items]
            for (k, _), r in zip(items, results):
                self.cache[k] = r
        return [self.cache[k] for k in keys]

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None


def sample_placements(program, hw, rng: random.Random, n: int) -> list[dict[int, int]]:
    out = [constructive_placement(program, hw)]
    jitters = (0.5, 1.0, 1.5, 2.5, 4.0)
    for i in range(n - 1):
        out.append(constructive_placement(program, hw, rng, jitter=jitters[i % len(jitters)]))
    return out


def neighbours(placement: dict[int, int], hw: nx.Graph, dist: dict, radius: int = 2):
    """Placements one swap-or-move away, restricted to nearby physical qubits."""
    occ = {p: l for l, p in placement.items()}
    seen = set()
    for l, p in placement.items():
        for q, d in dist[p].items():
            if d == 0 or d > radius:
                continue
            m = occ.get(q)
            if m is not None:
                key = (min(l, m), max(l, m))
                if key in seen:
                    continue
                seen.add(key)
                nb = dict(placement)
                nb[l], nb[m] = q, p
            else:
                nb = dict(placement)
                nb[l] = q
            yield nb


def perturb(placement: dict[int, int], rng: random.Random, k: int = 2) -> dict[int, int]:
    pl = dict(placement)
    logicals = list(pl)
    for _ in range(k):
        a, b = rng.sample(logicals, 2)
        pl[a], pl[b] = pl[b], pl[a]
    return pl


def local_search(evaluator: Evaluator, hw: nx.Graph, start: dict[int, int],
                 start_cost: float, deadline: float, rng: random.Random,
                 on_improve=None):
    """Iterated first-improvement local search over shuffled neighbour batches. Returns (cost, placement, ops)."""
    dist = dict(nx.all_pairs_shortest_path_length(hw))
    best_cost, best_pl, best_ops = start_cost, dict(start), None
    cur_cost, cur_pl = start_cost, dict(start)
    while time.monotonic() < deadline:
        nbs = list(neighbours(cur_pl, hw, dist))
        if not nbs:
            break
        rng.shuffle(nbs)
        moved = False
        for i in range(0, len(nbs), evaluator.batch):
            if time.monotonic() >= deadline:
                break
            chunk = nbs[i:i + evaluator.batch]
            results = evaluator(chunk)
            j = min(range(len(chunk)), key=lambda x: results[x][0])
            c, ops = results[j]
            if c < cur_cost:
                cur_cost, cur_pl, moved = c, chunk[j], True
                if c < best_cost:
                    best_cost, best_pl, best_ops = c, dict(chunk[j]), ops
                    if on_improve is not None:
                        on_improve(best_pl, best_ops)
                break
        if moved:
            continue
        # Local optimum: kick from the incumbent and keep going.
        cur_pl = perturb(best_pl, rng, k=rng.choice((2, 3)))
        cur_cost = evaluator([cur_pl])[0][0]
    return best_cost, best_pl, best_ops


def search_placements(program, hw, deadline: float, rng: random.Random,
                      n_samples: int = 400, n_starts: int = 3, offer=None,
                      extra: list[tuple[str, dict[int, int]]] = (),
                      workers: int | None = None):
    """Sample, then local-search. Calls offer(placement, ops, tag) on finds.

    Returns the placements sorted by score, best first, as (cost, tag, placement).
    """
    t_start = time.monotonic()
    ev = Evaluator(program, hw, workers, allow_spawn=deadline - t_start > 6.0)
    try:
        tagged = list(extra) + [(f"sample{i}", pl)
                                for i, pl in enumerate(sample_placements(program, hw, rng, n_samples))]
        # Sample in batches so a slow instance still respects the deadline.
        sample_deadline = t_start + 0.4 * (deadline - t_start)
        scored = []
        batch = 64
        for i in range(0, len(tagged), batch):
            if time.monotonic() > sample_deadline and scored:
                break
            chunk = tagged[i:i + batch]
            for (tag, pl), (c, ops) in zip(chunk, ev([pl for _, pl in chunk])):
                if ops is None:
                    continue
                if offer is not None and (not scored or c < min(x[0] for x in scored)):
                    offer(pl, ops, f"sample/{tag}")
                scored.append((c, tag, pl))
        scored.sort(key=lambda x: x[0])

        # Local search from the best distinct placements, splitting the time.
        starts, keys = [], set()
        for c, tag, pl in scored:
            k = tuple(sorted(pl.items()))
            if k not in keys:
                keys.add(k)
                starts.append((c, tag, pl))
            if len(starts) >= n_starts:
                break
        found = []
        for j, (c, tag, pl) in enumerate(starts):
            now = time.monotonic()
            share = (deadline - now) / (len(starts) - j)
            if share <= 0:
                break
            cb = (lambda p, o, t=tag: offer(p, o, f"ls/{t}")) if offer is not None else None
            lc, lpl, _ = local_search(ev, hw, pl, c, now + share, rng, on_improve=cb)
            found.append((lc, f"ls/{tag}", lpl))
        return sorted(found + scored, key=lambda x: x[0])
    finally:
        ev.close()


# -- polish ----------------------------------------------------------------


def polish(problem: Problem, placements: list[dict[int, int]], deadline: float,
           incumbent: float = INF, on_improve=None):
    """Re-route the best placements under many beam settings; the beam is
    non-monotone in width and weight, so a different setting often finds a
    cheaper routing from the same start."""
    best = incumbent
    settings = [(w, mp_, wt) for w in (4, 8, 12, 24, 32, 64, 128, 256)
                for mp_ in (6, 12, 24) for wt in (0.4, 0.6, 1.0)]
    for w, mp_, wt in settings:            # breadth first across placements
        for pl in placements:
            if time.monotonic() > deadline:
                return best
            s = beam_search(problem, problem.initial(pl), width=w, max_paths=mp_, weight=wt,
                            incumbent=best, deadline=deadline)
            if s is not None and s.cost < best:
                best = s.cost
                if on_improve is not None:
                    on_improve(pl, s.ops())
    return best
