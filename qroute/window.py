"""Window LNS: re-solve a window of gates exactly, re-route the rest.

The move-set search is converged on large instances, and the exact model
(exact.py) is too big to solve whole beyond ~24 gates. A window of W gates is
small enough: fix the mapping where the window starts, forbid each physical
qubit until the incumbent's schedule frees it, solve the window in the
full-freedom move space, then re-route everything after it with the beam and
keep the result only if the scorer's total improves.
"""
from __future__ import annotations

import random
import time

from .ir import evaluate
from .mdp import Problem, State
from .search import beam_search


def _replay(problem: Problem, placement, ops) -> list[State]:
    """States at gate-round boundaries along a solution."""
    s = problem.initial(placement)
    cuts = [s]
    for op in ops:
        if op[0] == "SWAP":
            s = problem.advance(problem.apply_swap_raw(s, op[1], op[2]))
            if s.k != cuts[-1].k:
                cuts.append(s)
    return cuts


def window_lns(program, hw, placement, ops, deadline: float, rng: random.Random,
               window: int = 10, extra_layers: int = 3, step_limit: float = 8.0,
               on_improve=None, verbose: bool = False):
    """Returns (score, placement, ops) -- the best found, never worse."""
    from .exact import solve_exact

    problem = Problem(program, hw)
    best_score = evaluate(program, hw, placement, ops)[0]
    best_ops = list(ops)
    logicals = problem.logicals
    tried = 0
    while time.monotonic() < deadline - 0.5:
        cuts = [c for c in _replay(problem, placement, best_ops) if c.k < problem.n_ops]
        c = rng.choice(cuts)
        a = c.k
        b = min(problem.n_ops, a + window)
        sub = [program[k] for k in range(a, b)]
        if any(op[0] != "2Q" for op in sub):
            sub = [op for op in sub if op[0] == "2Q"]
        pos = {logicals[i]: p for i, p in enumerate(c.pos)}
        # Window time starts at the earliest free layer among its qubits.
        base = min(c.tau)
        busy = {p: max(0, t - base) for p, t in enumerate(c.tau) if t > base}
        # How many layers the incumbent spends on this window, plus slack.
        T = max(1, _window_layers(problem, c, best_ops, b) + extra_layers)
        r = solve_exact(sub, hw, T=T, strict=False, lazy=True, initial=pos, busy=busy,
                        all_logicals=logicals,
                        time_limit=min(step_limit, deadline - time.monotonic()))
        tried += 1
        if r["ops"] is None:
            continue
        # Splice: window ops (sub-program indices -> global), then beam the rest.
        s = c
        for op in r["ops"]:
            if op[0] == "SWAP":
                s = problem.advance(problem.apply_swap_raw(s, op[1], op[2]))
        if s.k < b:
            continue                               # window not fully executed
        tail = beam_search(problem, s, width=32) if s.k < problem.n_ops else s
        if tail is None:
            continue
        cand = tail.ops()
        sc = evaluate(program, hw, placement, cand)[0]
        if sc < best_score:
            if verbose:
                print(f"    [window] gates {a}-{b - 1}: {best_score:.1f} -> {sc:.1f}")
            best_score, best_ops = sc, cand
            if on_improve is not None:
                on_improve(placement, cand)
    return best_score, placement, best_ops


def _window_layers(problem: Problem, c: State, ops, b: int) -> int:
    """Layers the incumbent uses between state c and gate b executing."""
    s = c
    start = min(c.tau)
    # Walk the incumbent's remaining SWAPs from c until gate b is reached.
    done = c.ops()
    rest = ops[len(done):]
    for op in rest:
        if s.k >= b:
            break
        if op[0] == "SWAP":
            s = problem.advance(problem.apply_swap_raw(s, op[1], op[2]))
    return max(s.tau) - start
