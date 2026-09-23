"""Policies and search over the routing MDP.

Branching is on *move sets*, not single SWAPs: for the active gate we commit
to a whole shortest path plus a meeting point along it. That guarantees the
gate executes (a single-SWAP greedy with a one-gate front layer has no
progress guarantee and wanders until it hits a step cap), and it exposes the
meet-in-the-middle choice directly -- splitting the walk across both endpoints
puts the SWAPs on disjoint qubits, so the ASAP scheduler packs them into
shared layers.

Successor scoring is f(t) = t.cost + V(t), where t.cost is the exact partial
objective (swaps + 0.5 * depth so far) and V is either the hand-written
lookahead heuristic or the learned GraphSAGE value function.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable

import networkx as nx

from .mdp import Problem, State

INF = float("inf")
ValueFn = Callable[[list[State]], list[float]]


def gate_moves(problem: Problem, s: State, max_paths: int = 12) -> list[State]:
    """Successors that each execute the active gate, one per (path, split)."""
    _, a, b = problem.ops[s.k]
    pa, pb = s.pos[a], s.pos[b]
    out: list[State] = []
    seen: set[tuple] = set()
    paths = []
    for path in nx.all_shortest_paths(problem.hw, pa, pb):
        paths.append(path)
        if len(paths) >= max_paths:
            break
    for path in paths:
        d = len(path) - 1
        for split in range(d):              # forward swaps taken from pa's end
            t = s
            for j in range(split):
                t = problem.apply_swap_raw(t, path[j], path[j + 1])
            for j in range(d, split + 1, -1):
                t = problem.apply_swap_raw(t, path[j], path[j - 1])
            t = problem.advance(t)
            key = (t.k, t.pos, t.depth)
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return out


def pre_swaps(problem: Problem, s: State, lookahead: int = 6,
              free_only: bool = False) -> list[tuple[int, int]]:
    """Pre-positioning SWAPs: each shortens one of the next `lookahead` gates
    (after the active one) and leaves the active gate's qubits alone.

    The narrow move set only ever moves qubits for the active gate, along its
    shortest paths. These let the search start moving qubits for later gates
    first -- a move that changes the mapping the intervening gates see, and
    that the ASAP scheduler can often slot into idle layers. With `free_only`,
    keep just the SWAPs that fit under the current depth (no depth cost).
    """
    _, a, b = problem.ops[s.k]
    active = {s.pos[a], s.pos[b]}
    dist, nbrs = problem.dist, problem.nbrs
    out: set[tuple[int, int]] = set()
    for g in range(s.k + 1, min(problem.n_ops, s.k + 1 + lookahead)):
        op = problem.ops[g]
        if op[0] != "2Q":
            continue
        pu, pv = s.pos[op[1]], s.pos[op[2]]
        d = dist[pu][pv]
        if d <= 1:
            continue
        for p, q in ((pu, pv), (pv, pu)):
            if p in active:
                continue
            for n in nbrs[p]:
                if n in active or dist[n][q] >= d:
                    continue
                if free_only and 1 + max(s.tau[p], s.tau[n]) > s.depth:
                    continue
                out.add((p, n) if p < n else (n, p))
    return sorted(out)


def wide_moves(problem: Problem, s: State, max_paths: int = 12, lookahead: int = 6,
               free_only: bool = False) -> list[State]:
    """gate_moves, plus gate_moves after each single pre-positioning SWAP.

    One pre-SWAP per round; chains of them build up over successive rounds.
    """
    out = gate_moves(problem, s, max_paths)
    seen = {(t.k, t.pos, t.depth) for t in out}
    for p, q in pre_swaps(problem, s, lookahead, free_only):
        for t in gate_moves(problem, problem.apply_swap_raw(s, p, q), max_paths):
            key = (t.k, t.pos, t.depth)
            if key not in seen:
                seen.add(key)
                out.append(t)
    return out


def moves(problem: Problem, s: State, max_paths: int, lookahead: int,
          free_only: bool = False) -> list[State]:
    if lookahead > 0:
        return wide_moves(problem, s, max_paths, lookahead, free_only)
    return gate_moves(problem, s, max_paths)


def _score(problem: Problem, states: list[State], value_fn: ValueFn | None,
           window: int, weight: float) -> list[float]:
    if value_fn is not None:
        return [s.cost + v for s, v in zip(states, value_fn(states))]
    return [s.cost + problem.heuristic(s, window, weight) for s in states]


def greedy_rollout(problem: Problem, s: State, rng: random.Random | None = None,
                   noise: float = 0.0, window: int = 12, weight: float = 0.6,
                   max_paths: int = 12, value_fn: ValueFn | None = None,
                   lookahead: int = 0) -> State | None:
    """Constrained SABRE over move sets. Terminates in exactly n_gates steps."""
    while not problem.is_terminal(s):
        succ = moves(problem, s, max_paths, lookahead)
        if not succ:
            return None
        scores = _score(problem, succ, value_fn, window, weight)
        if noise and rng is not None:
            scores = [x + rng.random() * noise for x in scores]
        s = succ[min(range(len(succ)), key=scores.__getitem__)]
    return s


def beam_search(problem: Problem, s0: State, width: int = 1500, window: int = 12,
                weight: float = 0.6, max_paths: int = 12, incumbent: float = INF,
                value_fn: ValueFn | None = None, deadline: float | None = None,
                lookahead: int = 0, free_only: bool = False,
                prune: int = 0, prune_fn: ValueFn | None = None,
                keep_narrow: bool = True) -> State | None:
    """Beam search over move sets, with a (gate, mapping) transposition table.

    One round per program gate, so search depth is bounded and known. With a
    `deadline` (time.monotonic()), gives up when it passes and returns None --
    a partial beam holds no complete solution worth returning.

    Pruning (`prune` > 0 with a `prune_fn`, the policy net's cost-to-go): the
    wide move set gives each parent 5-6x more successors, and the hand-written
    heuristic misjudges the pre-positioning ones. The policy scores every
    successor (step cost + predicted cost-to-go) and each parent keeps only its
    `prune` best -- plus, with `keep_narrow`, all of its ordinary moves, so
    pruning can only add candidates to the narrow search, never remove them.
    """
    if problem.is_terminal(s0):
        return s0
    pruning = prune > 0 and prune_fn is not None and lookahead > 0
    beam = [s0]
    seen: dict[tuple, float] = {}
    best_final: State | None = None
    best_cost = incumbent

    while beam:
        if deadline is not None and time.monotonic() > deadline:
            return None
        pending: list[State] = []
        keys: set[tuple] = set()
        if pruning:
            expansions = _pruned(problem, beam, max_paths, lookahead, free_only,
                                 prune, prune_fn, keep_narrow)
        else:
            expansions = [moves(problem, s, max_paths, lookahead, free_only) for s in beam]
        for succ in expansions:
            for t in succ:
                if t.cost >= best_cost:
                    continue
                if problem.is_terminal(t):
                    best_cost, best_final = t.cost, t
                    continue
                key = (t.k, t.pos)
                if key in keys:
                    continue
                keys.add(key)
                pending.append(t)
        if not pending:
            break
        scores = _score(problem, pending, value_fn, window, weight)
        scored = []
        for t, sc in zip(pending, scores):
            key = (t.k, t.pos)
            prev = seen.get(key)
            if prev is not None and prev <= sc:
                continue
            seen[key] = sc
            scored.append((sc, t))
        scored.sort(key=lambda x: x[0])
        beam = [t for _, t in scored[:width]]

    return best_final


def _pruned(problem: Problem, beam: list[State], max_paths: int, lookahead: int,
            free_only: bool, prune: int, prune_fn: ValueFn,
            keep_narrow: bool) -> list[list[State]]:
    """Per parent: its narrow moves (optionally) + the policy's top `prune` others.

    All successors of the round are scored in one batch -- one GPU call per
    round, not one per parent.
    """
    narrow = [gate_moves(problem, s, max_paths) for s in beam]
    extra = []
    for s, nm in zip(beam, narrow):
        keys = {(t.k, t.pos) for t in nm}
        succ = wide_moves(problem, s, max_paths, lookahead, free_only)
        extra.append([t for t in succ if (t.k, t.pos) not in keys])
    pool = [t for nm, ex in zip(narrow, extra) for t in (ex if keep_narrow else nm + ex)]
    vals = prune_fn(pool) if pool else []
    out, i = [], 0
    for nm, ex in zip(narrow, extra):
        cands = ex if keep_narrow else nm + ex
        q = [t.cost + v for t, v in zip(cands, vals[i:i + len(cands)])]
        i += len(cands)
        order = sorted(range(len(cands)), key=q.__getitem__)[:prune]
        kept = [cands[j] for j in order]
        out.append(nm + kept if keep_narrow else kept)
    return out


def path_states(problem: Problem, placement: dict[int, int], ops: list[tuple]) -> list[State]:
    """Replay a solution, returning the state after every SWAP (training data)."""
    s = problem.initial(placement)
    out = [s]
    for op in ops:
        if op[0] == "SWAP":
            s = problem.advance(problem.apply_swap_raw(s, op[1], op[2]))
            out.append(s)
    return out
