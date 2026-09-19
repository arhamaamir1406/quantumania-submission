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


def _score(problem: Problem, states: list[State], value_fn: ValueFn | None,
           window: int, weight: float) -> list[float]:
    if value_fn is not None:
        return [s.cost + v for s, v in zip(states, value_fn(states))]
    return [s.cost + problem.heuristic(s, window, weight) for s in states]


def greedy_rollout(problem: Problem, s: State, rng: random.Random | None = None,
                   noise: float = 0.0, window: int = 12, weight: float = 0.6,
                   max_paths: int = 12, value_fn: ValueFn | None = None) -> State | None:
    """Constrained SABRE over move sets. Terminates in exactly n_gates steps."""
    while not problem.is_terminal(s):
        moves = gate_moves(problem, s, max_paths)
        if not moves:
            return None
        scores = _score(problem, moves, value_fn, window, weight)
        if noise and rng is not None:
            scores = [x + rng.random() * noise for x in scores]
        s = moves[min(range(len(moves)), key=scores.__getitem__)]
    return s


def beam_search(problem: Problem, s0: State, width: int = 1500, window: int = 12,
                weight: float = 0.6, max_paths: int = 12, incumbent: float = INF,
                value_fn: ValueFn | None = None) -> State | None:
    """Beam search over move sets, with a (gate, mapping) transposition table.

    One round per program gate, so search depth is bounded and known.
    """
    if problem.is_terminal(s0):
        return s0
    beam = [s0]
    seen: dict[tuple, float] = {}
    best_final: State | None = None
    best_cost = incumbent

    while beam:
        pending: list[State] = []
        keys: set[tuple] = set()
        for s in beam:
            for t in gate_moves(problem, s, max_paths):
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


def path_states(problem: Problem, placement: dict[int, int], ops: list[tuple]) -> list[State]:
    """Replay a solution, returning the state after every SWAP (training data)."""
    s = problem.initial(placement)
    out = [s]
    for op in ops:
        if op[0] == "SWAP":
            s = problem.advance(problem.apply_swap_raw(s, op[1], op[2]))
            out.append(s)
    return out
