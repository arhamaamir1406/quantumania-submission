"""Policies and search over the routing MDP.

Branching is on *move sets*, not single SWAPs: for the active gate we commit
to a whole shortest path plus a meeting point along it. That guarantees the
gate executes (a single-SWAP greedy with a one-gate front layer has no
progress guarantee and wanders), and it exposes the meet-in-the-middle choice
directly -- splitting the walk across both endpoints puts the SWAPs on
disjoint qubits, so the ASAP scheduler packs them into shared layers.
"""
from __future__ import annotations

import random

import networkx as nx

from .mdp import Problem, State

INF = float("inf")


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
        for split in range(d):          # forward swaps taken from pa's end
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


def _f(problem: Problem, s: State, window: int, weight: float) -> float:
    return s.cost + problem.heuristic(s, window, weight)


def greedy_rollout(problem: Problem, s: State, rng: random.Random | None = None,
                   noise: float = 0.0, window: int = 12, weight: float = 0.6,
                   max_paths: int = 12) -> State | None:
    """Constrained SABRE over move sets. Terminates in exactly n_gates steps."""
    while not problem.is_terminal(s):
        best, best_score = None, INF
        for t in gate_moves(problem, s, max_paths):
            score = _f(problem, t, window, weight)
            if noise and rng is not None:
                score += rng.random() * noise
            if score < best_score:
                best, best_score = t, score
        if best is None:
            return None
        s = best
    return s


def beam_search(problem: Problem, s0: State, width: int = 1500, window: int = 12,
                weight: float = 0.6, max_paths: int = 12,
                incumbent: float = INF) -> State | None:
    """Beam search over move sets, with a (gate, mapping) transposition table.

    One round per program gate, so the search depth is bounded and known.
    """
    if problem.is_terminal(s0):
        return s0
    beam = [s0]
    seen: dict[tuple, float] = {}
    best_final: State | None = None
    best_cost = incumbent

    while beam:
        scored: list[tuple[float, State]] = []
        for s in beam:
            for t in gate_moves(problem, s, max_paths):
                if t.cost >= best_cost:
                    continue
                if problem.is_terminal(t):
                    best_cost, best_final = t.cost, t
                    continue
                key = (t.k, t.pos)
                score = _f(problem, t, window, weight)
                prev = seen.get(key)
                if prev is not None and prev <= score:
                    continue
                seen[key] = score
                scored.append((score, t))
        scored.sort(key=lambda x: x[0])
        beam = [t for _, t in scored[:width]]

    return best_final
