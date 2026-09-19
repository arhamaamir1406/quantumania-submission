"""Portfolio solver: run every strategy, self-score, return the best.

Because the scorer is available and cheap, there is no reason to ever ship a
strategy that loses on a given instance. The baseline is kept in the portfolio
as a safety net, so the result can never be worse than it, and never invalid.
"""
from __future__ import annotations

import random
import time

import networkx as nx

from starter_kit.baseline_routing import solve as baseline_solve

from .ir import evaluate, materialize
from .mdp import Problem
from .placement import (constructive_placement, embed_placement,
                        identity_placement, random_placement)
from .search import beam_search, greedy_rollout

INF = float("inf")


class Candidate:
    __slots__ = ("score", "placement", "routed", "tag")

    def __init__(self, score, placement, routed, tag):
        self.score, self.placement, self.routed, self.tag = score, placement, routed, tag


def solve(program: list[tuple], hardware_graph: nx.Graph, budget: float = 10.0,
          seeds: int = 48, beam_width: int = 1200, verbose: bool = False):
    """Returns (initial_placement, routed_program)."""
    t0 = time.monotonic()
    rng = random.Random(0xC0FFEE)
    problem = Problem(program, hardware_graph)
    best: Candidate | None = None

    def offer(placement, ops, tag):
        nonlocal best
        score, routed, msg = evaluate(program, hardware_graph, placement, ops)
        if score == INF:
            if verbose:
                print(f"    [{tag}] INVALID: {msg}")
            return
        if best is None or score < best.score:
            best = Candidate(score, dict(placement), routed, tag)
            if verbose:
                print(f"    [{tag}] {score:.1f}  <- best")
        elif verbose:
            print(f"    [{tag}] {score:.1f}")

    # -- safety net -------------------------------------------------------
    bpl, brouted = baseline_solve(list(program), hardware_graph)
    from starter_kit.scorer import core_score, validate_routed_program
    ok, _ = validate_routed_program(program, hardware_graph, bpl, brouted)
    if ok:
        c = Candidate(core_score(brouted), bpl, brouted, "baseline")
        best = c
        if verbose:
            print(f"    [baseline] {c.score:.1f}")

    # -- zero-SWAP embedding ---------------------------------------------
    emb = embed_placement(program, hardware_graph)
    if emb is not None:
        s = problem.initial(emb)
        if problem.is_terminal(s):
            offer(emb, s.ops(), "embed")

    # -- rollouts from a spread of placements -----------------------------
    placements = [("constructive", constructive_placement(program, hardware_graph)),
                  ("identity", identity_placement(program, hardware_graph))]
    if emb is not None:
        placements.append(("embed-seed", emb))
    for i in range(seeds):
        if time.monotonic() - t0 > budget * 0.55:
            break
        if i % 3 == 0:
            p = constructive_placement(program, hardware_graph, rng, jitter=1.5)
            placements.append((f"constructive-j{i}", p))
        else:
            placements.append((f"random{i}", random_placement(program, hardware_graph, rng)))

    scored_placements = []
    for tag, pl in placements:
        if time.monotonic() - t0 > budget * 0.7:
            break
        s = greedy_rollout(problem, problem.initial(pl), rng=rng,
                           noise=0.0 if tag.startswith(("constructive", "identity", "embed")) else 0.15)
        if s is None:
            continue
        scored_placements.append((s.cost, tag, pl))
        offer(pl, s.ops(), f"rollout/{tag}")

    # -- beam search from the most promising placements -------------------
    scored_placements.sort(key=lambda x: x[0])
    for _, tag, pl in scored_placements[:3]:
        if time.monotonic() - t0 > budget:
            break
        s = beam_search(problem, problem.initial(pl), width=beam_width,
                        incumbent=best.score if best else INF)
        if s is not None:
            offer(pl, s.ops(), f"beam/{tag}")

    assert best is not None, "portfolio produced no valid candidate"
    return best.placement, best.routed
