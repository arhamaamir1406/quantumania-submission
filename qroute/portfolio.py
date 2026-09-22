"""Portfolio solver: run every strategy, self-score, return the best.

Because the scorer is available and cheap, there is no reason to ever ship a
strategy that loses on a given instance. The baseline is kept in the portfolio
as a safety net, so the result can never be worse than it, and never invalid.

The learned value function is one portfolio member among several, and it is
admitted on exactly the same terms as the rest: it wins an instance only if it
self-scores better. A regression in the net can cost search time; it can never
cost score.
"""
from __future__ import annotations

import os
import random
import time

import networkx as nx

from starter_kit.baseline_routing import solve as baseline_solve

from .bounds import score_lower_bound
from .ir import evaluate
from .mdp import Problem
from .placement import embed_placement, identity_placement
from .placement_search import search_placements
from .search import beam_search, greedy_rollout

INF = float("inf")

# Checkpoints are tried in order; the first that loads wins. The small net is
# listed first on CPU because an 18M-parameter forward pass over a 1200-wide
# beam does not fit in a 10-second budget without a GPU.
CHECKPOINTS_GPU = ("models/value_large.pt", "models/value_base.pt", "models/value_small.pt")
CHECKPOINTS_CPU = ("models/value_small.pt", "models/value_tiny.pt", "models/value_base.pt")


class Candidate:
    __slots__ = ("score", "placement", "routed", "tag")

    def __init__(self, score, placement, routed, tag):
        self.score, self.placement, self.routed, self.tag = score, placement, routed, tag


_NET_CACHE: dict = {}


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_value_net(device: str, paths: tuple[str, ...] | None = None):
    """Load the best available checkpoint for `device`, or None."""
    key = (device, paths)
    if key in _NET_CACHE:
        return _NET_CACHE[key]
    net = None
    if paths is None:
        paths = CHECKPOINTS_GPU if device.startswith("cuda") else CHECKPOINTS_CPU
    try:
        from .gnn import load_net
        for p in paths:
            if os.path.exists(p):
                net = load_net(p, device)
                if net is not None:
                    break
    except Exception:
        net = None
    _NET_CACHE[key] = net
    return net


def solve(program: list[tuple], hardware_graph: nx.Graph, budget: float = 10.0,
          seeds: int = 48, beam_width: int = 1200, verbose: bool = False,
          net=None, net_device: str | None = None, net_quantile: int | None = None,
          net_beam: int | None = None, net_paths: tuple[str, ...] | None = None,
          use_net: bool = True, placement_share: float = 0.75,
          workers: int | None = None):
    """Returns (initial_placement, routed_program)."""
    t0 = time.monotonic()
    rng = random.Random(0xC0FFEE)
    problem = Problem(program, hardware_graph)
    best: Candidate | None = None

    def left():
        return budget - (time.monotonic() - t0)

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
        best = Candidate(core_score(brouted), bpl, brouted, "baseline")
        if verbose:
            print(f"    [baseline] {best.score:.1f}")

    # -- zero-SWAP embedding ---------------------------------------------
    emb = embed_placement(program, hardware_graph)
    if emb is not None:
        s = problem.initial(emb)
        if problem.is_terminal(s):
            offer(emb, s.ops(), "embed")

    # A candidate on the provable floor cannot be beaten; stop spending budget.
    floor = score_lower_bound(program, hardware_graph)[0]
    if best is not None and best.score <= floor:
        return best.placement, best.routed

    # -- placement search: sample widely, then local-search the best -------
    # The starting placement is the main lever on the score; routing from a
    # fixed placement saturates. See placement_search.py.
    extra = [("identity", identity_placement(program, hardware_graph))]
    if emb is not None:
        extra.append(("embed-seed", emb))
    scored_placements = search_placements(
        program, hardware_graph, deadline=t0 + budget * placement_share, rng=rng,
        offer=offer, extra=extra, workers=workers)

    # -- beam search from the most promising placements -------------------
    scored_placements.sort(key=lambda x: x[0])
    for _, tag, pl in scored_placements[:3]:
        if time.monotonic() - t0 > budget:
            break
        s = beam_search(problem, problem.initial(pl), width=beam_width,
                        incumbent=best.score if best else INF, deadline=t0 + budget)
        if s is not None:
            offer(pl, s.ops(), f"beam/{tag}")
        if time.monotonic() - t0 > budget:
            break

    # -- learned value function ------------------------------------------
    if use_net and scored_placements:
        device = _pick_device(net_device)
        if net is None:
            net = load_value_net(device, net_paths)
        if net is not None:
            try:
                from .gnn import make_value_fn
                vf = make_value_fn(net, problem, device=device, quantile=net_quantile,
                                   max_batch=1024 if device.startswith("cuda") else 256)
                # An 18M-parameter net is ~3ms/state on CPU and ~30us on a GPU,
                # so the affordable beam width differs by two orders of magnitude.
                width = net_beam if net_beam is not None else (
                    beam_width if device.startswith("cuda") else max(16, beam_width // 24))
                for _, tag, pl in scored_placements[:2]:
                    if left() <= 0:
                        break
                    s = beam_search(problem, problem.initial(pl), width=width,
                                    incumbent=best.score if best else INF, value_fn=vf,
                                    deadline=t0 + budget)
                    if s is not None:
                        offer(pl, s.ops(), f"net-beam/{tag}")
                # A greedy rollout under the net is cheap and sometimes escapes
                # a beam that the incumbent pruned too aggressively.
                if left() > 0 and scored_placements:
                    _, tag, pl = scored_placements[0]
                    s = greedy_rollout(problem, problem.initial(pl), value_fn=vf)
                    if s is not None:
                        offer(pl, s.ops(), f"net-greedy/{tag}")
            except Exception as exc:                       # never let the net break a run
                if verbose:
                    print(f"    [net] skipped: {exc}")

    assert best is not None, "portfolio produced no valid candidate"
    return best.placement, best.routed
