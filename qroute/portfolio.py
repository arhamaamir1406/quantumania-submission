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
from .placement_search import polish, search_placements
from .search import beam_search, greedy_rollout

INF = float("inf")

# Checkpoints are tried in order; the first that loads wins. The small net is
# listed first on CPU because an 18M-parameter forward pass over a 1200-wide
# beam does not fit in a 10-second budget without a GPU.
CHECKPOINTS_GPU = ("models/value_large.pt", "models/value_base.pt", "models/value_small.pt")
CHECKPOINTS_CPU = ("models/value_small.pt", "models/value_tiny.pt", "models/value_base.pt")


def _to_ir(routed: list[tuple]) -> list[tuple]:
    """Physical routed program -> IR ops (SWAPs kept, program ops indexed)."""
    out, k = [], 0
    for op in routed:
        if op[0] == "SWAP":
            out.append(op)
        else:
            out.append(("GATE", k))
            k += 1
    return out


class _Optimal(Exception):
    """Raised by offer() when a candidate reaches the provable floor."""


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


def _policy_fn(problem, path, device, verbose):
    """The pruning policy's cost-to-go function, or None if unavailable."""
    try:
        from .gnn import load_net, make_value_fn
        dev = _pick_device(device)
        net = load_net(path, dev)
        if net is None:
            return None
        return make_value_fn(net, problem, device=dev,
                             max_batch=2048 if dev.startswith("cuda") else 256)
    except Exception as exc:                  # torch missing, bad checkpoint, ...
        if verbose:
            print(f"    [policy] skipped: {exc}")
        return None


def solve(program: list[tuple], hardware_graph: nx.Graph, budget: float = 10.0,
          seeds: int = 48, beam_width: int = 1200, verbose: bool = False,
          net=None, net_device: str | None = None, net_quantile: int | None = None,
          net_beam: int | None = None, net_paths: tuple[str, ...] | None = None,
          use_net: bool = True, placement_share: float = 0.65,
          polish_share: float = 0.9, workers: int | None = None, seed: int = 0xC0FFEE,
          use_exact: bool = True, exact_max_gates: int = 24, exact_share: float = 0.5,
          use_window: bool = False, window_share: float = 0.4, window_size: int = 12,
          use_sabre: bool = False, sabre_share: float = 0.08,
          use_policy: bool = True, policy_path: str = "models/policy_small_r2.pt",
          policy_share: float = 0.5, policy_prune: int = 4, policy_lookahead: int = 4):
    """Returns (initial_placement, routed_program)."""
    t0 = time.monotonic()
    rng = random.Random(seed)
    problem = Problem(program, hardware_graph)
    best: Candidate | None = None
    floor = score_lower_bound(program, hardware_graph)[0]
    # Hand the back part of the budget to the exact model: whole-instance on
    # small programs, window LNS on large ones.
    if use_exact:
        try:
            from . import exact
            if exact.available():
                small = sum(1 for op in program if op[0] == "2Q") <= exact_max_gates
                share = exact_share if small else (window_share if use_window else 0.0)
                placement_share *= 1 - share
                polish_share *= 1 - share
        except ImportError:
            pass

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
            if score <= floor:
                raise _Optimal
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

    # A candidate on the provable floor cannot be beaten: offer() raises
    # _Optimal and the remaining stages are skipped.
    try:
        # -- zero-SWAP embedding ---------------------------------------------
        emb = embed_placement(program, hardware_graph)
        if emb is not None:
            s = problem.initial(emb)
            if problem.is_terminal(s):
                offer(emb, s.ops(), "embed")

        # -- placement search: sample widely, then local-search the best -------
        # The starting placement is the main lever on the score; routing from a
        # fixed placement saturates. See placement_search.py.
        extra = [("identity", identity_placement(program, hardware_graph))]
        if emb is not None:
            extra.append(("embed-seed", emb))
        if use_sabre:
            # Qiskit SABRE's initial layouts as extra seeds (optional dependency).
            from . import sabre_seed
            if sabre_seed.available():
                for i, pl in enumerate(sabre_seed.sabre_placements(
                        program, hardware_graph, time.monotonic() + budget * sabre_share,
                        seed=rng.randrange(1 << 30))):
                    extra.append((f"sabre{i}", pl))
        scored_placements = search_placements(
            program, hardware_graph, deadline=t0 + budget * placement_share, rng=rng,
            offer=offer, extra=extra, workers=workers)

        # -- polish: re-route the best placements under many beam settings -----
        # The beam is non-monotone in width and weight, so a different setting
        # often routes the same placement more cheaply. This replaced a single
        # width-1200 beam, which never beat the placement search's own result.
        scored_placements.sort(key=lambda x: x[0])
        top, keys = [], set()
        for _, tag, pl in scored_placements:
            k = tuple(sorted(pl.items()))
            if k not in keys:
                keys.add(k)
                top.append((tag, pl))
            if len(top) >= 12:
                break
        tags = {tuple(sorted(pl.items())): tag for tag, pl in top}
        polish_end = t0 + budget * polish_share
        pvf = None
        if use_policy:
            pvf = _policy_fn(problem, policy_path, net_device, verbose)
            if pvf is not None:
                # Split the polish window: plain polish first, then the pruned
                # wide-move search on the same placements.
                start = t0 + budget * placement_share
                polish_end = start + (polish_end - start) * (1 - policy_share)
        polish(problem, [pl for _, pl in top], deadline=polish_end,
               incumbent=best.score if best else INF,
               on_improve=lambda pl, ops: offer(pl, ops, f"polish/{tags[tuple(sorted(pl.items()))]}"))

        # -- policy-pruned wide search: pre-positioning moves ----------------
        # The wide move set adds SWAPs that start moving qubits for the next
        # few gates. Under the hand-written heuristic it loses (that heuristic
        # overrates those moves and they crowd the beam), pruned or not. The
        # policy net both prunes (each parent keeps its best few) and ranks.
        if pvf is not None:
            end = t0 + budget * polish_share
            for w in (8, 16, 32, 64, 128, 256):
                for tag, pl in top[:6]:
                    if time.monotonic() > end:
                        break
                    s = beam_search(problem, problem.initial(pl), width=w,
                                    incumbent=best.score if best else INF,
                                    lookahead=policy_lookahead, prune=policy_prune,
                                    prune_fn=pvf, value_fn=pvf, deadline=end)
                    if s is not None:
                        offer(pl, s.ops(), f"policy{w}/{tag}")

        # -- exact layered model (CP-SAT), seeded with the incumbent --------
        # A different search space: any SWAP on any edge in any layer. It
        # finds trade-offs the move-set search cannot express (qaoa_random:
        # 6 SWAPs/depth 12 -> 7 SWAPs/depth 9). Optional, like torch.
        if use_exact and best is not None and left() > 2:
            try:
                from . import exact
                n2q = sum(1 for op in program if op[0] == "2Q")
                if exact.available() and n2q <= exact_max_gates:
                    ops = _to_ir(best.routed)
                    _, T = exact.hint_layers(program, hardware_graph, best.placement, ops,
                                             monotone=False)
                    # Room for trade-offs that spend depth to save SWAPs, capped
                    # by the deepest solution that could still beat the incumbent.
                    s_floor = score_lower_bound(program, hardware_graph)[1]
                    T = max(1, min(T + 2, int(2 * (best.score - 0.5 - s_floor))))
                    r = exact.solve_exact(program, hardware_graph, T=T, strict=False,
                                          time_limit=max(1.0, left() - 0.5),
                                          hint=(best.placement, ops))
                    if r["ops"] is not None:
                        offer(r["placement"], r["ops"], "exact")
            except _Optimal:
                raise
            except Exception as exc:                   # never let the exact stage break a run
                if verbose:
                    print(f"    [exact] skipped: {exc}")

        # -- window LNS: exact windows + beam re-route, for large instances --
        # Opt-in: at a 60 s budget it loses to spending that time on
        # placement search + polish (A/B on dense_random: 0 wins in 4 seeds).
        if use_window and use_exact and best is not None and left() > 3:
            try:
                from . import exact
                n2q = sum(1 for op in program if op[0] == "2Q")
                if exact.available() and n2q > exact_max_gates:
                    from .window import window_lns
                    window_lns(program, hardware_graph, best.placement, _to_ir(best.routed),
                               deadline=t0 + budget, rng=rng, window=window_size,
                               on_improve=lambda pl, ops: offer(pl, ops, "window"),
                               verbose=verbose)
            except _Optimal:
                raise
            except Exception as exc:                   # never let the window stage break a run
                if verbose:
                    print(f"    [window] skipped: {exc}")

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

    except _Optimal:
        pass

    assert best is not None, "portfolio produced no valid candidate"
    return best.placement, best.routed
