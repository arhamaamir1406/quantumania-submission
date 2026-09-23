"""Exact placement + routing + scheduling as a CP-SAT model.

This is the member of the portfolio that can *prove* things. It optimises the
competition objective directly -- minimise `2 * swaps + depth`, i.e. twice the
score, so everything stays integral -- over a time-layered model of the whole
routed circuit:

    x[l][p][t]   logical l sits on physical p at the start of layer t
    s[e][t]      a SWAP on hardware edge e occupies layer t
    y[k][e][t]   2Q gate k executes on hardware edge e in layer t
    act[t]       layer t is used (depth = sum of act)

with the obvious constraints: the mapping is a partial permutation, every
physical qubit does at most one thing per layer, the mapping only changes
through SWAPs, and a gate runs on an edge holding exactly its two logicals.

The scorer's order rule
-----------------------
The scorer does not accept a layered circuit, it accepts a *list*, and it
requires the gates in that list to appear in exact program order; depth is
then the ASAP layering of the list. So a layered solution is only realisable
if it can be linearised with gates in program order. That fails exactly when a
chain of ops sharing physical qubits runs from a gate to an *earlier* gate --
e.g. gate 7 at layer 1 feeds a SWAP at layer 2 which feeds gate 3 at layer 3.

Forbidding "gate layers must be non-decreasing in program index" would rule
that out, but it is far too strong: it forbids all the legitimate parallelism
the ASAP scheduler finds between later and earlier independent gates, and the
model would then not be exact. Instead each physical qubit carries a watermark

    H[p][t] >= the largest gate index with a dependency chain into (p, t)

which only grows, flows across SWAPs, and is set to k by gate k; gate k may run
on (p, q) only if both watermarks are still below k. That is precisely the
"no backward chain" condition, so the model is exact:

  * every model solution linearises (topological sort of the physical
    dependency DAG plus program-order edges, which is acyclic by the
    watermark), and the ASAP depth of that list is <= the model's depth;
  * every valid routed list, ASAP-layered, is a model solution.

Hence the solver's objective bound is a genuine lower bound on the score of
*any* routing whose depth fits in the horizon `T`, and choosing
`T >= 2 * (incumbent - swap_floor)` removes even that caveat (a deeper routing
cannot beat the incumbent). When CP-SAT reports OPTIMAL under such a horizon,
the result is certified optimal.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import networkx as nx

from starter_kit.scorer import schedule_layers_ordered

from .ir import evaluate

try:
    from ortools.sat.python import cp_model
    HAVE_ORTOOLS = True
except Exception:                                    # pragma: no cover
    HAVE_ORTOOLS = False


@dataclass
class ExactResult:
    placement: dict | None      # logical -> physical, or None if nothing found
    ops: list | None            # IR ops: ("SWAP", p, q) / ("GATE", k)
    score: float                # realised score (inf if none)
    bound: float                # proven lower bound on the score within horizon T
    optimal: bool               # CP-SAT proved optimality within horizon T
    certified: bool             # horizon large enough that `bound` holds for every routing
    horizon: int
    status: str
    seconds: float


def _gate_windows(two_q: list[tuple[int, int, int]], T: int) -> list[tuple[int, int]]:
    """Earliest/latest layer (0-indexed) for each 2Q gate, from logical chains.

    Two gates sharing a logical qubit can never share a layer, and program
    order fixes which comes first, so the logical ASAP/ALAP layers bound every
    routing's layer for that gate.
    """
    last: dict[int, int] = {}
    lo = []
    for _, a, b in two_q:
        t = max(last.get(a, -1), last.get(b, -1)) + 1
        lo.append(t)
        last[a] = last[b] = t
    nxt: dict[int, int] = {}
    hi = [0] * len(two_q)
    for k in range(len(two_q) - 1, -1, -1):
        _, a, b = two_q[k]
        t = min(nxt.get(a, T), nxt.get(b, T)) - 1
        hi[k] = t
        nxt[a] = nxt[b] = t
    return list(zip(lo, hi))


def layered_from_routed(program, hw, placement, routed):
    """ASAP-layer a routed list: -> (layers of ('S',p,q) / ('G',k,p,q), mappings).

    `mappings[t]` is logical -> physical at the start of layer t.
    """
    layers = schedule_layers_ordered(routed)
    # attach program indices to gates, in order
    two_q_idx = [i for i, op in enumerate(program) if op[0] == "2Q"]
    gate_ids = iter(range(len(two_q_idx)))
    tag = {}
    for pos_in_list, op in enumerate(routed):
        if op[0] == "2Q":
            tag[pos_in_list] = next(gate_ids)
    # rebuild layer assignment per op, mirroring schedule_layers_ordered
    last: dict[int, int] = {}
    out: list[list[tuple]] = [[] for _ in layers]
    for i, op in enumerate(routed):
        if op[0] == "1Q":
            continue
        L = 1 + max(last.get(q, 0) for q in op[1:])
        for q in op[1:]:
            last[q] = L
        if op[0] == "SWAP":
            out[L - 1].append(("S", op[1], op[2]))
        else:
            out[L - 1].append(("G", tag[i], op[1], op[2]))
    maps = []
    cur = dict(placement)
    occ = {p: l for l, p in cur.items()}
    for layer in out:
        maps.append(dict(cur))
        for op in layer:
            if op[0] == "S":
                _, p, q = op
                a, b = occ.get(p), occ.get(q)
                if a is not None:
                    cur[a] = q
                if b is not None:
                    cur[b] = p
                occ[p], occ[q] = b, a
    maps.append(dict(cur))
    return out, maps


def linearize(program, layers) -> list[tuple]:
    """Layered solution -> IR op list with gates in exact program order.

    Topological sort of (per-physical-qubit order) + (program order between
    consecutive 2Q gates), breaking ties by layer. The watermark constraint
    guarantees this graph is acyclic. 1Q ops are emitted just before the next
    2Q gate (or at the end), which keeps them in program order and costs no
    depth.
    """
    import heapq

    two_q_idx = [i for i, op in enumerate(program) if op[0] == "2Q"]
    nodes = []                       # (layer, kind, payload)
    for t, layer in enumerate(layers):
        for op in layer:
            nodes.append((t, op))
    n = len(nodes)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    by_phys: dict[int, list[int]] = {}
    gate_node: dict[int, int] = {}
    for i, (t, op) in enumerate(nodes):
        qs = op[1:] if op[0] == "S" else op[2:]
        for q in qs:
            by_phys.setdefault(q, []).append(i)
        if op[0] == "G":
            gate_node[op[1]] = i
    for q, lst in by_phys.items():
        lst.sort(key=lambda i: nodes[i][0])
        for u, v in zip(lst, lst[1:]):
            succ[u].append(v)
            indeg[v] += 1
    for k in range(len(two_q_idx) - 1):
        u, v = gate_node[k], gate_node[k + 1]
        succ[u].append(v)
        indeg[v] += 1
    heap = [(nodes[i][0], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (nodes[v][0], v))
    if len(order) != n:
        raise ValueError("layered solution is not linearisable in program order")

    ops: list[tuple] = []
    next_prog = 0
    for u in order:
        _, op = nodes[u]
        if op[0] == "S":
            ops.append(("SWAP", op[1], op[2]))
        else:
            pi = two_q_idx[op[1]]
            while next_prog < pi:          # pending 1Q ops
                ops.append(("GATE", next_prog))
                next_prog += 1
            ops.append(("GATE", pi))
            next_prog = pi + 1
    while next_prog < len(program):
        ops.append(("GATE", next_prog))
        next_prog += 1
    return ops


def solve_exact(program: list[tuple], hw: nx.Graph, *, time_limit: float = 30.0,
                horizon: int | None = None, hint=None, upper: float | None = None,
                swap_floor: int = 0, workers: int = 8, verbose: bool = False,
                seed: int = 0, fix_depth: int | None = None,
                depth_weight: int = 1, order: bool = True,
                tube: int | None = None) -> ExactResult:
    """Solve (or improve) one instance with CP-SAT.

    hint:        (placement, routed) to warm-start from -- usually the
                 portfolio's incumbent.
    upper:       incumbent score; solutions scoring more are excluded (ties
                 are kept, so a hint scoring exactly `upper` stays feasible).
    horizon:     number of layers T; defaults to one that makes the bound
                 certified when `upper` is known, capped for tractability.
    tube:        with a hint, restrict every gate to within `tube` layers of
                 its layer in the hint -- a neighbourhood that spans the whole
                 horizon (so depth can be compressed globally) but is small
                 enough for CP-SAT on instances too big to solve whole.
                 The bound then only holds inside the tube, so it is never
                 reported as certified.
    """
    t_start = time.monotonic()
    if not HAVE_ORTOOLS:
        return ExactResult(None, None, math.inf, 0.0, False, False, 0, "no-ortools", 0.0)

    logicals = sorted({q for op in program for q in op[1:]})
    L = len(logicals)
    lid = {l: i for i, l in enumerate(logicals)}
    P = sorted(hw.nodes)
    edges = [tuple(sorted(e)) for e in hw.edges]
    E = len(edges)
    inc = {p: [] for p in P}
    for ei, (p, q) in enumerate(edges):
        inc[p].append(ei)
        inc[q].append(ei)
    two_q = [("2Q", lid[op[1]], lid[op[2]]) for op in program if op[0] == "2Q"]
    m = len(two_q)

    hint_layers = hint_maps = None
    if hint is not None:
        hint_layers, hint_maps = layered_from_routed(program, hw, hint[0], hint[1])

    if horizon is None:
        logical_depth = max((w[0] for w in _gate_windows(two_q, 10 ** 6)), default=-1) + 1
        base = len(hint_layers) if hint_layers is not None else logical_depth + 6
        if upper is not None and math.isfinite(upper):
            cert = int(math.floor(2 * (upper - swap_floor)))
            horizon = max(base, min(cert, base + 12))
        else:
            horizon = base + 4
    T = max(horizon, 1)
    certified_h = (tube is None and order and upper is not None and math.isfinite(upper)
                   and T >= 2 * (upper - swap_floor) - 1)

    windows = _gate_windows(two_q, T)
    if tube is not None and hint_layers is not None:
        at = {}
        for t, layer in enumerate(hint_layers):
            for op in layer:
                if op[0] == "G":
                    at[op[1]] = t
        windows = [(max(lo, at[k] - tube), min(hi, at[k] + tube)) if k in at else (lo, hi)
                   for k, (lo, hi) in enumerate(windows)]
    if any(lo > hi for lo, hi in windows):
        return ExactResult(None, None, math.inf, 0.0, False, False, T, "horizon-too-short",
                           time.monotonic() - t_start)

    M = cp_model.CpModel()
    x = [[[M.NewBoolVar("") for _ in range(T + 1)] for _ in P] for _ in range(L)]
    s = [[M.NewBoolVar("") for _ in range(T)] for _ in range(E)]
    y: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for k in range(m):
        lo, hi = windows[k]
        for ei in range(E):
            for t in range(lo, hi + 1):
                y[k, ei, t] = M.NewBoolVar("")
    act = [M.NewBoolVar("") for _ in range(T)]
    H = [[M.NewIntVar(-1, m, "") for _ in range(T + 1)] for _ in P]

    pidx = {p: i for i, p in enumerate(P)}

    # mapping is an injection at every layer
    for t in range(T + 1):
        for l in range(L):
            M.AddExactlyOne(x[l][pi][t] for pi in range(len(P)))
        for pi in range(len(P)):
            M.AddAtMostOne(x[l][pi][t] for l in range(L))

    # gates: exactly one (edge, layer); logicals present on that edge
    gates_at = {}          # (pi, t) -> list of y vars touching pi at t
    for (k, ei, t), v in y.items():
        p, q = edges[ei]
        for r in (p, q):
            gates_at.setdefault((pidx[r], t), []).append(v)
    for k in range(m):
        _, a, b = two_q[k]
        lo, hi = windows[k]
        M.AddExactlyOne(y[k, ei, t] for ei in range(E) for t in range(lo, hi + 1))
        for ei in range(E):
            p, q = edges[ei]
            ip, iq = pidx[p], pidx[q]
            for t in range(lo, hi + 1):
                v = y[k, ei, t]
                M.AddBoolOr([x[a][ip][t], x[a][iq][t]]).OnlyEnforceIf(v)
                M.AddBoolOr([x[b][ip][t], x[b][iq][t]]).OnlyEnforceIf(v)
                M.AddImplication(v, act[t])

    # one op per physical qubit per layer
    for pi, p in enumerate(P):
        for t in range(T):
            M.AddAtMostOne([s[ei][t] for ei in inc[p]] + gates_at.get((pi, t), []))

    # mapping dynamics
    for pi, p in enumerate(P):
        for t in range(T):
            sw = [s[ei][t] for ei in inc[p]]
            for l in range(L):
                # no swap on p  =>  unchanged
                M.Add(x[l][pi][t + 1] - x[l][pi][t] <= sum(sw))
                M.Add(x[l][pi][t] - x[l][pi][t + 1] <= sum(sw))
    for ei, (p, q) in enumerate(edges):
        ip, iq = pidx[p], pidx[q]
        for t in range(T):
            v = s[ei][t]
            M.AddImplication(v, act[t])
            for l in range(L):
                M.Add(x[l][ip][t + 1] == x[l][iq][t]).OnlyEnforceIf(v)
                M.Add(x[l][iq][t + 1] == x[l][ip][t]).OnlyEnforceIf(v)

    # program-order watermark (order=False drops it: a relaxation, for bounds)
    for pi in range(len(P) if order else 0):
        M.Add(H[pi][0] == -1)
        for t in range(T):
            M.Add(H[pi][t + 1] >= H[pi][t])
    for ei, (p, q) in enumerate(edges if order else []):
        ip, iq = pidx[p], pidx[q]
        for t in range(T):
            v = s[ei][t]
            M.Add(H[ip][t + 1] >= H[iq][t]).OnlyEnforceIf(v)
            M.Add(H[iq][t + 1] >= H[ip][t]).OnlyEnforceIf(v)
    for (k, ei, t), v in (y.items() if order else ()):
        p, q = edges[ei]
        for r in (pidx[p], pidx[q]):
            M.Add(H[r][t] <= k - 1).OnlyEnforceIf(v)
            M.Add(H[r][t + 1] >= k).OnlyEnforceIf(v)

    # depth: used layers form a prefix
    for t in range(T - 1):
        M.AddImplication(act[t + 1], act[t])

    n_swaps = sum(s[ei][t] for ei in range(E) for t in range(T))
    obj2 = 2 * n_swaps + depth_weight * sum(act)
    M.Add(n_swaps >= swap_floor)
    # depth >= logical depth
    logical_depth = max((w[0] for w in windows), default=-1) + 1
    M.Add(sum(act) >= logical_depth)
    # Ties with `upper` are allowed so that a hint scoring exactly `upper`
    # stays feasible. With depth_weight=0 the objective is SWAPs alone and
    # `upper` caps the SWAP count.
    if upper is not None and math.isfinite(upper):
        if depth_weight:
            M.Add(obj2 <= int(round(2 * upper)))
        else:
            M.Add(n_swaps <= int(upper))
    if fix_depth is not None:                          # depth-sliced proofs
        M.Add(sum(act) == fix_depth)
    M.Minimize(obj2)

    # Dominance cuts. Each removes only solutions that have a strictly-no-worse
    # twin with one SWAP deleted, so they are safe inside optimality proofs.
    #  * the same SWAP twice in a row is the identity;
    #  * a SWAP of two empty qubits is the identity;
    #  * a SWAP with no later op on either of its qubits changes no gate's
    #    mapping, and deleting it cannot raise any ASAP layer.
    busy_op = [[None] * T for _ in P]
    for pi, p in enumerate(P):
        for t in range(T):
            lits = [s[ei][t] for ei in inc[p]] + gates_at.get((pi, t), [])
            b = M.NewBoolVar("")
            M.Add(b == sum(lits))
            busy_op[pi][t] = b
    for ei, (p, q) in enumerate(edges):
        ip, iq = pidx[p], pidx[q]
        for t in range(T):
            v = s[ei][t]
            if t + 1 < T:
                M.AddBoolOr([v.Not(), s[ei][t + 1].Not()])
            M.AddBoolOr([v.Not()] + [x[l][ip][t] for l in range(L)] + [x[l][iq][t] for l in range(L)])
            M.AddBoolOr([v.Not()] + [busy_op[r][t2] for r in (ip, iq) for t2 in range(t + 1, T)])

    # ---- hints / LNS fixing ------------------------------------------------
    def set_hint(layers, maps):
        """Hint every variable -- a complete solution, not a partial one."""
        Th = len(layers)
        for t in range(T + 1):
            inv = {pp: lid[ll] for ll, pp in maps[min(t, Th)].items()}
            for l in range(L):
                for pi, p in enumerate(P):
                    M.AddHint(x[l][pi][t], 1 if inv.get(p) == l else 0)
        eidx = {e: i for i, e in enumerate(edges)}
        swset, gset = set(), set()
        wm = [-1] * len(P)                       # watermark, replayed
        busy_at = set()
        for t in range(T):
            if order:
                for pi in range(len(P)):
                    M.AddHint(H[pi][t], wm[pi])
            for op in (layers[t] if t < Th else ()):
                if op[0] == "S":
                    p, q = op[1], op[2]
                    swset.add((eidx[tuple(sorted((p, q)))], t))
                    ip, iq = pidx[p], pidx[q]
                    wm[ip] = wm[iq] = max(wm[ip], wm[iq])
                else:
                    k, p, q = op[1], op[2], op[3]
                    gset.add((k, eidx[tuple(sorted((p, q)))], t))
                    ip, iq = pidx[p], pidx[q]
                    wm[ip] = wm[iq] = k
                busy_at.update({(pidx[op[-2]], t), (pidx[op[-1]], t)})
        if order:
            for pi in range(len(P)):
                M.AddHint(H[pi][T], wm[pi])
        for ei in range(E):
            for t in range(T):
                M.AddHint(s[ei][t], 1 if (ei, t) in swset else 0)
        for (k, ei, t), v in y.items():
            M.AddHint(v, 1 if (k, ei, t) in gset else 0)
        for pi in range(len(P)):
            for t in range(T):
                M.AddHint(busy_op[pi][t], 1 if (pi, t) in busy_at else 0)
        for t in range(T):
            M.AddHint(act[t], 1 if t < Th else 0)

    if hint_layers is not None and len(hint_layers) <= T:
        set_hint(hint_layers, hint_maps)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.1, time_limit - (time.monotonic() - t_start))
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = verbose
    status = solver.Solve(M)
    sname = solver.StatusName(status)
    bound2 = solver.BestObjectiveBound() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) \
        else (math.inf if status == cp_model.INFEASIBLE else 0.0)
    elapsed = time.monotonic() - t_start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # INFEASIBLE with an upper bound means: nothing <= upper in this horizon
        bnd = (2 * upper + 1) / 2 if (status == cp_model.INFEASIBLE and upper is not None) else 0.0
        return ExactResult(None, None, math.inf, bnd, False, certified_h, T, sname, elapsed)

    if not order:
        # A relaxation: its solutions need not be emittable, only its values
        # (objective and bound, in units of the model objective / 2) matter.
        return ExactResult(None, None, solver.ObjectiveValue() / 2.0, bound2 / 2.0,
                           status == cp_model.OPTIMAL, certified_h, T, sname, elapsed)

    placement = {}
    for l in range(L):
        for pi, p in enumerate(P):
            if solver.Value(x[l][pi][0]):
                placement[logicals[l]] = p
    layers: list[list[tuple]] = [[] for _ in range(T)]
    for ei in range(E):
        for t in range(T):
            if solver.Value(s[ei][t]):
                layers[t].append(("S",) + edges[ei])
    for (k, ei, t), v in y.items():
        if solver.Value(v):
            layers[t].append(("G", k) + edges[ei])
    ops = linearize(program, layers)
    score, _, msg = evaluate(program, hw, placement, ops)
    return ExactResult(placement, ops, score, bound2 / 2.0, status == cp_model.OPTIMAL,
                       certified_h, T, sname, elapsed)
