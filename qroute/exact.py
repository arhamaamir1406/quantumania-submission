"""Exact layered model of placement + routing, solved with CP-SAT.

This is a different search space from the move-set MDP: any SWAP on any
hardware edge in any layer, not only along the active gate's shortest path.

Variables, for layers t = 0..T-1:

    x[q, p, t]   logical q sits on physical p at the start of layer t (t = 0..T)
    s[e, t]      a SWAP on hardware edge e runs in layer t
    y[i, e, t]   program gate i runs on hardware edge e in layer t
    u[t]         layer t is used (used layers form a prefix)

Constraints: each logical on exactly one physical and each physical holding
at most one logical; ops in a layer act on disjoint physical qubits; a gate
runs on an edge whose endpoints hold its two logicals; SWAPs permute the
mapping into the next layer and every untouched qubit keeps its occupant.

Objective: minimise 2 * swaps + depth (= 2 * score).

Two orderings:

* ``strict`` -- gate layers are non-decreasing in program order. Emitting ops
  layer by layer (program order within a layer) then yields a routed program
  that passes the scorer, whose ASAP depth is at most the model's depth. Used
  as a solver.
* ``relaxed`` -- only gates sharing a logical qubit are ordered. The ASAP
  layering of *any* valid routed program is feasible here (ops in an ASAP
  layer are disjoint, per-qubit order is layer order, and two gates sharing a
  logical are chained through the qubits it moved across), so the relaxed
  optimum is a **lower bound** on the true optimum -- provided T is at least
  the depth of any solution that could beat the incumbent. Used for proofs.
"""
from __future__ import annotations

import networkx as nx

from .ir import evaluate

try:
    from ortools.sat.python import cp_model
except ImportError:                                     # optional dependency
    cp_model = None


def available() -> bool:
    return cp_model is not None


def hint_layers(program, hw, placement, ops, monotone: bool = True):
    """Layer every op of a solution for use as a solver hint.

    ASAP per physical qubit -- exactly the scorer's layering. With `monotone`,
    gate layers are additionally non-decreasing in program order, which the
    strict model requires. Returns (layers: list of (kind, payload, layer), depth).
    """
    pos = dict(placement)
    occ = {p: l for l, p in pos.items()}
    last: dict[int, int] = {}
    gate_floor = 0
    out = []
    for op in ops:
        if op[0] == "SWAP":
            _, p, q = op
            t = max(last.get(p, -1), last.get(q, -1)) + 1
            last[p] = last[q] = t
            a, b = occ.get(p), occ.get(q)
            if a is not None:
                pos[a] = q
            if b is not None:
                pos[b] = p
            occ[p], occ[q] = b, a
            out.append(("SWAP", (p, q), t))
        else:
            k = op[1]
            if program[k][0] != "2Q":
                out.append(("GATE", k, None))
                continue
            _, a, b = program[k]
            pa, pb = pos[a], pos[b]
            t = max(last.get(pa, -1), last.get(pb, -1)) + 1
            if monotone:
                t = max(t, gate_floor)
                gate_floor = t
            last[pa] = last[pb] = t
            out.append(("GATE", k, t))
    depth = 1 + max((t for *_, t in out if t is not None), default=-1)
    return out, depth


def emit(program, layered):
    """Order a layered solution into a routed program, or None if impossible.

    Constraints: ops on the same physical qubit keep their layer order, and
    program ops keep program order. Any topological order of these satisfies
    the scorer, whose ASAP depth is then at most the model's (per-qubit order
    is unchanged). A cycle means the relaxed solution reorders gates in a way
    no routed program can express.
    """
    import heapq
    n = len(layered)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    by_q: dict[int, list[int]] = {}
    for idx, (t, op, e) in enumerate(layered):
        for p in e:
            by_q.setdefault(p, []).append(idx)
    for p, lst in by_q.items():
        lst.sort(key=lambda i: layered[i][0])
        for u_, v_ in zip(lst, lst[1:]):
            succ[u_].append(v_)
            indeg[v_] += 1
    gidx = sorted((layered[i][1][1], i) for i in range(n) if layered[i][1][0] == "GATE")
    for (_, u_), (_, v_) in zip(gidx, gidx[1:]):
        succ[u_].append(v_)
        indeg[v_] += 1
    heap = [(layered[i][0], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for v_ in succ[i]:
            indeg[v_] -= 1
            if indeg[v_] == 0:
                heapq.heappush(heap, (layered[v_][0], v_))
    if len(order) != n:
        return None
    ops = [layered[i][1] for i in order]
    # 1Q ops: slot each right after the preceding program op
    full = []
    k0 = 0
    while k0 < len(program) and program[k0][0] != "2Q":
        full.append(("GATE", k0))
        k0 += 1
    for op in ops:
        full.append(op)
        if op[0] == "GATE":
            k = op[1] + 1
            while k < len(program) and program[k][0] != "2Q":
                full.append(("GATE", k))
                k += 1
    return full


def solve_exact(program, hw: nx.Graph, T: int, strict: bool = True, time_limit: float = 60.0,
                hint=None, workers: int = 0, upper: float | None = None, verbose: bool = False,
                fix_depth: int | None = None, round_limit: float | None = None,
                lazy: bool = True, initial: dict | None = None, busy: dict | None = None,
                all_logicals=None):
    """Returns dict(status, objective (score), bound (score), placement, ops).

    Window mode (used by window.py): `initial` fixes the starting mapping,
    `busy[p]` forbids ops on physical p before that layer (the qubit is still
    occupied by earlier ops), and `all_logicals` tracks logicals the window's
    gates do not touch but which still occupy positions. The score returned
    is then only meaningful to the caller, who splices and re-scores.
    """
    if cp_model is None:
        raise RuntimeError("ortools is not installed")
    gates = [(k, op[1], op[2]) for k, op in enumerate(program) if op[0] == "2Q"]
    logicals = sorted(all_logicals) if all_logicals is not None else \
        sorted({q for op in program for q in op[1:]})
    phys = sorted(hw.nodes)
    edges = [tuple(sorted(e)) for e in hw.edges]
    inc = {p: [j for j, e in enumerate(edges) if p in e] for p in phys}

    m = cp_model.CpModel()
    x = {(q, p, t): m.NewBoolVar(f"x{q}_{p}_{t}") for q in logicals for p in phys for t in range(T + 1)}
    s = {(j, t): m.NewBoolVar(f"s{j}_{t}") for j in range(len(edges)) for t in range(T)}
    y = {(i, j, t): m.NewBoolVar(f"y{i}_{j}_{t}")
         for i in range(len(gates)) for j in range(len(edges)) for t in range(T)}
    u = [m.NewBoolVar(f"u{t}") for t in range(T)]

    for t in range(T + 1):
        for q in logicals:
            m.AddExactlyOne(x[q, p, t] for p in phys)
        for p in phys:
            m.AddAtMostOne(x[q, p, t] for q in logicals)

    if initial is not None:
        for q in logicals:
            m.Add(x[q, initial[q], 0] == 1)
    if busy:
        for p, b in busy.items():
            for t in range(min(b, T)):
                for j in inc[p]:
                    m.Add(s[j, t] == 0)
                    for i in range(len(gates)):
                        m.Add(y[i, j, t] == 0)

    layer_of = []
    for i, (k, a, b) in enumerate(gates):
        m.AddExactlyOne(y[i, j, t] for j in range(len(edges)) for t in range(T))
        for j, (p1, p2) in enumerate(edges):
            for t in range(T):
                v = y[i, j, t]
                m.AddBoolOr([x[a, p1, t], x[a, p2, t]]).OnlyEnforceIf(v)
                m.AddBoolOr([x[b, p1, t], x[b, p2, t]]).OnlyEnforceIf(v)
        L = m.NewIntVar(0, T - 1, f"L{i}")
        m.Add(L == sum(t * y[i, j, t] for j in range(len(edges)) for t in range(T)))
        layer_of.append(L)

    # disjointness: each physical qubit in at most one op per layer
    for t in range(T):
        for p in phys:
            m.AddAtMostOne([s[j, t] for j in inc[p]] + [y[i, j, t] for i in range(len(gates)) for j in inc[p]])

    # mapping transitions
    for t in range(T):
        for p in phys:
            moved = [s[j, t] for j in inc[p]]
            for q in logicals:
                # untouched: occupant persists
                m.AddImplication(x[q, p, t], x[q, p, t + 1]).OnlyEnforceIf([v.Not() for v in moved])
                m.AddImplication(x[q, p, t + 1], x[q, p, t]).OnlyEnforceIf([v.Not() for v in moved])
            for j in inc[p]:
                r = edges[j][0] if edges[j][1] == p else edges[j][1]
                for q in logicals:
                    m.AddImplication(x[q, r, t], x[q, p, t + 1]).OnlyEnforceIf(s[j, t])
                    m.AddImplication(x[q, p, t + 1], x[q, r, t]).OnlyEnforceIf(s[j, t])

    # ordering
    for i in range(len(gates)):
        for i2 in range(i + 1, len(gates)):
            share = {gates[i][1], gates[i][2]} & {gates[i2][1], gates[i2][2]}
            if share:
                m.Add(layer_of[i] < layer_of[i2])
            elif strict:
                m.Add(layer_of[i] <= layer_of[i2])

    # used layers form a prefix
    for t in range(T):
        for j in range(len(edges)):
            m.AddImplication(s[j, t], u[t])
        for i in range(len(gates)):
            for j in range(len(edges)):
                m.AddImplication(y[i, j, t], u[t])
        if t + 1 < T:
            m.AddImplication(u[t + 1], u[t])

    swaps = sum(s.values())
    obj = 2 * swaps + sum(u)
    if upper is not None:
        m.Add(obj <= int(round(2 * upper)))
    if fix_depth is not None:                  # used for depth-sliced proofs
        m.Add(sum(u) == fix_depth)
    m.Minimize(obj)

    eidx = {e: j for j, e in enumerate(edges)}
    gi = {k: i for i, (k, _, _) in enumerate(gates)}

    def set_hint(placement, ops):
        """Hint every variable from a solution; skipped if it needs > T layers."""
        m.ClearHints()
        layered, used_T = hint_layers(program, hw, placement, ops, monotone=strict)
        if used_T > T:
            return
        pos = dict(placement)
        occ = {p: l for l, p in pos.items()}
        sw_at, g_at = {}, {}
        for kind, payload, t in layered:
            if kind == "SWAP":
                sw_at.setdefault(t, []).append(payload)
            elif t is not None:
                g_at.setdefault(t, []).append(payload)
        for t in range(T + 1):
            for q in logicals:
                for p in phys:
                    m.AddHint(x[q, p, t], pos[q] == p)
            if t == T:
                break
            m.AddHint(u[t], t < used_T)
            ss = {eidx[tuple(sorted(e))] for e in sw_at.get(t, [])}
            gg = {}
            for k in g_at.get(t, []):
                _, a, b = program[k]
                gg[gi[k]] = eidx[tuple(sorted((pos[a], pos[b])))]
            for j in range(len(edges)):
                m.AddHint(s[j, t], j in ss)
                for i in range(len(gates)):
                    m.AddHint(y[i, j, t], gg.get(i) == j)
            for (p, q2) in sw_at.get(t, []):
                a, b = occ.get(p), occ.get(q2)
                if a is not None:
                    pos[a] = q2
                if b is not None:
                    pos[b] = p
                occ[p], occ[q2] = b, a

    def extract(value):
        placement = {q: next(p for p in phys if value(x[q, p, 0])) for q in logicals}
        layered = []
        for t in range(T):
            for j, e in enumerate(edges):
                if value(s[j, t]):
                    layered.append((t, ("SWAP", e[0], e[1]), e))
            for i, (k, _, _) in enumerate(gates):
                for j, e in enumerate(edges):
                    if value(y[i, j, t]):
                        layered.append((t, ("GATE", k), e))
        return placement, layered

    best = {"score": float("inf"), "placement": None, "ops": None}

    def consider(placement, layered):
        ops = emit(program, layered)
        if ops is None:
            return False
        if initial is None:
            sc = evaluate(program, hw, placement, ops)[0]
        else:                                  # window: model objective
            sc = sum(1 for _, op, _ in layered if op[0] == "SWAP") + \
                0.5 * (1 + max((t for t, _, _ in layered), default=-1))
        if sc < best["score"]:
            best.update(score=sc, placement=placement, ops=ops)
        return True

    class _Callback(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            consider(*extract(self.Value))

    if hint is not None and initial is None:
        best["score"] = evaluate(program, hw, *hint)[0]
        best["placement"], best["ops"] = hint
        set_hint(*hint)

    # Lazy ordering: solve the relaxed model; if its solution cannot be
    # emitted, every cycle contains a qubit chain from a later gate j back to
    # an earlier gate i, so add layer(i) <= layer(j) for exactly those pairs
    # and re-solve from the best emittable solution.
    import time as _time
    deadline = _time.monotonic() + time_limit
    first_bound, status, rounds, n_lazy = None, "UNKNOWN", 0, 0
    while True:
        left = deadline - _time.monotonic()
        if left <= 0.2:
            break
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = min(left, round_limit) if round_limit else left
        if workers:
            solver.parameters.num_workers = workers
        solver.parameters.log_search_progress = verbose
        st = solver.Solve(m, _Callback())
        status = solver.StatusName(st)
        if first_bound is None:
            first_bound = solver.BestObjectiveBound() / 2
        rounds += 1
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if st == cp_model.UNKNOWN and round_limit and n_lazy:
                round_limit *= 2                   # constrained model: search longer
                continue
            break
        placement, layered = extract(solver.Value)
        if consider(placement, layered) or not lazy:
            break
        pairs = _backward_pairs(layered, gi)
        if not pairs:
            break
        for i, i2 in pairs:
            m.Add(layer_of[i] <= layer_of[i2])
        n_lazy += len(pairs)
        if best["ops"] is not None:
            set_hint(best["placement"], best["ops"])

    # Only the first round is a pure relaxation, so only its bound is a valid
    # lower bound on the true optimum (given T covers every better solution).
    return {"status": status, "bound": first_bound,
            "objective": best["score"] if best["ops"] is not None else None,
            "score": best["score"], "placement": best["placement"], "ops": best["ops"],
            "rounds": rounds, "lazy_pairs": n_lazy}


def _backward_pairs(layered, gi):
    """Gate pairs (i, i2), i < i2 in program order, where a chain of ops on
    shared physical qubits at increasing layers runs from i2 back to i."""
    by_q: dict[int, list[int]] = {}
    for idx, (t, op, e) in enumerate(layered):
        for p in e:
            by_q.setdefault(p, []).append(idx)
    succ = [set() for _ in layered]
    for lst in by_q.values():
        lst.sort(key=lambda i: layered[i][0])
        for a, b in zip(lst, lst[1:]):
            succ[a].add(b)
    gate_nodes = {idx: gi[op[1]] for idx, (t, op, e) in enumerate(layered) if op[0] == "GATE"}
    pairs = set()
    for src, g_src in gate_nodes.items():
        stack, seen = [src], {src}
        while stack:
            v = stack.pop()
            for w in succ[v]:
                if w in seen:
                    continue
                seen.add(w)
                if w in gate_nodes and gate_nodes[w] < g_src:
                    pairs.add((gate_nodes[w], g_src))
                stack.append(w)
    return sorted(pairs)
