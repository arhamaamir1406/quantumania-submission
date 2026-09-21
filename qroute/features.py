"""State -> tensor encoding for the routing value network.

Three streams, all derived from the same MDP state:

* **physical nodes** (one token per hardware qubit) -- occupancy, the
  scheduler's clock `tau` and its slack, active-gate geometry, and what the
  resident logical qubit wants next.
* **hardware edges** (one token per hardware edge) -- what applying *this*
  SWAP would do to the active gate and to the near-term gate window. This is
  the stream that carries "is this SWAP useful", which a node-only encoder can
  only express indirectly.
* **gate-window tokens** (one token per upcoming program op) -- the next `W`
  ops with their current physical separation, contention against each other,
  and depth slack. This is the program's future, which the previous encoder
  compressed into a single scalar.

Everything is vectorised across a batch of states: a beam of 1500 states is
one pass of numpy fancy indexing, not 1500 Python loops. Static per-problem
data (distances, distance buckets, Laplacian positional encodings, degree,
centrality) is computed once in `GraphContext` and shared by every state.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch

from .mdp import Problem, State

# ---------------------------------------------------------------------------
# feature widths -- imported by the model to size its input projections
# ---------------------------------------------------------------------------
N_NODE_DYN = 22        # per physical qubit, per state
N_NODE_STATIC = 6      # per physical qubit, per problem (+ Laplacian PE)
N_EDGE_DYN = 12        # per hardware edge, per state
N_EDGE_STATIC = 4      # per hardware edge, per problem
N_GATE = 18            # per gate-window token
N_GLOBAL = 22          # per state

WINDOW = 16            # gate-window length
EDGE_WINDOW = 8        # window depth used for edge lookahead (cost control)
PE_DIM = 8             # Laplacian eigenvector positional encoding dims
DIST_BUCKETS = 12      # attention-bias buckets over hardware distance
GAMMA = 0.85           # geometric decay over the gate window


class GraphContext:
    """Static, per-(hardware, program) tensors shared by every state."""

    def __init__(self, problem: Problem, window: int = WINDOW,
                 edge_window: int = EDGE_WINDOW, pe_dim: int = PE_DIM):
        self.p = problem
        self.window = window
        self.edge_window = min(edge_window, window)
        hw = problem.hw
        nodes = problem.nodes
        n = self.n = problem.n_phys
        index = {u: i for i, u in enumerate(nodes)}
        self.index = index

        # ---- distances ----------------------------------------------------
        dist = np.zeros((n, n), dtype=np.float32)
        for u, row in problem.dist.items():
            iu = index[u]
            for v, d in row.items():
                dist[iu, index[v]] = d
        self.dist = dist
        self.diam = max(1.0, float(dist.max()))
        self.dist_bucket = torch.from_numpy(
            np.minimum(dist, DIST_BUCKETS - 1).astype(np.int64))

        # ---- adjacency / edges --------------------------------------------
        a = np.zeros((n, n), dtype=np.float32)
        edges = []
        for u, v in hw.edges:
            iu, iv = index[u], index[v]
            a[iu, iv] = a[iv, iu] = 1.0
            edges.append((iu, iv) if iu < iv else (iv, iu))
        edges.sort()
        self.eu = np.array([e[0] for e in edges], dtype=np.int64)
        self.ev = np.array([e[1] for e in edges], dtype=np.int64)
        self.n_edges = len(edges)
        self.adj = torch.from_numpy(a)
        self.a_norm = torch.from_numpy(a / np.maximum(a.sum(1, keepdims=True), 1.0))
        self.edge_u = torch.from_numpy(self.eu)
        self.edge_v = torch.from_numpy(self.ev)

        deg = a.sum(1)
        self.deg = deg
        maxdeg = max(1.0, float(deg.max()))

        # ---- Laplacian positional encoding --------------------------------
        # Eigenvectors of the symmetric normalised Laplacian. Sign is arbitrary,
        # so training flips signs at random; that keeps the net from reading
        # them as node identities.
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
        lap = np.eye(n, dtype=np.float32) - (dinv[:, None] * a * dinv[None, :])
        w, vecs = np.linalg.eigh(lap)
        order = np.argsort(w)[1:pe_dim + 1]          # drop the trivial one
        pe = np.zeros((n, pe_dim), dtype=np.float32)
        take = vecs[:, order]
        pe[:, :take.shape[1]] = take
        self.lap_pe = torch.from_numpy(pe)

        # ---- static node features -----------------------------------------
        closeness = dist.sum(1)
        ecc = dist.max(1)
        try:
            btw = nx.betweenness_centrality(hw)
            btw_arr = np.array([btw[u] for u in nodes], dtype=np.float32)
        except Exception:                                   # pragma: no cover
            btw_arr = np.zeros(n, dtype=np.float32)
        ns = np.zeros((n, N_NODE_STATIC), dtype=np.float32)
        ns[:, 0] = deg / maxdeg
        ns[:, 1] = closeness / max(1.0, float(closeness.max()))
        ns[:, 2] = ecc / self.diam
        ns[:, 3] = btw_arr
        ns[:, 4] = (a @ deg) / max(1.0, float((a @ deg).max()))   # neighbour degree
        ns[:, 5] = float(problem.n_log) / max(1, n)
        self.node_static = torch.from_numpy(ns)

        # ---- static edge features -----------------------------------------
        es = np.zeros((self.n_edges, N_EDGE_STATIC), dtype=np.float32)
        es[:, 0] = deg[self.eu] / maxdeg
        es[:, 1] = deg[self.ev] / maxdeg
        # edge betweenness: fraction of node pairs whose shortest path could use it
        onpath = np.zeros(self.n_edges, dtype=np.float32)
        for e, (iu, iv) in enumerate(zip(self.eu, self.ev)):
            hit = (dist[:, iu][:, None] + 1 + dist[iv, :][None, :] == dist)
            hit |= (dist[:, iv][:, None] + 1 + dist[iu, :][None, :] == dist)
            onpath[e] = hit.mean()
        es[:, 2] = onpath
        es[:, 3] = np.minimum(dist[self.eu].min(0), dist[self.ev].min(0)).mean()
        self.edge_static = torch.from_numpy(es)

        # ---- program lookup tables ----------------------------------------
        m, nl = problem.n_ops, max(1, problem.n_log)
        self.m = m
        ops = problem.ops
        op_a = np.zeros(max(1, m), dtype=np.int64)
        op_b = np.zeros(max(1, m), dtype=np.int64)
        op_is2q = np.zeros(max(1, m), dtype=np.float32)
        for k, op in enumerate(ops):
            if op[0] == "2Q":
                op_a[k], op_b[k], op_is2q[k] = op[1], op[2], 1.0
            else:
                op_a[k] = op_b[k] = op[1]
        self.op_a, self.op_b, self.op_is2q = op_a, op_b, op_is2q

        # next 2Q op at or after k touching logical l, its partner, and how many
        # such ops remain
        nxt = np.full((m + 1, nl), -1, dtype=np.int64)
        partner = np.full((m + 1, nl), 0, dtype=np.int64)
        rem = np.zeros((m + 1, nl), dtype=np.float32)
        for k in range(m - 1, -1, -1):
            nxt[k] = nxt[k + 1]
            partner[k] = partner[k + 1]
            rem[k] = rem[k + 1]
            op = ops[k]
            if op[0] == "2Q":
                x, y = op[1], op[2]
                nxt[k, x], partner[k, x] = k, y
                nxt[k, y], partner[k, y] = k, x
                rem[k, x] += 1.0
                rem[k, y] += 1.0
        self.nxt, self.partner, self.rem = nxt, partner, rem

        self.offsets = np.arange(window, dtype=np.int64)
        self.decay = (GAMMA ** self.offsets).astype(np.float32)


class Encoder:
    """Vectorised `list[State]` -> feature tensors. One pass per beam."""

    def __init__(self, problem: Problem, ctx: GraphContext | None = None,
                 window: int = WINDOW):
        self.p = problem
        self.ctx = ctx if ctx is not None else GraphContext(problem, window=window)
        self.window = self.ctx.window

    # -- numpy encoding ----------------------------------------------------
    def encode_numpy(self, states: list[State]):
        c = self.ctx
        p = self.p
        b, n, W = len(states), c.n, self.window
        m = max(1, p.n_ops)
        nE = c.n_edges
        br = np.arange(b)[:, None]

        POS = np.array([s.pos for s in states], dtype=np.int64) if p.n_log else \
            np.zeros((b, 1), dtype=np.int64)
        OCC = np.array([s.occ for s in states], dtype=np.int64)
        TAU = np.array([s.tau for s in states], dtype=np.float32)
        K = np.array([s.k for s in states], dtype=np.int64)
        DEPTH = np.array([s.depth for s in states], dtype=np.float32)
        SWAPS = np.array([s.swaps for s in states], dtype=np.float32)

        live = OCC >= 0
        tmax = TAU.max(1, keepdims=True)
        slack = (tmax - TAU) / (tmax + 1.0)

        # ---- gate window ---------------------------------------------------
        raw = K[:, None] + c.offsets[None, :]              # (b, W)
        valid = (raw < p.n_ops).astype(np.float32)
        idx = np.minimum(raw, max(0, p.n_ops - 1))
        la, lb = c.op_a[idx], c.op_b[idx]                  # (b, W) logicals
        is2q = c.op_is2q[idx] * valid
        pa_w = POS[br, la]                                 # (b, W) physicals
        pb_w = POS[br, lb]
        dw = c.dist[pa_w, pb_w] * is2q                     # separation now

        pa0, pb0 = pa_w[:, 0], pb_w[:, 0]                  # active gate
        active = (K < p.n_ops).astype(np.float32)
        d_ab = c.dist[pa0, pb0] * active

        # ---- node features -------------------------------------------------
        xn = np.zeros((b, n, N_NODE_DYN), dtype=np.float32)
        d_to_a = c.dist[pa0]                               # (b, n)
        d_to_b = c.dist[pb0]
        xn[:, :, 0] = live
        xn[:, :, 1] = slack
        xn[:, :, 2] = TAU / (DEPTH[:, None] + 1.0)
        xn[:, :, 3] = (TAU >= tmax)
        xn[np.arange(b), pa0, 4] = active
        xn[np.arange(b), pb0, 5] = active
        xn[:, :, 6] = d_to_a / c.diam
        xn[:, :, 7] = d_to_b / c.diam
        xn[:, :, 8] = (d_to_a + d_to_b <= d_ab[:, None])   # on a shortest path
        xn[:, :, 9] = np.minimum(d_to_a, d_to_b) / c.diam

        lsafe = np.where(live, OCC, 0)
        j_next = c.nxt[K[:, None], lsafe]                  # (b, n)
        has_next = live & (j_next >= 0)
        partner = np.where(has_next, c.partner[K[:, None], lsafe], 0)
        p_partner = POS[br, partner]
        gap = np.where(has_next, j_next - K[:, None], m).astype(np.float32)
        xn[:, :, 10] = c.dist[np.arange(n)[None, :], p_partner] / c.diam * has_next
        xn[:, :, 11] = has_next / (1.0 + gap)
        xn[:, :, 12] = np.minimum(gap, m) / m
        xn[:, :, 13] = c.rem[K[:, None], lsafe] / m * has_next
        xn[:, :, 14] = has_next

        # how often this physical qubit appears in the gate window, and how far
        # its partner is on average there
        hit_a = (pa_w[:, None, :] == np.arange(n)[None, :, None]) & (is2q[:, None, :] > 0)
        hit_b = (pb_w[:, None, :] == np.arange(n)[None, :, None]) & (is2q[:, None, :] > 0)
        hit = hit_a | hit_b                                 # (b, n, W)
        wdec = c.decay[None, None, :]
        xn[:, :, 15] = (hit * wdec).sum(2) / max(1e-6, c.decay.sum())
        cnt = hit.sum(2)
        xn[:, :, 16] = cnt / W
        xn[:, :, 17] = (hit * dw[:, None, :]).sum(2) / np.maximum(cnt, 1) / c.diam
        first = np.where(hit.any(2), hit.argmax(2), W)
        xn[:, :, 18] = first / W
        xn[:, :, 19] = 1.0 / (1.0 + first)
        # distance to the nearest unoccupied qubit -- room to move into.
        # A large *finite* sentinel, not inf: when every physical qubit is
        # occupied there is no free target at all, and inf would poison the
        # whole feature tensor (and every loss downstream of it).
        far = 2.0 * c.diam
        freed = np.where(live, far, 0.0)
        xn[:, :, 20] = np.minimum((c.dist[None, :, :] + freed[:, None, :]).min(2),
                                  far) / c.diam
        xn[:, :, 21] = (tmax - TAU[np.arange(b), pa0][:, None]) / (tmax + 1.0)

        # ---- edge features -------------------------------------------------
        eu, ev = c.eu, c.ev
        xe = np.zeros((b, nE, N_EDGE_DYN), dtype=np.float32)
        occ_u, occ_v = live[:, eu], live[:, ev]

        def swapped(pos_arr):
            """Where each physical index lands after swapping every edge."""
            at_u = pos_arr[:, :, None] == eu[None, None, :]
            at_v = pos_arr[:, :, None] == ev[None, None, :]
            out = np.broadcast_to(pos_arr[:, :, None], pos_arr.shape + (nE,)).copy()
            out = np.where(at_u, ev[None, None, :], out)
            out = np.where(at_v, eu[None, None, :], out)
            return out                                      # (b, W, E)

        ew = c.edge_window
        npa = swapped(pa_w[:, :ew])
        npb = swapped(pb_w[:, :ew])
        d_after = c.dist[npa, npb]                          # (b, ew, E)
        d_before = dw[:, :ew, None]
        gain = (d_before - d_after) * is2q[:, :ew, None]
        dec = c.decay[None, :ew, None]

        xe[:, :, 0] = occ_u & occ_v
        xe[:, :, 1] = (occ_u | occ_v)
        xe[:, :, 2] = gain[:, 0, :]                          # active-gate gain
        xe[:, :, 3] = (gain * dec).sum(1) / max(1e-6, c.decay[:ew].sum())
        xe[:, :, 4] = gain.max(1)
        xe[:, :, 5] = (gain > 0).mean(1)
        tau_e = np.maximum(TAU[:, eu], TAU[:, ev])
        xe[:, :, 6] = (tmax - tau_e) / (tmax + 1.0)
        xe[:, :, 7] = tau_e / (DEPTH[:, None] + 1.0)
        on_path = (d_to_a[:, eu] + 1 + d_to_b[:, ev]) <= d_ab[:, None]
        on_path |= (d_to_a[:, ev] + 1 + d_to_b[:, eu]) <= d_ab[:, None]
        xe[:, :, 8] = on_path * active[:, None]
        # does this edge already hold a pair that wants to interact soon?
        pair_hit = (((pa_w[:, :ew, None] == eu[None, None, :]) &
                     (pb_w[:, :ew, None] == ev[None, None, :])) |
                    ((pa_w[:, :ew, None] == ev[None, None, :]) &
                     (pb_w[:, :ew, None] == eu[None, None, :])))
        pair_hit = pair_hit & (is2q[:, :ew, None] > 0)
        xe[:, :, 9] = (pair_hit * dec).sum(1)
        xe[:, :, 10] = pair_hit[:, 0, :]
        xe[:, :, 11] = (d_after[:, 0, :] <= 1) * active[:, None]

        # ---- gate-window tokens --------------------------------------------
        xgt = np.zeros((b, W, N_GATE), dtype=np.float32)
        tau_a = TAU[br, pa_w]
        tau_b = TAU[br, pb_w]
        share_active = (((la == la[:, :1]) | (la == lb[:, :1]) |
                         (lb == la[:, :1]) | (lb == lb[:, :1])) * is2q)
        prev_la = np.concatenate([la[:, :1], la[:, :-1]], 1)
        prev_lb = np.concatenate([lb[:, :1], lb[:, :-1]], 1)
        share_prev = ((la == prev_la) | (la == prev_lb) |
                      (lb == prev_la) | (lb == prev_lb)) * is2q
        overlap = (((la[:, :, None] == la[:, None, :]) | (la[:, :, None] == lb[:, None, :]) |
                    (lb[:, :, None] == la[:, None, :]) | (lb[:, :, None] == lb[:, None, :]))
                   * is2q[:, None, :]).sum(2)

        xgt[:, :, 0] = valid
        xgt[:, :, 1] = is2q
        xgt[:, :, 2] = c.offsets[None, :] / W
        xgt[:, :, 3] = c.decay[None, :]
        xgt[:, :, 4] = dw / c.diam
        xgt[:, :, 5] = np.maximum(dw - 1, 0) / c.diam
        xgt[:, :, 6] = is2q / (1.0 + dw)
        xgt[:, :, 7] = (dw <= 1) * is2q
        xgt[:, :, 8] = (tmax - np.maximum(tau_a, tau_b)) / (tmax + 1.0)
        xgt[:, :, 9] = np.abs(tau_a - tau_b) / (tmax + 1.0)
        xgt[:, :, 10] = share_active
        xgt[:, :, 11] = share_prev
        xgt[:, :, 12] = overlap / W
        xgt[:, :, 13] = c.rem[K[:, None], la] / m * is2q
        xgt[:, :, 14] = c.rem[K[:, None], lb] / m * is2q
        xgt[:, :, 15] = (np.maximum(dw - 1, 0) + 0.5 * ((np.maximum(dw - 1, 0) + 1) // 2)) / c.diam
        xgt[:, :, 16] = (dw > 1) * is2q
        xgt[:, :, 17] = c.decay[None, :] * np.maximum(dw - 1, 0) / c.diam

        # ---- globals --------------------------------------------------------
        xg = np.zeros((b, N_GLOBAL), dtype=np.float32)
        rem_ops = (p.n_ops - K).astype(np.float32)
        lb_swaps = (np.maximum(dw - 1, 0) * is2q).sum(1)
        xg[:, 0] = rem_ops / m
        xg[:, 1] = K / m
        xg[:, 2] = DEPTH / m
        xg[:, 3] = SWAPS / m
        xg[:, 4] = (SWAPS + 0.5 * DEPTH) / m
        xg[:, 5] = d_ab / c.diam
        xg[:, 6] = (dw * is2q).sum(1) / np.maximum(is2q.sum(1), 1) / c.diam
        xg[:, 7] = lb_swaps / m
        xg[:, 8] = c.rem[K].sum(1) / (2.0 * m)                  # remaining 2Q ops
        xg[:, 9] = tmax[:, 0] / m
        xg[:, 10] = TAU.mean(1) / (tmax[:, 0] + 1.0)
        xg[:, 11] = TAU.std(1) / (tmax[:, 0] + 1.0)
        xg[:, 12] = live.mean(1)
        xg[:, 13] = p.n_log / max(1, c.n)
        xg[:, 14] = c.diam / 10.0
        xg[:, 15] = c.n_edges / max(1, c.n)
        xg[:, 16] = np.log1p(m) / 10.0
        xg[:, 17] = np.log1p(c.n) / 5.0
        xg[:, 18] = active
        xg[:, 19] = (dw <= 1).mean(1)
        xg[:, 20] = (DEPTH + 1.0) / (K + 1.0)
        xg[:, 21] = SWAPS / np.maximum(K, 1)

        # Nothing below should be able to produce a non-finite value, but a
        # single NaN here silently destroys a training run, so it is cheap
        # insurance to clamp rather than to trust.
        for arr in (xn, xe, xgt, xg):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e4, neginf=-1e4)

        return {
            "node": xn, "edge": xe, "gate": xgt, "glob": xg,
            "gate_pa": pa_w.astype(np.int64), "gate_pb": pb_w.astype(np.int64),
            "cost": (SWAPS + 0.5 * DEPTH).astype(np.float32),
        }

    def encode(self, states: list[State]) -> dict:
        out = self.encode_numpy(states)
        return {k: torch.from_numpy(v) for k, v in out.items()}


def context_tensors(ctx: GraphContext, device="cpu", dtype=torch.float32) -> dict:
    """Static tensors the model needs, moved to `device` once."""
    return {
        "node_static": ctx.node_static.to(device=device, dtype=dtype),
        "edge_static": ctx.edge_static.to(device=device, dtype=dtype),
        "lap_pe": ctx.lap_pe.to(device=device, dtype=dtype),
        "a_norm": ctx.a_norm.to(device=device, dtype=dtype),
        "adj": ctx.adj.to(device=device, dtype=dtype),
        "dist_bucket": ctx.dist_bucket.to(device=device),
        "edge_u": ctx.edge_u.to(device=device),
        "edge_v": ctx.edge_v.to(device=device),
    }
