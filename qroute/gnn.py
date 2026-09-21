"""The learned value function: a multi-stream graph transformer over routing states.

What it predicts
----------------
Cost-to-go, decomposed the way the competition scores it::

    swaps_to_go  ~ head_s(z)          >= 0
    depth_to_go  ~ head_d(z)          >= 0
    V            = swaps_to_go + 0.5 * depth_to_go

The objective is *structural*, not a loss-weighting choice: the network has no
way to emit a value that is not `swaps + 0.5 * depth` of something. Search then
ranks successors by `f(t) = t.cost + V(t)`, where `t.cost` is the exact partial
score so far, so `f` is an estimate of the final score and nothing else.

Both heads emit `n_quantiles` monotone quantiles (cumulative softplus) rather
than a point estimate. Beam search can then be run optimistically -- ranking on
a low quantile keeps candidates whose *downside* is good, which matters when
the top of the beam is crowded with states of near-identical mean value.

Architecture
------------
Three token streams (see `features.py`) fused by repeated attention:

* **node stream** -- one token per physical qubit. Gated message passing along
  hardware edges (so locality is a hard structural prior), then full
  self-attention biased by shortest-path distance buckets, Graphormer-style, so
  a qubit can also attend directly to a qubit four hops away without four
  rounds of propagation.
* **edge stream** -- one token per hardware edge, updated from its endpoints
  each block. This is the stream that represents *candidate SWAPs*.
* **gate stream** -- one token per upcoming program op, self-attending over the
  window (so contention between future gates is visible) and cross-attending to
  the node stream, bound to geometry by gathering node embeddings at the two
  physical qubits the gate's logicals currently occupy.

Globals condition every block through FiLM, so "10 gates left, depth already
26" modulates the whole computation rather than being concatenated at the end.

Sizes: `tiny` ~0.4M, `small` ~2.7M, `base` ~17M, `large` ~57M parameters. The
default is `base`; it trains comfortably on a single 5090 and is the size the
shipped checkpoint uses.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (DIST_BUCKETS, N_EDGE_DYN, N_EDGE_STATIC, N_GATE,
                       N_GLOBAL, N_NODE_DYN, N_NODE_STATIC, PE_DIM, WINDOW,
                       Encoder, GraphContext, context_tensors)
from .mdp import Problem, State

PRESETS = {
    "tiny":  dict(d_model=64,  n_blocks=3,  n_heads=4, d_ff=128,  d_edge=32,  gate_every=2),
    "small": dict(d_model=128, n_blocks=4,  n_heads=4, d_ff=384,  d_edge=64,  gate_every=2),
    "base":  dict(d_model=256, n_blocks=8,  n_heads=8, d_ff=1024, d_edge=128, gate_every=2),
    "large": dict(d_model=384, n_blocks=12, n_heads=8, d_ff=1536, d_edge=192, gate_every=2),
}


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    """Per-block scale/shift conditioned on the global feature vector."""

    def __init__(self, d_glob: int, d: int):
        super().__init__()
        self.lin = nn.Linear(d_glob, 2 * d)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, g):
        scale, shift = self.lin(g).chunk(2, dim=-1)
        return h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BiasedAttention(nn.Module):
    """Multi-head attention with an additive per-head bias table."""

    def __init__(self, d: int, n_heads: int, d_kv: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        d_kv = d_kv or d
        self.h = n_heads
        self.dk = d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d_kv, d)
        self.v = nn.Linear(d_kv, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, kv=None, bias=None, key_mask=None):
        kv = x if kv is None else kv
        b, n, _ = x.shape
        m = kv.shape[1]
        q = self.q(x).view(b, n, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(b, m, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(b, m, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        if bias is not None:
            att = att + bias
        if key_mask is not None:                      # (b, m), 1 = keep
            att = att.masked_fill(~key_mask[:, None, None, :].bool(), float("-inf"))
            att = torch.nan_to_num(att, neginf=-1e4)
        att = self.drop(att.softmax(-1))
        out = (att @ v).transpose(1, 2).reshape(b, n, -1)
        return self.o(out)


def _ffn(d: int, d_ff: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_ff, d))


class GraphBlock(nn.Module):
    """Edge update -> gated message passing -> distance-biased attention ->
    cross-attention to the gate window -> FFN. Pre-LN residual throughout."""

    def __init__(self, d: int, d_edge: int, n_heads: int, d_ff: int,
                 d_glob: int, dropout: float = 0.0):
        super().__init__()
        self.ln_e = nn.LayerNorm(d_edge)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d + d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, d_edge))

        self.ln_m = nn.LayerNorm(d)
        self.msg = nn.Sequential(
            nn.Linear(2 * d + d_edge, d), nn.GELU(), nn.Linear(d, d))
        self.msg_gate = nn.Linear(d_edge, d)
        self.msg_out = nn.Linear(d, d)

        self.ln_a = nn.LayerNorm(d)
        self.attn = BiasedAttention(d, n_heads, dropout=dropout)

        self.ln_x = nn.LayerNorm(d)
        self.ln_xk = nn.LayerNorm(d)
        self.cross = BiasedAttention(d, n_heads, dropout=dropout)

        self.ln_f = nn.LayerNorm(d)
        self.ffn = _ffn(d, d_ff, dropout)
        self.film = FiLM(d_glob, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, gate, g, ctx, dist_bias, gate_mask):
        eu, ev = ctx["edge_u"], ctx["edge_v"]
        hu, hv = h[:, eu], h[:, ev]

        # --- edge stream -------------------------------------------------
        en = self.ln_e(e)
        # symmetric in (u, v): average the two orientations
        e = e + self.edge_mlp(torch.cat([hu, hv, en], -1)) \
              + self.edge_mlp(torch.cat([hv, hu, en], -1))

        # --- gated message passing along hardware edges --------------------
        hn = self.ln_m(h)
        hun, hvn = hn[:, eu], hn[:, ev]
        gate_w = torch.sigmoid(self.msg_gate(e))
        m_to_u = self.msg(torch.cat([hun, hvn, e], -1)) * gate_w
        m_to_v = self.msg(torch.cat([hvn, hun, e], -1)) * gate_w
        agg = torch.zeros_like(h)
        agg.index_add_(1, eu, m_to_u)
        agg.index_add_(1, ev, m_to_v)
        agg = agg / ctx["deg_clamped"]
        h = h + self.drop(self.msg_out(agg))

        # --- distance-biased global attention over physical qubits ---------
        h = h + self.drop(self.attn(self.ln_a(h), bias=dist_bias))

        # --- node <- gate-window cross-attention ---------------------------
        h = h + self.drop(self.cross(self.ln_x(h), kv=self.ln_xk(gate),
                                     key_mask=gate_mask))

        # --- FFN + global conditioning -------------------------------------
        h = h + self.drop(self.ffn(self.ln_f(h)))
        return self.film(h, g), e


class GateBlock(nn.Module):
    """Self-attention over the gate window, then cross-attention to the nodes."""

    def __init__(self, d: int, n_heads: int, d_ff: int, d_glob: int,
                 dropout: float = 0.0):
        super().__init__()
        self.ln_s = nn.LayerNorm(d)
        self.self_attn = BiasedAttention(d, n_heads, dropout=dropout)
        self.ln_c = nn.LayerNorm(d)
        self.ln_ck = nn.LayerNorm(d)
        self.cross = BiasedAttention(d, n_heads, dropout=dropout)
        self.ln_f = nn.LayerNorm(d)
        self.ffn = _ffn(d, d_ff, dropout)
        self.film = FiLM(d_glob, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, gate, h, g, rel_bias, gate_mask):
        gate = gate + self.drop(self.self_attn(self.ln_s(gate), bias=rel_bias,
                                               key_mask=gate_mask))
        gate = gate + self.drop(self.cross(self.ln_c(gate), kv=self.ln_ck(h)))
        gate = gate + self.drop(self.ffn(self.ln_f(gate)))
        return self.film(gate, g)


class AttentionPool(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = BiasedAttention(d, n_heads)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, key_mask=None):
        q = self.query.expand(h.shape[0], -1, -1)
        return self.attn(q, kv=self.ln(h), key_mask=key_mask).squeeze(1)


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------
class RoutingNet(nn.Module):
    def __init__(self, d_model: int = 256, n_blocks: int = 8, n_heads: int = 8,
                 d_ff: int = 1024, d_edge: int = 128, gate_every: int = 2,
                 n_quantiles: int = 5, window: int = WINDOW,
                 dropout: float = 0.0, pe_dim: int = PE_DIM):
        super().__init__()
        self.cfg = dict(d_model=d_model, n_blocks=n_blocks, n_heads=n_heads,
                        d_ff=d_ff, d_edge=d_edge, gate_every=gate_every,
                        n_quantiles=n_quantiles, window=window, dropout=dropout,
                        pe_dim=pe_dim)
        self.n_quantiles = n_quantiles
        self.window = window
        d = d_model
        dg = d // 2

        self.glob_mlp = nn.Sequential(
            nn.Linear(N_GLOBAL, dg), nn.GELU(), nn.Linear(dg, dg), nn.GELU())

        self.node_in = nn.Sequential(
            nn.Linear(N_NODE_DYN + N_NODE_STATIC + pe_dim, d), nn.GELU(),
            nn.Linear(d, d))
        self.edge_in = nn.Sequential(
            nn.Linear(N_EDGE_DYN + N_EDGE_STATIC, d_edge), nn.GELU(),
            nn.Linear(d_edge, d_edge))
        self.gate_in = nn.Sequential(
            nn.Linear(N_GATE, d), nn.GELU(), nn.Linear(d, d))
        # gate tokens are bound to geometry by the node embeddings at their
        # two current physical locations (symmetrised: sum and |difference|)
        self.gate_bind = nn.Linear(2 * d, d)
        self.gate_pos = nn.Parameter(torch.randn(1, window, d) * 0.02)

        self.dist_bias = nn.Embedding(DIST_BUCKETS, n_heads)
        nn.init.zeros_(self.dist_bias.weight)
        self.gate_rel_bias = nn.Parameter(torch.zeros(n_heads, window, window))

        self.blocks = nn.ModuleList(
            GraphBlock(d, d_edge, n_heads, d_ff, dg, dropout) for _ in range(n_blocks))
        n_gate_blocks = max(1, n_blocks // max(1, gate_every))
        self.gate_blocks = nn.ModuleList(
            GateBlock(d, n_heads, d_ff, dg, dropout) for _ in range(n_gate_blocks))
        self.gate_every = gate_every

        self.ln_node = nn.LayerNorm(d)
        self.ln_gate = nn.LayerNorm(d)
        self.ln_edge = nn.LayerNorm(d_edge)
        self.pool = AttentionPool(d, n_heads)
        self.edge_pool = nn.Sequential(nn.Linear(d_edge, d), nn.GELU())

        d_pool = 3 * d + 2 * d + d + dg      # node(mean,max,attn) gate(first,mean) edge glob
        self.trunk = nn.Sequential(
            nn.Linear(d_pool, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU())
        self.head_swaps = nn.Linear(d, n_quantiles)
        self.head_depth = nn.Linear(d, n_quantiles)
        # start near zero cost-to-go so early search is not wildly misled
        for head in (self.head_swaps, self.head_depth):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -2.0)

    # -- forward -----------------------------------------------------------
    def forward(self, batch: dict, ctx: dict):
        """`batch` holds the per-state tensors, `ctx` the static graph tensors."""
        node, edge = batch["node"], batch["edge"]
        gate_x, glob = batch["gate"], batch["glob"]
        b, n, _ = node.shape

        g = self.glob_mlp(glob)

        ns = ctx["node_static"].unsqueeze(0).expand(b, -1, -1)
        pe = ctx["lap_pe"].unsqueeze(0).expand(b, -1, -1)
        if self.training and ctx.get("pe_flip", True):
            sign = torch.randint(0, 2, (b, 1, pe.shape[-1]), device=pe.device,
                                 dtype=pe.dtype) * 2 - 1
            pe = pe * sign
        h = self.node_in(torch.cat([node, ns, pe], -1))

        es = ctx["edge_static"].unsqueeze(0).expand(b, -1, -1)
        e = self.edge_in(torch.cat([edge, es], -1))

        gate = self.gate_in(gate_x) + self.gate_pos[:, :gate_x.shape[1]]
        gate_mask = batch["gate"][:, :, 0] > 0.5            # feature 0 is validity
        # keep at least one live key so attention never sees an all-masked row
        gate_mask = gate_mask | (torch.arange(gate_mask.shape[1],
                                              device=gate_mask.device) == 0)

        idx = torch.arange(b, device=node.device)[:, None]
        ha = h[idx, batch["gate_pa"]]
        hb = h[idx, batch["gate_pb"]]
        gate = gate + self.gate_bind(torch.cat([ha + hb, (ha - hb).abs()], -1))

        dist_bias = self.dist_bias(ctx["dist_bucket"]).permute(2, 0, 1).unsqueeze(0)
        rel_bias = self.gate_rel_bias[None, :, :gate.shape[1], :gate.shape[1]]

        gi = 0
        for i, blk in enumerate(self.blocks):
            h, e = blk(h, e, gate, g, ctx, dist_bias, gate_mask)
            if (i + 1) % self.gate_every == 0 and gi < len(self.gate_blocks):
                gate = self.gate_blocks[gi](gate, h, g, rel_bias, gate_mask)
                gi += 1

        h = self.ln_node(h)
        gate = self.ln_gate(gate)
        e = self.edge_pool(self.ln_edge(e))
        mask = gate_mask.unsqueeze(-1).to(gate.dtype)
        pooled = torch.cat([
            h.mean(1), h.amax(1), self.pool(h),
            gate[:, 0], (gate * mask).sum(1) / mask.sum(1).clamp(min=1.0),
            e.mean(1), g,
        ], -1)
        z = self.trunk(pooled)

        swaps = torch.cumsum(F.softplus(self.head_swaps(z)), dim=-1)
        depth = torch.cumsum(F.softplus(self.head_depth(z)), dim=-1)
        return {"swaps": swaps, "depth": depth, "value": swaps + 0.5 * depth}

    # -- convenience --------------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_preset(cls, name: str = "base", **overrides):
        cfg = dict(PRESETS[name])
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def from_checkpoint(cls, ck: dict):
        net = cls(**ck["cfg"])
        net.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        net.eval()
        return net


# ---------------------------------------------------------------------------
# inference: state list -> cost-to-go
# ---------------------------------------------------------------------------
def build_ctx(problem: Problem, ctx_graph: GraphContext, device, dtype):
    t = context_tensors(ctx_graph, device=device, dtype=dtype)
    deg = ctx_graph.adj.sum(1).clamp(min=1.0)
    t["deg_clamped"] = deg.to(device=device, dtype=dtype).view(1, -1, 1)
    t["pe_flip"] = False
    return t


def make_value_fn(net: RoutingNet, problem: Problem, device: str = "cpu",
                  quantile: int | None = None, max_batch: int = 512,
                  dtype: torch.dtype | None = None, cache: bool = True):
    """Returns `f(states) -> list[float]`, the predicted cost-to-go.

    Terminal states are forced to exactly 0 -- the search must never pay a
    hallucinated cost for having finished. Repeated states inside one call are
    deduplicated, and (optionally) memoised across calls, which matters because
    a beam regularly revisits the same mapping at the same gate.
    """
    dev = torch.device(device)
    if dtype is None:
        dtype = torch.float32
    net = net.to(device=dev, dtype=dtype).eval()
    gctx = GraphContext(problem, window=net.window)
    enc = Encoder(problem, gctx)
    ctx = build_ctx(problem, gctx, dev, dtype)
    q = net.n_quantiles // 2 if quantile is None else quantile
    q = max(0, min(net.n_quantiles - 1, q))
    memo: dict = {} if cache else None

    def _run(states):
        out = np.empty(len(states), dtype=np.float32)
        for i in range(0, len(states), max_batch):
            chunk = states[i:i + max_batch]
            batch = enc.encode(chunk)
            batch = {k: (v.to(dev) if v.dtype in (torch.int64,)
                         else v.to(device=dev, dtype=dtype))
                     for k, v in batch.items()}
            with torch.inference_mode():
                pred = net(batch, ctx)["value"][:, q]
            out[i:i + len(chunk)] = pred.float().cpu().numpy()
        return out

    def value_fn(states: list[State]) -> list[float]:
        if not states:
            return []
        vals = [0.0] * len(states)
        todo, todo_idx, keys = [], [], []
        seen: dict = {}
        for i, s in enumerate(states):
            if problem.is_terminal(s):
                continue
            key = (s.k, s.pos, s.tau, s.depth, s.swaps)
            if memo is not None and key in memo:
                vals[i] = memo[key]
                continue
            j = seen.get(key)
            if j is None:
                seen[key] = len(todo)
                todo.append(s)
                todo_idx.append([i])
                keys.append(key)
            else:
                todo_idx[j].append(i)
        if todo:
            preds = _run(todo)
            for j, idxs in enumerate(todo_idx):
                v = float(preds[j])
                if memo is not None:
                    memo[keys[j]] = v
                for i in idxs:
                    vals[i] = v
        return vals

    return value_fn


def load_net(path: str, device: str = "cpu") -> RoutingNet | None:
    """Load a checkpoint, tolerating a missing file or a torch without CUDA."""
    import os
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = RoutingNet.from_checkpoint(ck)
        return net.to(device)
    except Exception:
        return None
