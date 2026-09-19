"""A small inductive GraphSAGE value function over the hardware graph.

Predicts cost-to-go (remaining swaps + 0.5 * remaining depth) for a routing
state, replacing the hand-written lookahead heuristic in rollouts and beam
search.

SAGE is chosen deliberately over an attention model: it is inductive, so a net
trained on this 20-qubit graph transfers to other topologies unchanged, and
mean aggregation over a degree-3 graph needs nothing more expressive.

Implemented densely (a row-normalised adjacency matmul) rather than via
PyTorch Geometric, so the only added dependency is torch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdp import Problem, State

N_NODE_FEATURES = 9
N_GLOBAL_FEATURES = 6


class SAGELayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.self_lin = nn.Linear(d_in, d_out)
        self.nbr_lin = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, a_norm):
        return self.self_lin(x) + self.nbr_lin(torch.matmul(a_norm, x))


class ValueNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        dims = [N_NODE_FEATURES] + [hidden] * layers
        self.convs = nn.ModuleList(SAGELayer(dims[i], dims[i + 1]) for i in range(layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + N_GLOBAL_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_node, x_glob, a_norm):
        h = x_node
        for conv in self.convs:
            h = torch.relu(conv(h, a_norm))
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-6)     # SAGE normalisation
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        out = self.head(torch.cat([pooled, x_glob], dim=-1)).squeeze(-1)
        return F.softplus(out)                                 # cost-to-go >= 0


class Encoder:
    """Vectorised state -> features. Encodes a whole beam in one pass."""

    def __init__(self, problem: Problem, window: int = 5):
        self.p = problem
        self.window = window
        n = problem.n_phys
        self.n = n
        nodes = problem.nodes
        self.dist = np.zeros((n, n), dtype=np.float32)
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                self.dist[i, j] = problem.dist[u][v]
        self.diam = max(1.0, float(self.dist.max()))
        deg = np.array([problem.hw.degree(u) for u in nodes], dtype=np.float32)
        self.deg = deg / max(1.0, float(deg.max()))

        a = np.zeros((n, n), dtype=np.float32)
        for u, v in problem.hw.edges:
            a[u, v] = 1.0
            a[v, u] = 1.0
        self.a_norm = torch.from_numpy(a / np.maximum(a.sum(1, keepdims=True), 1.0))

        # next 2Q gate at or after k involving each logical, and its partner there
        m, nl = problem.n_ops, max(1, problem.n_log)
        self.nxt = np.full((m + 1, nl), -1, dtype=np.int32)
        self.partner = np.full((m + 1, nl), -1, dtype=np.int32)
        for k in range(m - 1, -1, -1):
            self.nxt[k] = self.nxt[k + 1]
            self.partner[k] = self.partner[k + 1]
            op = problem.ops[k]
            if op[0] == "2Q":
                a_, b_ = op[1], op[2]
                self.nxt[k, a_] = k
                self.partner[k, a_] = b_
                self.nxt[k, b_] = k
                self.partner[k, b_] = a_

    def encode(self, states: list[State]):
        b, n = len(states), self.n
        xn = np.zeros((b, n, N_NODE_FEATURES), dtype=np.float32)
        xg = np.zeros((b, N_GLOBAL_FEATURES), dtype=np.float32)
        m = max(1, self.p.n_ops)
        for i, s in enumerate(states):
            occ = np.asarray(s.occ, dtype=np.int32)
            tau = np.asarray(s.tau, dtype=np.float32)
            pos = np.asarray(s.pos, dtype=np.int32)
            active = s.k < self.p.n_ops
            if active:
                op = self.p.ops[s.k]
                pa, pb = (int(pos[op[1]]), int(pos[op[2]])) if op[0] == "2Q" else (0, 0)
            else:
                pa = pb = 0
            live = occ >= 0
            tmax = float(tau.max())
            xn[i, :, 0] = live
            xn[i, :, 1] = (tmax - tau) / (tmax + 1.0)
            xn[i, pa, 2] = 1.0
            xn[i, pb, 3] = 1.0
            xn[i, :, 6] = self.deg
            xn[i, :, 7] = self.dist[:, pa] / self.diam
            xn[i, :, 8] = self.dist[:, pb] / self.diam
            if active:
                k = s.k
                for p in np.nonzero(live)[0]:
                    l = int(occ[p])
                    j = int(self.nxt[k, l])
                    if j < 0:
                        continue
                    part = int(self.partner[k, l])
                    xn[i, p, 4] = self.dist[p, int(pos[part])] / self.diam
                    xn[i, p, 5] = 1.0 / (1.0 + float(j - k))
            xg[i, 0] = (self.p.n_ops - s.k) / m
            xg[i, 1] = s.depth / m
            xg[i, 2] = s.swaps / m
            if active:
                xg[i, 3] = self.dist[pa, pb] / self.diam
                tot = cnt = 0.0
                for j in range(s.k, min(self.p.n_ops, s.k + self.window)):
                    o = self.p.ops[j]
                    if o[0] == "2Q":
                        tot += self.dist[int(pos[o[1]]), int(pos[o[2]])]
                        cnt += 1
                xg[i, 4] = (tot / cnt / self.diam) if cnt else 0.0
            xg[i, 5] = self.p.n_log / self.p.n_phys
        return torch.from_numpy(xn), torch.from_numpy(xg), self.a_norm


def make_value_fn(net: ValueNet, problem: Problem, device: str = "cpu"):
    """Returns f(states) -> list[float], an estimate of cost-to-go."""
    enc = Encoder(problem)
    net.eval()

    def value_fn(states: list[State]) -> list[float]:
        if not states:
            return []
        xn, xg, a = enc.encode(states)
        with torch.no_grad():
            out = net(xn.to(device), xg.to(device), a.to(device))
        return out.cpu().numpy().tolist()

    return value_fn
