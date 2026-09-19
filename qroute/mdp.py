"""The routing MDP whose return is exactly swaps + 0.5 * depth.

State carries the scheduler's clock (tau: per-physical-qubit last-used layer),
which is what lets the search reason about the depth term that distance-based
heuristics are blind to.

Step cost:
    SWAP  ->  1 + 0.5 * max(0, new_depth - old_depth)
    GATE  ->      0.5 * max(0, new_depth - old_depth)

Summed over an episode this is identically swaps + 0.5 * depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True, slots=True)
class State:
    pos: tuple          # logical -> physical
    occ: tuple          # physical -> logical, or -1
    tau: tuple          # physical -> last layer used
    k: int              # index of next unexecuted program op
    swaps: int
    depth: int
    chain: tuple | None  # cons list of ops, reversed

    @property
    def cost(self) -> float:
        return self.swaps + 0.5 * self.depth

    def ops(self) -> list[tuple]:
        out = []
        node = self.chain
        while node is not None:
            out.append(node[1])
            node = node[0]
        out.reverse()
        return out


class Problem:
    """Static per-(program, hardware) data shared by all states."""

    def __init__(self, program: list[tuple], hw: nx.Graph):
        self.program = program
        self.hw = hw
        self.nodes = sorted(hw.nodes)
        self.n_phys = len(self.nodes)
        self.logicals = sorted({q for op in program for q in op[1:]})
        self.n_log = len(self.logicals)
        self.lidx = {l: i for i, l in enumerate(self.logicals)}
        self.dist = dict(nx.all_pairs_shortest_path_length(hw))
        self.nbrs = {p: tuple(hw.neighbors(p)) for p in self.nodes}
        self.adj = {p: set(hw.neighbors(p)) for p in self.nodes}
        # program normalised to internal logical indices
        self.ops: list[tuple] = []
        for op in program:
            if op[0] == "2Q":
                self.ops.append(("2Q", self.lidx[op[1]], self.lidx[op[2]]))
            else:
                self.ops.append(("1Q", self.lidx[op[1]]))
        self.n_ops = len(self.ops)

    def initial(self, placement: dict[int, int]) -> State:
        pos = [0] * self.n_log
        for l, p in placement.items():
            pos[self.lidx[l]] = p
        occ = [-1] * self.n_phys
        for i, p in enumerate(pos):
            occ[p] = i
        return self.advance(State(tuple(pos), tuple(occ), (0,) * self.n_phys,
                                  0, 0, 0, None))

    def d(self, p: int, q: int) -> int:
        return self.dist[p][q]

    def advance(self, s: State) -> State:
        """Execute every op that is immediately executable (1Q, or adjacent 2Q)."""
        pos, tau, k, depth, chain = s.pos, s.tau, s.k, s.depth, s.chain
        while k < self.n_ops:
            op = self.ops[k]
            if op[0] == "1Q":
                chain = (chain, ("GATE", k))
                k += 1
                continue
            _, a, b = op
            pa, pb = pos[a], pos[b]
            if pb not in self.adj[pa]:
                break
            layer = 1 + max(tau[pa], tau[pb])
            t = list(tau)
            t[pa] = t[pb] = layer
            tau = tuple(t)
            depth = max(depth, layer)
            chain = (chain, ("GATE", k))
            k += 1
        return State(pos, s.occ, tau, k, s.swaps, depth, chain)

    def apply_swap(self, s: State, p: int, q: int) -> State:
        pos = list(s.pos)
        occ = list(s.occ)
        a, b = occ[p], occ[q]
        if a >= 0:
            pos[a] = q
        if b >= 0:
            pos[b] = p
        occ[p], occ[q] = b, a
        layer = 1 + max(s.tau[p], s.tau[q])
        tau = list(s.tau)
        tau[p] = tau[q] = layer
        return self.advance(State(tuple(pos), tuple(occ), tuple(tau), s.k,
                                  s.swaps + 1, max(s.depth, layer),
                                  (s.chain, ("SWAP", p, q))))

    def actions(self, s: State) -> list[tuple[int, int]]:
        """SWAP candidates: edges incident to either qubit of the active gate."""
        op = self.ops[s.k]
        _, a, b = op
        pa, pb = s.pos[a], s.pos[b]
        cands = set()
        for p in (pa, pb):
            for n in self.nbrs[p]:
                cands.add((p, n) if p < n else (n, p))
        return sorted(cands)

    def is_terminal(self, s: State) -> bool:
        return s.k >= self.n_ops

    def heuristic(self, s: State, window: int = 12, weight: float = 0.5) -> float:
        """Optimistic-ish cost-to-go: active gate distance plus decayed lookahead."""
        if s.k >= self.n_ops:
            return 0.0
        _, a, b = self.ops[s.k]
        d0 = self.d(s.pos[a], s.pos[b])
        h = (d0 - 1) + 0.5 * ((d0 - 1 + 1) // 2)
        tot, n = 0.0, 0
        for j in range(s.k + 1, min(self.n_ops, s.k + 1 + window)):
            o = self.ops[j]
            if o[0] != "2Q":
                continue
            tot += max(0, self.d(s.pos[o[1]], s.pos[o[2]]) - 1)
            n += 1
        if n:
            h += weight * tot / n
        return h
