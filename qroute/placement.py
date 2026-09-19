"""Initial placement strategies (logical -> physical)."""
from __future__ import annotations

import random

import networkx as nx
from networkx.algorithms import isomorphism


def interaction_graph(program: list[tuple], weighted: bool = True) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from({q for op in program for q in op[1:]})
    for op in program:
        if op[0] != "2Q":
            continue
        a, b = op[1], op[2]
        if g.has_edge(a, b):
            g[a][b]["weight"] += 1
        else:
            g.add_edge(a, b, weight=1)
    return g


def embed_placement(program: list[tuple], hw: nx.Graph) -> dict[int, int] | None:
    """Exact zero-SWAP placement via subgraph monomorphism, or None.

    Two O(1) rejections first: an interaction graph with more edges, or a
    higher maximum degree, than the hardware cannot possibly embed. Those
    kill the expensive VF2 searches on dense programs.
    """
    inter = interaction_graph(program)
    if inter.number_of_edges() > hw.number_of_edges():
        return None
    if inter.number_of_nodes() > hw.number_of_nodes():
        return None
    if inter.number_of_edges() and max(dict(inter.degree()).values()) > max(dict(hw.degree()).values()):
        return None
    gm = isomorphism.GraphMatcher(hw, inter)
    if not gm.subgraph_is_monomorphic():
        return None
    return {logical: phys for phys, logical in gm.mapping.items()}


def identity_placement(program: list[tuple], hw: nx.Graph) -> dict[int, int]:
    logicals = sorted({q for op in program for q in op[1:]})
    nodes = sorted(hw.nodes)
    return {l: nodes[i] for i, l in enumerate(logicals)}


def constructive_placement(program: list[tuple], hw: nx.Graph, rng: random.Random | None = None,
                           jitter: float = 0.0) -> dict[int, int]:
    """Grow a placement outward from the busiest logical qubit.

    Repeatedly take the unplaced logical with the most interaction weight to
    already-placed qubits, and give it the free physical qubit minimising the
    weighted distance to those neighbours.
    """
    inter = interaction_graph(program)
    dist = dict(nx.all_pairs_shortest_path_length(hw))
    centrality = {p: sum(dist[p].values()) for p in hw.nodes}
    traffic = {n: sum(d["weight"] for _, _, d in inter.edges(n, data=True)) for n in inter.nodes}

    placed: dict[int, int] = {}
    free = set(hw.nodes)
    order_seed = max(inter.nodes, key=lambda n: (traffic[n], -n)) if inter.nodes else None
    if order_seed is None:
        return {}
    hub = min(free, key=lambda p: (centrality[p], p))
    placed[order_seed] = hub
    free.discard(hub)

    remaining = set(inter.nodes) - {order_seed}
    while remaining:
        def bound(n):
            return sum(inter[n][m]["weight"] for m in inter.neighbors(n) if m in placed)
        nxt = max(remaining, key=lambda n: (bound(n), traffic[n], -n))
        best, best_cost = None, None
        for p in free:
            c = 0.0
            for m in inter.neighbors(nxt):
                if m in placed:
                    c += inter[nxt][m]["weight"] * dist[p][placed[m]]
            c += 0.01 * centrality[p]
            if jitter and rng is not None:
                c += rng.random() * jitter
            if best_cost is None or c < best_cost:
                best, best_cost = p, c
        placed[nxt] = best
        free.discard(best)
        remaining.discard(nxt)
    return placed


def random_placement(program: list[tuple], hw: nx.Graph, rng: random.Random) -> dict[int, int]:
    logicals = sorted({q for op in program for q in op[1:]})
    nodes = rng.sample(sorted(hw.nodes), len(logicals))
    return dict(zip(logicals, nodes))
