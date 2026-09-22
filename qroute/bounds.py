"""Provable lower bounds on the score of an instance.

These are not estimates. Every bound here is an argument about what *any*
valid routing must pay, so the gap between a solution and the bound is a real
optimality gap, and a claim to have beaten the bound is a claim to have broken
the rules.

SWAP bounds
-----------
Let `D` be the hardware's maximum degree (3 here) and let the *interaction
graph* have an edge for every distinct logical pair the program makes interact.

1. **Degree bound.** Logical qubit `v` must, at some moment, be adjacent to
   each of its `d(v)` distinct partners. At any instant it is adjacent to at
   most `D` qubits, and one SWAP adds at most `D - 1` qubits to its
   neighbourhood:

   * if `v` is one of the swapped qubits it moves to a node of degree `<= D`,
     one of whose neighbours is the node it just left, so at most `D - 1` of
     its new neighbours are new;
   * otherwise at most one of the two swapped nodes is adjacent to `v`, so at
     most one qubit newly arrives beside it.

   Hence `swaps >= ceil((d(v) - D) / (D - 1))` for every `v`.

2. **Edge-count bound.** Each SWAP moves two qubits, and each of them acquires
   at most `D - 1` partners it did not already have, so a SWAP realises at most
   `2 (D - 1)` previously-unrealised logical pairs. The initial placement
   realises at most `|E_hw|` of them. Hence
   `swaps >= ceil((|E_int| - min(|E_int|, |E_hw|)) / (2 (D - 1)))`.

3. **Embeddability.** If the interaction graph is not subgraph-monomorphic to
   the hardware graph, no placement makes every required pair adjacent at once,
   so at least one SWAP is required: `swaps >= 1`.

Depth bound
-----------
Physical depth is at least the logical program's ASAP depth: the scheduler may
only add layers, never remove them, and inserted SWAPs cannot make two gates
sharing a qubit commute into one layer.

Coupled hub bound
-----------------
The bounds above treat SWAPs and depth separately. For a logical qubit `v`
with `g(v)` gates and `d(v)` distinct partners, let `c` be the number of SWAPs
that move `v`. Every op touching `v` -- its gates and the SWAPs that move it
-- shares the physical qubit `v` occupies at that moment with the previous
such op, so they are serialised: `depth >= g(v) + c`. By the degree argument,
the `c` moving SWAPs add at most `(D - 1) c` new neighbours and every other
SWAP at most one, so at least `max(0, d(v) - D - (D - 1) c)` other SWAPs are
needed. Any solution has *some* `c`, so

    score >= min over c >= 0 of
             max(c + max(0, d(v) - D - (D-1) c), S_other)
             + 0.5 * max(g(v) + c, asap_depth)

where `S_other` is the edge-count/embeddability bound. Moving a hub is
cheaper per new partner (1.5 per `D - 1`) than bringing partners to it (1
each), but only up to the point where the extra depth outweighs it. This
bound proves `ghz_star` optimal at 6.5.

The final floor is the maximum of the coupled bound and
`max(degree, edge, embed) + 0.5 * asap_depth`.
"""
from __future__ import annotations

import math

import networkx as nx
from networkx.algorithms import isomorphism

from starter_kit.scorer import schedule_layers_ordered

from .placement import interaction_graph


def asap_depth(program: list[tuple]) -> int:
    """Depth of the logical program itself -- the floor on physical depth."""
    return len(schedule_layers_ordered(program))


def swap_lower_bound(program: list[tuple], hw: nx.Graph) -> tuple[int, str]:
    """A provable minimum SWAP count, with the argument that produced it."""
    inter = interaction_graph(program)
    if inter.number_of_edges() == 0:
        return 0, "no 2Q gates"
    degrees = dict(hw.degree())
    D = max(degrees.values()) if degrees else 0
    if D <= 1:
        return 0, "degenerate hardware"

    best, why = 0, "none"

    # 1. degree bound
    for v, d in inter.degree():
        if d > D:
            need = math.ceil((d - D) / (D - 1))
            if need > best:
                best, why = need, f"logical q{v} has {d} distinct partners > max degree {D}"

    # 2. edge-count bound
    ei, eh = inter.number_of_edges(), hw.number_of_edges()
    if ei > eh:
        need = math.ceil((ei - eh) / (2 * (D - 1)))
        if need > best:
            best, why = need, f"{ei} distinct pairs > {eh} hardware edges"

    # 3. embeddability
    if best == 0:
        if inter.number_of_nodes() > hw.number_of_nodes():
            return 1, "more logical qubits than physical"
        gm = isomorphism.GraphMatcher(hw, inter)
        if not gm.subgraph_is_monomorphic():
            return 1, "interaction graph does not embed in the hardware graph"
        return 0, "interaction graph embeds -- zero SWAPs is achievable"
    return best, why


def hub_bound(program: list[tuple], hw: nx.Graph, s_other: int, depth: int) -> tuple[float, str]:
    """The coupled SWAP/depth bound (see module docstring)."""
    inter = interaction_graph(program)
    D = max(dict(hw.degree()).values())
    gates = {}
    for op in program:
        if op[0] == "2Q":
            for q in op[1:]:
                gates[q] = gates.get(q, 0) + 1
    best, why = 0.0, ""
    for v, d in inter.degree():
        if d <= D:
            continue
        f = min(max(c + max(0, d - D - (D - 1) * c), s_other) + 0.5 * max(gates[v] + c, depth)
                for c in range(0, d + 1))
        if f > best:
            best, why = f, f"hub q{v}: {d} partners, {gates[v]} gates, moves trade SWAPs for depth"
    return best, why


def edge_embed_bound(program: list[tuple], hw: nx.Graph) -> int:
    """The SWAP floor from edge count and embeddability alone (no degree term)."""
    inter = interaction_graph(program)
    D = max(dict(hw.degree()).values())
    ei, eh = inter.number_of_edges(), hw.number_of_edges()
    if ei > eh:
        return math.ceil((ei - eh) / (2 * (D - 1)))
    if ei and inter.number_of_nodes() <= hw.number_of_nodes():
        if not isomorphism.GraphMatcher(hw, inter).subgraph_is_monomorphic():
            return 1
    return 0


def score_lower_bound(program: list[tuple], hw: nx.Graph) -> tuple[float, int, int, str]:
    """-> (floor, swap_floor, depth_floor, reason)."""
    s, why = swap_lower_bound(program, hw)
    d = asap_depth(program)
    floor = s + 0.5 * d
    hb, hwhy = hub_bound(program, hw, edge_embed_bound(program, hw), d)
    if hb > floor:
        floor, why = hb, hwhy
    return floor, s, d, why


def report(hw: nx.Graph | None = None) -> None:
    """Print the bound table for the six public benchmarks."""
    from starter_kit.benchmarks import BENCHMARKS
    from starter_kit.hardware import build_hardware_graph

    hw = hw or build_hardware_graph()
    hdr = f"{'benchmark':15s} {'swap>=':>7s} {'depth>=':>8s} {'score>=':>8s}  why"
    print(hdr)
    print("-" * (len(hdr) + 20))
    tot = ts = 0.0
    for name, prog in BENCHMARKS.items():
        floor, s, d, why = score_lower_bound(prog, hw)
        tot += floor
        ts += s
        print(f"{name:15s} {s:7d} {d:8d} {floor:8.1f}  {why}")
    print("-" * (len(hdr) + 20))
    print(f"{'TOTAL':15s} {ts:7.0f} {'':8s} {tot:8.1f}")


if __name__ == "__main__":
    report()
