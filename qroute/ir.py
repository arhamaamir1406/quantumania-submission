"""Solution IR.

A solution is (placement, ops) where ops is a list of
    ("SWAP", p, q)   physical SWAP
    ("GATE", k)      execute program[k] wherever its logical qubits currently sit

Physical gate arguments are never stored -- they are derived in materialize().
That makes every edit (delete/move/insert a SWAP) safe, and makes the
argument-order trap in the scorer structurally impossible.
"""
from __future__ import annotations

import networkx as nx

from starter_kit.scorer import core_score, validate_routed_program

SWAP = "SWAP"
GATE = "GATE"


def materialize(program: list[tuple], placement: dict[int, int], ops: list[tuple]) -> list[tuple]:
    pos = dict(placement)                       # logical -> physical
    occ = {p: l for l, p in placement.items()}  # physical -> logical
    out: list[tuple] = []
    for op in ops:
        if op[0] == SWAP:
            _, p, q = op
            a, b = occ.get(p), occ.get(q)
            if a is not None:
                pos[a] = q
            if b is not None:
                pos[b] = p
            occ[p], occ[q] = b, a
            out.append((SWAP, p, q))
        else:
            g = program[op[1]]
            if g[0] == "2Q":
                out.append(("2Q", pos[g[1]], pos[g[2]]))
            else:
                out.append(("1Q", pos[g[1]]))
    return out


def evaluate(program: list[tuple], hw: nx.Graph, placement: dict[int, int], ops: list[tuple]):
    """Return (score, routed_program, message). score is inf if invalid."""
    routed = materialize(program, placement, ops)
    ok, msg = validate_routed_program(program, hw, placement, routed)
    if not ok:
        return float("inf"), routed, msg
    return core_score(routed), routed, "ok"
