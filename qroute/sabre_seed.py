"""Starting placements from Qiskit's SABRE layout (optional; needs qiskit).

SABRE routes by the circuit's dependency DAG, so its own routings often break
the challenge's exact-order rule and cannot be submitted. Its *initial layouts*
are still good starting points: we take the layouts of its cheapest runs and
route them ourselves, in program order, like any other placement.
"""
from __future__ import annotations

import time

import networkx as nx


def available() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except ImportError:
        return False


def sabre_placements(program: list[tuple], hw: nx.Graph, deadline: float, top: int = 24,
                     seed: int = 0, trials: int = 20) -> list[dict[int, int]]:
    """Distinct SABRE initial layouts, ranked by SABRE's own (order-relaxed) score."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import SabreLayout
    from starter_kit.scorer import core_score

    nodes = sorted(hw.nodes)
    idx = {p: i for i, p in enumerate(nodes)}
    cmap = CouplingMap([[idx[a], idx[b]] for a, b in hw.edges]
                       + [[idx[b], idx[a]] for a, b in hw.edges])
    logicals = sorted({q for op in program for q in op[1:]})
    li = {q: i for i, q in enumerate(logicals)}
    qc = QuantumCircuit(len(logicals))
    for k, op in enumerate(program):
        if op[0] == "2Q":
            # Opaque, uniquely named gates: nothing can merge or cancel them.
            qc.append(Gate(f"g{k}", 2, []), [li[op[1]], li[op[2]]])
    if not qc.data:
        return []
    qidx = {q: i for i, q in enumerate(qc.qubits)}

    found: dict[tuple, float] = {}
    s = seed
    while time.monotonic() < deadline:
        pm = PassManager([SabreLayout(cmap, seed=s, max_iterations=4,
                                      swap_trials=trials, layout_trials=trials)])
        routed = pm.run(qc)
        s += 1
        layout = pm.property_set["layout"]
        pl = {}
        for vbit, p in layout.get_virtual_bits().items():
            if vbit in qidx:
                pl[logicals[qidx[vbit]]] = nodes[p]
        raw = [("SWAP" if inst.operation.name == "swap" else "2Q",)
               + tuple(nodes[routed.find_bit(q).index] for q in inst.qubits)
               for inst in routed.data]
        key = tuple(sorted(pl.items()))
        cost = core_score(raw)
        if cost < found.get(key, float("inf")):
            found[key] = cost
    ranked = sorted(found.items(), key=lambda kv: kv[1])[:top]
    return [dict(k) for k, _ in ranked]
