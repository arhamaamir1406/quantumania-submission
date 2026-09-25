"""QSITE 2026 Quantum Coalition -- Computational Track submission.

The required entry point:

    solve(program, hardware_graph) -> (initial_placement, routed_program)

Everything else lives in qroute/. Run `python -m qroute.eval` for the
benchmark table.
"""
from __future__ import annotations

import networkx as nx

from qroute.portfolio import solve as _portfolio_solve

DEFAULT_BUDGET = 60.0
# With a budget well past the default, programs too large for the
# whole-instance exact model switch to independent portfolio restarts
# (qroute/tube.py: solve_long), keeping the best. A single run plateaus by
# ~120 s; restarts keep paying. The challenge sets no time limit, so this is
# opt-in via `budget`.
LARGE_GATES = 24


def solve(program: list[tuple], hardware_graph: nx.Graph, budget: float = DEFAULT_BUDGET):
    """Place and route `program` onto `hardware_graph`.

    Args:
        program: list of ("2Q", i, j) and ("1Q", i) tuples on logical qubits.
        hardware_graph: networkx.Graph of physical qubit connectivity.
        budget: wall-clock seconds; the solver is anytime and returns the best
            candidate found within it. Never returns worse than the provided
            baseline, and never returns an invalid routing. Above 120 s,
            programs with more than 24 two-qubit gates get independent
            60 s restarts, keeping the best.

    Returns:
        (initial_placement, routed_program)
    """
    large = sum(1 for op in program if op[0] == "2Q") > LARGE_GATES
    if large and budget > 2 * DEFAULT_BUDGET:
        from qroute.tube import solve_long
        return solve_long(program, hardware_graph, budget=budget,
                          restart_budget=DEFAULT_BUDGET)
    if large:
        # Best of K parallel runs when the machine has the cores for it
        # (qroute/parallel.py); falls back to one run otherwise.
        from qroute.parallel import solve_parallel
        try:
            res = solve_parallel(program, hardware_graph, budget=budget)
        except Exception:
            res = None
        if res is not None:
            return res
    return _portfolio_solve(program, hardware_graph, budget=budget)


if __name__ == "__main__":
    from starter_kit.benchmarks import BENCHMARKS
    from starter_kit.hardware import build_hardware_graph
    from starter_kit.scorer import score_summary

    hw = build_hardware_graph()
    total = 0.0
    for name, prog in BENCHMARKS.items():
        placement, routed = solve(prog, hw)
        s = score_summary(prog, hw, placement, routed)
        total += s["score"]
        print(f"{name:15s} valid={s['valid']}  swaps={s['swap_count']:3d}  "
              f"depth={s['depth']:3d}  score={s['score']:6.1f}")
    print(f"{'TOTAL':15s} {total:.1f}")
