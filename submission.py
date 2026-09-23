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
# whole-instance exact model switch to portfolio restarts plus tube LNS
# (qroute/tube.py), which can keep improving after the portfolio alone has
# plateaued. The challenge sets no time limit, so this is opt-in via `budget`.
LARGE_GATES = 24


def solve(program: list[tuple], hardware_graph: nx.Graph, budget: float = DEFAULT_BUDGET):
    """Place and route `program` onto `hardware_graph`.

    Args:
        program: list of ("2Q", i, j) and ("1Q", i) tuples on logical qubits.
        hardware_graph: networkx.Graph of physical qubit connectivity.
        budget: wall-clock seconds; the solver is anytime and returns the best
            candidate found within it. Never returns worse than the provided
            baseline, and never returns an invalid routing. Above 120 s,
            programs with more than 24 two-qubit gates also get tube LNS.

    Returns:
        (initial_placement, routed_program)
    """
    large = sum(1 for op in program if op[0] == "2Q") > LARGE_GATES
    if large and budget > 2 * DEFAULT_BUDGET:
        from qroute import watermark
        if watermark.HAVE_ORTOOLS:
            from qroute.tube import solve_long
            return solve_long(program, hardware_graph, budget=budget,
                              restart_budget=DEFAULT_BUDGET)
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
