"""Regression tests for the exact watermark model and the LNS built on it.

Every routing is checked by the organisers' scorer, never by our own code.
Run with `python -m pytest tests -q` (needs ortools).
"""
from __future__ import annotations

import pytest

from starter_kit.benchmarks import BENCHMARKS
from starter_kit.hardware import build_hardware_graph
from starter_kit.scorer import score_summary

from qroute.ir import materialize
from qroute.search import beam_search
from qroute.mdp import Problem
from qroute.placement import constructive_placement

watermark = pytest.importorskip("qroute.watermark")
if not watermark.HAVE_ORTOOLS:
    pytest.skip("ortools not installed", allow_module_level=True)

HW = build_hardware_graph()


def _check(program, placement, ops):
    routed = materialize(program, placement, ops)
    s = score_summary(program, HW, placement, routed)
    assert s["valid"], s["message"]
    return s["score"]


def _beam(program):
    pl = constructive_placement(program, HW)
    p = Problem(program, HW)
    s = beam_search(p, p.initial(pl), width=16)
    return pl, s.ops()


def test_ghz_star_is_solved_to_its_proven_optimum():
    prog = BENCHMARKS["ghz_star"]
    r = watermark.solve_exact(prog, HW, time_limit=120, horizon=11, upper=7.5, workers=8)
    assert r.optimal and r.score == 6.5
    assert _check(prog, r.placement, r.ops) == 6.5


def test_order_rule_is_enforced_with_parallel_independent_gates():
    # gate 2 is independent of gates 0-1 and may share their first layer, but
    # the routed list must still present it third.
    prog = [("2Q", 0, 1), ("2Q", 0, 1), ("2Q", 2, 3), ("2Q", 0, 2), ("1Q", 3), ("2Q", 1, 3)]
    r = watermark.solve_exact(prog, HW, time_limit=30, horizon=8, workers=8)
    assert r.optimal
    assert _check(prog, r.placement, r.ops) == r.score


def test_relaxed_model_is_a_lower_bound():
    prog = BENCHMARKS["ladder_trotter"]
    r = watermark.solve_exact(prog, HW, time_limit=30, horizon=8, order=False, workers=8)
    assert r.bound <= 6.5              # 6.5 is proven optimal for the real problem


@pytest.mark.parametrize("name", ["qaoa_random", "dense_random"])
def test_tube_never_worsens_and_stays_valid(name):
    prog = BENCHMARKS[name]
    pl, ops = _beam(prog)
    inc = _check(prog, pl, ops)
    routed = materialize(prog, pl, ops)
    r = watermark.solve_exact(prog, HW, time_limit=20, hint=(pl, routed), upper=inc,
                              tube=2, workers=8)
    assert r.ops is not None
    assert _check(prog, r.placement, r.ops) <= inc
