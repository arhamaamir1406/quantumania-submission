"""Benchmark harness: score the portfolio against the provided baseline."""
from __future__ import annotations

import argparse
import time

from starter_kit.baseline_routing import solve as baseline_solve
from starter_kit.benchmarks import BENCHMARKS, benchmark_stats
from starter_kit.hardware import build_hardware_graph
from starter_kit.scorer import schedule_layers_ordered, score_summary

from .portfolio import solve

FLOOR = {"ghz_star": 3.5, "chain_trotter": 4.5, "ladder_trotter": 3.0,
         "qaoa_random": 4.0, "dense_random": 6.0, "vqe_layers": 3.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--beam", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--only", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    hw = build_hardware_graph()
    hdr = f"{'benchmark':15s} {'2Q':>4s} {'base':>7s} {'ours':>7s} {'swaps':>6s} {'depth':>6s} {'floor':>6s} {'gap':>6s} {'sec':>6s}"
    print(hdr)
    print("-" * len(hdr))

    tb = to = tf = 0.0
    for name, prog in BENCHMARKS.items():
        if args.only and args.only != name:
            continue
        bpl, brt = baseline_solve(list(prog), hw)
        base = score_summary(prog, hw, bpl, brt)["score"]
        if args.verbose:
            print(f"  == {name}")
        t0 = time.monotonic()
        pl, rt = solve(prog, hw, budget=args.budget, seeds=args.seeds,
                       beam_width=args.beam, verbose=args.verbose)
        el = time.monotonic() - t0
        s = score_summary(prog, hw, pl, rt)
        assert s["valid"], f"{name}: {s['message']}"
        floor = FLOOR.get(name, 0.5 * len(schedule_layers_ordered(prog)))
        tb += base; to += s["score"]; tf += floor
        print(f"{name:15s} {benchmark_stats(prog)['two_qubit_ops']:4d} {base:7.1f} "
              f"{s['score']:7.1f} {s['swap_count']:6d} {s['depth']:6d} {floor:6.1f} "
              f"{s['score']-floor:6.1f} {el:6.1f}")

    print("-" * len(hdr))
    print(f"{'TOTAL':15s} {'':4s} {tb:7.1f} {to:7.1f} {'':6s} {'':6s} {tf:6.1f} {to-tf:6.1f}")
    if tb:
        print(f"\nimprovement over baseline: {tb-to:.1f} points ({100*(tb-to)/tb:.1f}%)")


if __name__ == "__main__":
    main()
