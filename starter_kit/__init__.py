"""Starter-kit utilities for the computational track."""

from .baseline_routing import solve as baseline_solve
from .benchmarks import BENCHMARKS, benchmark_stats
from .hardware import HARDWARE_EDGES, HARDWARE_POSITIONS, build_hardware_graph
from .scorer import core_score, score_summary, schedule_layers_ordered, validate_routed_program
from .visualize import animate_layers, animate_routing, draw_hardware

__all__ = [
    "BENCHMARKS",
    "HARDWARE_EDGES",
    "HARDWARE_POSITIONS",
    "animate_layers",
    "animate_routing",
    "baseline_solve",
    "benchmark_stats",
    "build_hardware_graph",
    "core_score",
    "draw_hardware",
    "schedule_layers_ordered",
    "score_summary",
    "validate_routed_program",
]
