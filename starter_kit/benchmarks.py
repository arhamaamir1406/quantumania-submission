from __future__ import annotations

import random

def _star_program(center: int, leaves: range) -> list[tuple[str, int, int]]:
    return [("2Q", center, leaf) for leaf in leaves]


def _chain_program(length: int) -> list[tuple[str, int, int]]:
    return [("2Q", i, i + 1) for i in range(length - 1)]


def _ladder_program(rows: int, cols: int) -> list[tuple[str, int, int]]:
    program: list[tuple[str, int, int]] = []
    top = list(range(cols))
    bottom = list(range(cols, cols * 2))
    for row in (top, bottom):
        program.extend(("2Q", row[i], row[i + 1]) for i in range(cols - 1))
    program.extend(("2Q", top[i], bottom[i]) for i in range(cols))
    return program


def _random_pairs(num_qubits: int, num_pairs: int, seed: int) -> list[tuple[str, int, int]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, int, int]] = []
    seen = set()
    while len(pairs) < num_pairs:
        a, b = sorted(rng.sample(range(num_qubits), 2))
        if (a, b, len(pairs) % 3) in seen:
            continue
        seen.add((a, b, len(pairs) % 3))
        pairs.append(("2Q", a, b))
    return pairs


def _repeating_layers(width: int, repeats: int) -> list[tuple[str, int, int]]:
    layer = [("2Q", i, i + 1) for i in range(0, width - 1, 2)]
    bridge = [("2Q", i, i + 1) for i in range(1, width - 1, 2)]
    program: list[tuple[str, int, int]] = []
    for _ in range(repeats):
        program.extend(layer)
        program.extend(bridge)
    return program


BENCHMARKS = {
    "ghz_star": _star_program(0, range(1, 8)),
    "chain_trotter": _chain_program(10),
    "ladder_trotter": _ladder_program(2, 6),
    "qaoa_random": _random_pairs(12, 18, seed=7),
    "dense_random": _random_pairs(14, 40, seed=17),
    "vqe_layers": _repeating_layers(16, repeats=3),
}


def benchmark_stats(program: list[tuple]) -> dict[str, int]:
    logical_qubits = sorted({qubit for op in program for qubit in op[1:]})
    two_qubit_ops = sum(1 for op in program if op[0] == "2Q")
    one_qubit_ops = sum(1 for op in program if op[0] == "1Q")
    return {
        "logical_qubits": len(logical_qubits),
        "two_qubit_ops": two_qubit_ops,
        "one_qubit_ops": one_qubit_ops,
    }
