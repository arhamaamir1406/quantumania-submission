from __future__ import annotations

from collections.abc import Iterable

import networkx as nx


def used_logical_qubits(program: Iterable[tuple]) -> list[int]:
    logical_qubits = sorted({qubit for op in program for qubit in op[1:]})
    return logical_qubits


def validate_initial_placement(program: list[tuple], graph: nx.Graph, placement: dict[int, int]) -> tuple[bool, str]:
    logical_qubits = used_logical_qubits(program)
    if set(logical_qubits) != set(placement):
        return False, "placement must map every logical qubit used in the program"
    physical_qubits = list(placement.values())
    if len(physical_qubits) != len(set(physical_qubits)):
        return False, "placement must be injective"
    if not set(physical_qubits).issubset(set(graph.nodes)):
        return False, "placement uses physical qubits outside the hardware graph"
    return True, "ok"


def translate_back_to_logical(
    program: list[tuple], graph: nx.Graph, placement: dict[int, int], routed_program: list[tuple]
) -> tuple[bool, str, list[tuple]]:
    ok, message = validate_initial_placement(program, graph, placement)
    if not ok:
        return False, message, []

    physical_to_logical = {physical: logical for logical, physical in placement.items()}
    translated: list[tuple] = []

    for op in routed_program:
        kind = op[0]
        if kind == "SWAP":
            _, left, right = op
            if not graph.has_edge(left, right):
                return False, f"invalid SWAP on non-edge {(left, right)}", []
            left_logical = physical_to_logical.get(left)
            right_logical = physical_to_logical.get(right)
            physical_to_logical[left], physical_to_logical[right] = right_logical, left_logical
        elif kind == "2Q":
            _, left, right = op
            if not graph.has_edge(left, right):
                return False, f"2Q gate must act on an edge, got {(left, right)}", []
            if left not in physical_to_logical or right not in physical_to_logical:
                return False, "2Q gate uses an unoccupied physical qubit", []
            translated.append(("2Q", physical_to_logical[left], physical_to_logical[right]))
        elif kind == "1Q":
            _, qubit = op
            if qubit not in physical_to_logical:
                return False, "1Q gate uses an unoccupied physical qubit", []
            translated.append(("1Q", physical_to_logical[qubit]))
        else:
            return False, f"unknown operation kind: {kind}", []

    return True, "ok", translated


def validate_routed_program(
    program: list[tuple], graph: nx.Graph, placement: dict[int, int], routed_program: list[tuple]
) -> tuple[bool, str]:
    ok, message, translated = translate_back_to_logical(program, graph, placement, routed_program)
    if not ok:
        return False, message
    if translated != program:
        return False, "routed program does not preserve the original logical operation order"
    return True, "ok"


def schedule_layers_ordered(routed_program: list[tuple]) -> list[list[tuple]]:
    layers: list[list[tuple]] = []
    qubit_last_layer: dict[int, int] = {}

    for op in routed_program:
        if op[0] == "1Q":
            continue
        wires = op[1:]
        layer_index = 1 + max((qubit_last_layer.get(qubit, 0) for qubit in wires), default=0)
        while len(layers) < layer_index:
            layers.append([])
        layers[layer_index - 1].append(op)
        for qubit in wires:
            qubit_last_layer[qubit] = layer_index

    return layers


def core_score(routed_program: list[tuple]) -> float:
    swap_count = sum(1 for op in routed_program if op[0] == "SWAP")
    depth = len(schedule_layers_ordered(routed_program))
    return swap_count + 0.5 * depth


def score_summary(program: list[tuple], graph: nx.Graph, placement: dict[int, int], routed_program: list[tuple]) -> dict:
    valid, message = validate_routed_program(program, graph, placement, routed_program)
    layers = schedule_layers_ordered(routed_program)
    return {
        "valid": valid,
        "message": message,
        "swap_count": sum(1 for op in routed_program if op[0] == "SWAP"),
        "depth": len(layers),
        "score": core_score(routed_program) if valid else float("inf"),
        "layers": layers,
    }
