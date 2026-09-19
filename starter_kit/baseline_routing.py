from __future__ import annotations

import networkx as nx

from .scorer import used_logical_qubits


def identity_placement(program: list[tuple], hardware_graph: nx.Graph) -> dict[int, int]:
    logical_qubits = used_logical_qubits(program)
    physical_qubits = sorted(hardware_graph.nodes)
    return {logical: physical_qubits[index] for index, logical in enumerate(logical_qubits)}


def solve(program: list[tuple], hardware_graph: nx.Graph) -> tuple[dict[int, int], list[tuple]]:
    """Intentionally bad baseline: identity-style placement + greedy shortest-path routing."""
    placement = identity_placement(program, hardware_graph)
    routed_program: list[tuple] = []
    physical_to_logical = {physical: logical for logical, physical in placement.items()}

    for op in program:
        kind = op[0]
        if kind == "1Q":
            logical = op[1]
            routed_program.append(("1Q", placement[logical]))
            continue

        _, logical_left, logical_right = op
        physical_left = placement[logical_left]
        physical_right = placement[logical_right]

        if not hardware_graph.has_edge(physical_left, physical_right):
            path = nx.shortest_path(hardware_graph, physical_left, physical_right)
            for left, right in zip(path[:-2], path[1:-1]):
                left_logical = physical_to_logical.get(left)
                right_logical = physical_to_logical.get(right)
                routed_program.append(("SWAP", left, right))
                physical_to_logical[left], physical_to_logical[right] = right_logical, left_logical
                if left_logical is not None:
                    placement[left_logical] = right
                if right_logical is not None:
                    placement[right_logical] = left

        routed_program.append(("2Q", placement[logical_left], placement[logical_right]))

    return identity_placement(program, hardware_graph), routed_program
