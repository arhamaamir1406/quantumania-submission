from __future__ import annotations

import networkx as nx

HARDWARE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (2, 6),
    (5, 9),
    (7, 11),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (10, 14),
    (12, 13),
    (13, 14),
    (14, 15),
    (13, 17),
    (15, 19),
    (16, 17),
    (17, 18),
    (18, 19),
]

HARDWARE_POSITIONS = {
    0: (0.0, 4.0),
    1: (1.0, 4.0),
    2: (2.0, 4.0),
    3: (3.0, 4.0),
    4: (0.0, 3.0),
    5: (1.0, 3.0),
    6: (2.0, 3.0),
    7: (3.0, 3.0),
    8: (0.0, 2.0),
    9: (1.0, 2.0),
    10: (2.0, 2.0),
    11: (3.0, 2.0),
    12: (0.0, 1.0),
    13: (1.0, 1.0),
    14: (2.0, 1.0),
    15: (3.0, 1.0),
    16: (0.0, 0.0),
    17: (1.0, 0.0),
    18: (2.0, 0.0),
    19: (3.0, 0.0),
}


def build_hardware_graph() -> nx.Graph:
    """Return the 20-qubit heavy-hex-style teaching graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    graph.add_edges_from(HARDWARE_EDGES)
    return graph


def hardware_stats() -> dict[str, int]:
    graph = build_hardware_graph()
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "max_degree": max(dict(graph.degree()).values()),
        "diameter": nx.diameter(graph),
    }
