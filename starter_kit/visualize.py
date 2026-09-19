from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

from .hardware import HARDWARE_POSITIONS, build_hardware_graph


def draw_hardware(
    graph: nx.Graph | None = None,
    placement: dict[int, int] | None = None,
    highlight_edges: list[tuple[int, int]] | None = None,
    ax=None,
    title: str | None = None,
):
    graph = graph or build_hardware_graph()
    ax = ax or plt.gca()
    highlight_edges = highlight_edges or []

    nx.draw_networkx_edges(graph, HARDWARE_POSITIONS, ax=ax, edge_color="#94a3b8", width=1.8)
    if highlight_edges:
        nx.draw_networkx_edges(
            graph,
            HARDWARE_POSITIONS,
            ax=ax,
            edgelist=highlight_edges,
            edge_color="#dc2626",
            width=3.5,
        )
    nx.draw_networkx_nodes(graph, HARDWARE_POSITIONS, ax=ax, node_color="#e2e8f0", node_size=820)
    nx.draw_networkx_labels(graph, HARDWARE_POSITIONS, ax=ax, labels={node: str(node) for node in graph.nodes}, font_size=9)

    if placement:
        for logical, physical in placement.items():
            x, y = HARDWARE_POSITIONS[physical]
            ax.text(x, y + 0.22, f"L{logical}", ha="center", va="center", fontsize=11, color="#b91c1c", fontweight="bold")

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def animate_placements(
    frames: list[dict[int, int]],
    highlight_edges: list[tuple[int, int]] | None = None,
    title_prefix: str = "Routing frame",
    interval: int = 700,
):
    graph = build_hardware_graph()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    def _update(index: int):
        ax.clear()
        draw_hardware(
            graph=graph,
            placement=frames[index],
            highlight_edges=highlight_edges,
            ax=ax,
            title=f"{title_prefix} {index}",
        )
        return ax.collections + ax.lines + ax.texts

    return FuncAnimation(fig, _update, frames=len(frames), interval=interval, blit=False)


def animate_layers(layers: list[list[tuple]], interval: int = 900):
    graph = build_hardware_graph()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    def _update(index: int):
        ax.clear()
        highlight_edges = [tuple(op[1:]) for op in layers[index]]
        draw_hardware(graph=graph, highlight_edges=highlight_edges, ax=ax, title=f"Layer {index + 1}")
        return ax.collections + ax.lines + ax.texts

    return FuncAnimation(fig, _update, frames=len(layers), interval=interval, blit=False)


def animate_routing(
    initial_placement: dict[int, int],
    routed_program: list[tuple],
    graph: nx.Graph | None = None,
    interval: int = 800,
) -> FuncAnimation:
    """Animate qubit positions as the router executes SWAPs and gates.

    Each frame shows:
    - the current logical-to-physical mapping as labels on the hardware graph,
    - the edge involved in the current SWAP or 2Q gate (highlighted in red).

    Frames are generated for the initial state, every SWAP, and every 2Q gate.
    Single-qubit gates produce no frame (they don't move qubits or require adjacency).
    """
    graph = graph or build_hardware_graph()

    placement = dict(initial_placement)
    phys_to_log: dict[int, int | None] = {phys: log for log, phys in placement.items()}

    # (placement_snapshot, highlight_edges, title)
    frame_data: list[tuple[dict, list, str]] = [
        (dict(placement), [], "Initial placement"),
    ]

    for op in routed_program:
        kind = op[0]
        if kind == "SWAP":
            _, left, right = op
            log_left = phys_to_log.get(left)
            log_right = phys_to_log.get(right)
            phys_to_log[left], phys_to_log[right] = log_right, log_left
            if log_left is not None:
                placement[log_left] = right
            if log_right is not None:
                placement[log_right] = left
            frame_data.append((dict(placement), [(left, right)], f"SWAP({left}, {right})"))
        elif kind == "2Q":
            _, left, right = op
            l_label = f"L{phys_to_log[left]}" if left in phys_to_log else str(left)
            r_label = f"L{phys_to_log[right]}" if right in phys_to_log else str(right)
            frame_data.append((dict(placement), [(left, right)], f"Gate: {l_label} ↔ {r_label}"))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    def _update(index: int) -> list:
        ax.clear()
        snap, edges, title = frame_data[index]
        draw_hardware(graph=graph, placement=snap, highlight_edges=edges, ax=ax, title=title)
        return ax.collections + ax.lines + ax.texts

    return FuncAnimation(fig, _update, frames=len(frame_data), interval=interval, blit=False)
