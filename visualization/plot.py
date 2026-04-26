"""Visualization utilities for the disaster response system."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from shared.utils import unique_edges

def plot_graph(
    nodes: Mapping[str, Mapping[str, object]],
    graph: Mapping[str, Iterable[str]],
    risk_map: Mapping[str, float],
    paths: Sequence[Sequence[str]],
    survivors: Sequence[str],
    show_active: bool = False,
) -> Figure:
    """Render the map, inferred risks, and final rescue path with conditional movement highlighting."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Render structural edges
    for node1, node2 in unique_edges(graph):
        x1, y1 = _get_position(nodes[node1])
        x2, y2 = _get_position(nodes[node2])
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=2, zorder=1)

    # Render structural nodes
    road_x, road_y = _collect_points(nodes, "road")
    building_x, building_y = _collect_points(nodes, "building")
    gate_x, gate_y = _collect_points(nodes, "gate")

    ax.scatter(road_x, road_y, c="gray", s=70, label="Roads", zorder=2)
    ax.scatter(building_x, building_y, c="green", s=170, marker="s", label="Buildings", zorder=3)
    ax.scatter(gate_x, gate_y, c="blue", s=180, marker="D", label="Gates", zorder=4)

    # Render Bayesian risk zones
    for node, probability in risk_map.items():
        x, y = _get_position(nodes[node])
        ax.scatter(x, y, c="red", s=320, alpha=max(0.15, probability), zorder=5)

    # Render the traversed path and conditional active movement indicators
    if paths and paths[0]:
        path = paths[0]

        # Render historical path trail
        if len(path) >= 2:
            path_x = [_get_position(nodes[node])[0] for node in path]
            path_y = [_get_position(nodes[node])[1] for node in path]
            ax.plot(path_x, path_y, linewidth=3, color="orange", alpha=0.6, zorder=6, label="Traversed Path")

            # Highlight the most recent step with a directional arrow only during active animation
            if show_active:
                start_x, start_y = _get_position(nodes[path[-2]])
                end_x, end_y = _get_position(nodes[path[-1]])
                ax.annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=3, shrinkA=8, shrinkB=8),
                    zorder=7
                )

        # Mark current active location only during active animation
        if show_active:
            curr_x, curr_y = _get_position(nodes[path[-1]])
            ax.scatter(
                curr_x, curr_y, 
                c="cyan", s=350, marker="P", 
                edgecolors="black", zorder=8, label="Active Unit"
            )

    # Render survivors
    survivor_x = [_get_position(nodes[surv])[0] for surv in survivors]
    survivor_y = [_get_position(nodes[surv])[1] for surv in survivors]
    if survivor_x and survivor_y:
        ax.scatter(
            survivor_x, survivor_y, 
            c="gold", s=220, marker="*", 
            edgecolors="black", linewidths=1, 
            zorder=9, label="Survivors"
        )

    # Provide a legend entry for high-risk areas
    ax.scatter([], [], c="red", s=150, alpha=0.5, label="High Risk Area")

    # Render node labels
    for label, data in nodes.items():
        x, y = _get_position(data)
        ax.text(
            x, y + 0.25, 
            label, 
            fontsize=10, 
            fontweight='bold',
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2', zorder=10)
        )

    ax.set_title("AI Disaster Response System")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(alpha=0.2)
    
    # Deduplicate and render legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    
    fig.tight_layout()
    return fig


def draw_graph(
    nodes: Mapping[str, Mapping[str, object]],
    graph: Mapping[str, Iterable[str]],
    risk_map: Mapping[str, float],
    paths: Sequence[Sequence[str]],
    survivors: Sequence[str],
    show_active: bool = False,
) -> Figure:
    """Backward-compatible wrapper for the main plotting function."""
    return plot_graph(nodes, graph, risk_map, paths, survivors, show_active)


def _collect_points(
    nodes: Mapping[str, Mapping[str, object]],
    kind: str,
) -> List[List[float]]:
    """Aggregate coordinates for a specific node type."""
    x_values = []
    y_values = []
    for data in nodes.values():
        if data["kind"] == kind:
            x, y = _get_position(data)
            x_values.append(x)
            y_values.append(y)
    return x_values, y_values


def _get_position(node_data: Mapping[str, object]) -> Tuple[float, float]:
    """Extract coordinates from node configuration."""
    if "coord" in node_data:
        return node_data["coord"]
    return node_data["pos"]