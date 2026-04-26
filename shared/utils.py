"""Shared helper utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Coordinate = Tuple[float, float]


def distance(point1: Coordinate, point2: Coordinate) -> float:
    """Return Euclidean distance between two 2D points."""
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return (dx * dx + dy * dy) ** 0.5


def path_cost(nodes: Mapping[str, Mapping[str, object]], path: Sequence[str]) -> float:
    """Compute the travel cost of a node path."""
    if len(path) < 2:
        return 0.0

    total = 0.0
    for current, nxt in zip(path, path[1:]):
        total += distance(nodes[current]["coord"], nodes[nxt]["coord"])
    return total


def normalize_probabilities(distribution: Mapping[str, float]) -> Dict[str, float]:
    """Normalize a probability-like mapping into a proper distribution."""
    total = sum(distribution.values())
    if total == 0:
        return {key: 0.0 for key in distribution}
    return {key: value / total for key, value in distribution.items()}


def unique_edges(graph: Mapping[str, Iterable[str]]) -> List[Tuple[str, str]]:
    """Return undirected graph edges without duplicates."""
    seen = set()
    edges: List[Tuple[str, str]] = []
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edge = tuple(sorted((node, neighbor)))
            if edge not in seen:
                seen.add(edge)
                edges.append((node, neighbor))
    return edges
