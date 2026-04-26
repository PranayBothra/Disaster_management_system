from __future__ import annotations

from typing import Mapping

from shared.utils import distance


def heuristic(node1: Mapping[str, object], node2: Mapping[str, object]) -> float:
    """Euclidean heuristic between two node records."""
    return distance(node1["coord"], node2["coord"])
