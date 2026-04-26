from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

from group2_routing.heuristic import heuristic
from shared.utils import distance, path_cost


def astar(
    graph: Dict[str, List[str]],
    nodes: Dict[str, Dict[str, object]],
    start: str,
    goal: str,
) -> Tuple[List[str], float, List[str]]:
    """Compute a shortest path with A* using Euclidean distance."""
    if start not in graph or goal not in graph:
        raise ValueError(f"Unknown A* nodes: start={start}, goal={goal}")

    frontier: List[Tuple[float, float, str]] = [(heuristic(nodes[start], nodes[goal]), 0.0, start)]
    came_from = {start: None}
    g_score = {name: float("inf") for name in nodes}
    g_score[start] = 0.0
    explored: List[str] = []
    visited = set()

    while frontier:
        _, current_g, current = heapq.heappop(frontier)
        if current in visited:
            continue

        visited.add(current)
        explored.append(current)
        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return path, current_g, explored

        for neighbor in graph[current]:
            step_cost = distance(nodes[current]["coord"], nodes[neighbor]["coord"])
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(nodes[neighbor], nodes[goal])
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

    return [], float("inf"), explored


def _reconstruct_path(came_from: Dict[str, str | None], goal: str) -> List[str]:
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
