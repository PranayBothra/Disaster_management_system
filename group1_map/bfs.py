from __future__ import annotations

import random
from collections import deque
from typing import Dict, List


def bfs(graph: Dict[str, List[str]], start: str) -> List[str]:
    """Traverse the graph and return all reachable nodes."""
    if start not in graph:
        raise ValueError(f"Unknown BFS start node: {start}")

    visited = []
    seen = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        visited.append(current)
        
        # Using .get() prevents KeyError if a node has no outgoing edges
        for neighbor in graph.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    return visited


def generate_random_survivors(visited_nodes: List[str]) -> List[str]:
    """Randomly select 2-4 building nodes from the reachable nodes."""
    building_nodes = [node for node in visited_nodes if node.startswith("B")]
    
    if len(building_nodes) < 2:
        raise ValueError("At least two reachable building nodes are required to generate random survivors.")

    survivor_count = random.randint(2, min(4, len(building_nodes)))
    survivors = random.sample(building_nodes, k=survivor_count)
    
    return survivors
