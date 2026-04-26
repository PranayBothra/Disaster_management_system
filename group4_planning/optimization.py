from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from shared.config import alpha, w1, w2, w3


def compute_priority(
    people_prob: float,
    risk_prob: float,
    vulnerability: float,
    distance: float,
) -> float:
    #Compute the exact user-specified ethical priority score.
    return (
        w1 * people_prob +
        w2 * risk_prob +
        w3 * vulnerability
    ) / (1 + alpha * distance)


def simulated_annealing(
    targets: Sequence[str],
    priorities: Dict[str, float],
    distance_lookup: Dict[Tuple[str, str], float],
    start_node: str,
    iterations: int = 800,
    initial_temp: float = 8.0,
    cooling_rate: float = 0.995,
) -> Tuple[List[str], float]:
    #Optimize the rescue order while allowing uphill moves to escape local minima.
    if not targets:
        return [], 0.0

    rng = random.Random(21)
    current_order = list(targets)
    rng.shuffle(current_order)
    current_score = _route_score(current_order, priorities, distance_lookup, start_node)

    best_order = current_order[:]
    best_score = current_score
    temperature = initial_temp

    for _ in range(iterations):
        candidate_order = current_order[:]
        if len(candidate_order) > 1:
            i, j = sorted(rng.sample(range(len(candidate_order)), 2))
            candidate_order[i], candidate_order[j] = candidate_order[j], candidate_order[i]

        candidate_score = _route_score(candidate_order, priorities, distance_lookup, start_node)
        delta = candidate_score - current_score

        if delta > 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
            current_order = candidate_order
            current_score = candidate_score

        if current_score > best_score:
            best_order = current_order[:]
            best_score = current_score

        temperature = max(temperature * cooling_rate, 1e-6)

    return best_order, best_score


def _route_score(
    order: Sequence[str],
    priorities: Dict[str, float],
    distance_lookup: Dict[Tuple[str, str], float],
    start_node: str,
) -> float:
    cumulative_distance = 0.0
    score = 0.0
    current = start_node

    for node in order:
        cumulative_distance += distance_lookup[(current, node)]
        score += priorities[node] / (1 + cumulative_distance)
        current = node

    return score
