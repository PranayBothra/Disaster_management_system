from __future__ import annotations
from typing import Dict, Sequence, Tuple

def assign_ambulance(
    ordered_targets: Sequence[str],
    distance_lookup: Dict[Tuple[str, str], float],
    start_node: str = "G1",
) -> Dict[str, object]:


    agent = {
        "agent": "Rescue-1",
        "route": [],
        "distance": 0.0
    }

    current_node = start_node

    for target in ordered_targets:
        travel_cost = distance_lookup[(current_node, target)]
        agent["route"].append({"target": target, "travel_cost": round(travel_cost, 2)})
        agent["distance"] += travel_cost
        current_node = target

    agent["distance"] = round(agent["distance"], 2)

    return agent
