
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pandas as pd


def create_graph() -> Tuple[Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    """Create a small road network with intersections, buildings, and gates."""
    nodes = {
        "G1": {"coord": (0, 5), "kind": "gate"},
        "G2": {"coord": (10, 5), "kind": "gate"},
        "I1": {"coord": (2, 8), "kind": "road"},
        "I2": {"coord": (5, 8), "kind": "road"},
        "I3": {"coord": (8, 8), "kind": "road"},
        "I4": {"coord": (2, 5), "kind": "road"},
        "I5": {"coord": (5, 5), "kind": "road"},
        "I6": {"coord": (8, 5), "kind": "road"},
        "I7": {"coord": (2, 2), "kind": "road"},
        "I8": {"coord": (5, 2), "kind": "road"},
        "I9": {"coord": (8, 2), "kind": "road"},
        "B1": {"coord": (1, 9), "kind": "building"},
        "B2": {"coord": (5, 9.5), "kind": "building"},
        "B3": {"coord": (9, 7.5), "kind": "building"},
        "B4": {"coord": (1, 4), "kind": "building"},
        "B5": {"coord": (5.5, 0.5), "kind": "building"},
        "B6": {"coord": (9, 3), "kind": "building"},
    }

    graph = {
        "G1": ["I4", "I7"],
        "G2": ["I6", "I9"],
        "I1": ["I2", "I4", "B1"],
        "I2": ["I1", "I3", "I5", "B2"],
        "I3": ["I2", "I6", "B3"],
        "I4": ["G1", "I1", "I5", "I7", "B4"],
        "I5": ["I2", "I4", "I6", "I8"],
        "I6": ["G2", "I3", "I5", "I9", "B3"],
        "I7": ["G1", "I4", "I8"],
        "I8": ["I5", "I7", "I9", "B5"],
        "I9": ["G2", "I6", "I8", "B6"],
        "B1": ["I1"],
        "B2": ["I2"],
        "B3": ["I3", "I6"],
        "B4": ["I4"],
        "B5": ["I8"],
        "B6": ["I9"],
    }

    _validate_graph(nodes, graph)
    return nodes, graph


def create_custom_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, object]]]:
    """Create a graph from user-supplied dataframes without failing on bad rows."""
    nodes: Dict[str, Dict[str, object]] = {}
    graph: Dict[str, List[str]] = {}

    safe_nodes_df = _ensure_dataframe(nodes_df)
    safe_edges_df = _ensure_dataframe(edges_df)

    node_id_col = _find_column(safe_nodes_df, ["node id", "node_id", "id"])
    x_col = _find_column(safe_nodes_df, ["x"])
    y_col = _find_column(safe_nodes_df, ["y"])
    type_col = _find_column(safe_nodes_df, ["type", "kind"])

    if all(column is not None for column in [node_id_col, x_col, y_col, type_col]):
        for _, row in safe_nodes_df.iterrows():
            if _is_empty_row(row):
                continue

            node_id = _clean_string(row.get(node_id_col))
            node_type = _clean_string(row.get(type_col)).lower()

            if not node_id or not node_type:
                continue

            try:
                x_value = float(str(row.get(x_col)).strip())
                y_value = float(str(row.get(y_col)).strip())
            except (TypeError, ValueError):
                continue

            nodes[node_id] = {"pos": (x_value, y_value), "kind": node_type}
            graph.setdefault(node_id, [])

    source_col = _find_column(safe_edges_df, ["source"])
    target_col = _find_column(safe_edges_df, ["target"])

    if all(column is not None for column in [source_col, target_col]):
        for _, row in safe_edges_df.iterrows():
            if _is_empty_row(row):
                continue

            source = _clean_string(row.get(source_col))
            target = _clean_string(row.get(target_col))

            if not source or not target or source == target:
                continue
            if source not in nodes or target not in nodes:
                continue

            graph.setdefault(source, [])
            graph.setdefault(target, [])
            if target not in graph[source]:
                graph[source].append(target)
            if source not in graph[target]:
                graph[target].append(source)

    for node_id in nodes:
        graph.setdefault(node_id, [])

    _validate_graph(nodes, graph)
    return graph, nodes


def generate_random_graph(
    num_intersections: int = 8,
    num_buildings: int = 5,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, object]]]:
    """Generate a realistic scattered graph with multiple gates and alternate routes."""
    num_intersections = max(3, int(num_intersections))
    num_buildings = max(1, int(num_buildings))

    nodes: Dict[str, Dict[str, object]] = {}
    graph: Dict[str, List[str]] = {}

    # 1. Place Gates (Randomly 1 or 2 gates at opposite sides)
    num_gates = random.choice([1, 2])
    gates = []
    for i in range(1, num_gates + 1):
        gate_id = f"G{i}"
        x_coord = 0.0 if i == 1 else random.uniform(9.0, 12.0)
        y_coord = random.uniform(1.0, 9.0)
        nodes[gate_id] = {"coord": (x_coord, y_coord), "kind": "gate"}
        graph[gate_id] = []
        gates.append(gate_id)

    # 2. Place Intersections scattered across the 2D plane
    intersections = []
    for index in range(1, num_intersections + 1):
        node_id = f"I{index}"
        # Spread randomly instead of a single straight line
        nodes[node_id] = {
            "coord": (random.uniform(2.0, 8.0), random.uniform(1.0, 9.0)), 
            "kind": "road"
        }
        graph[node_id] = []
        intersections.append(node_id)

    # 3. Guarantee Connectivity (Build a Random Tree)
    # Connect the first intersection to the first Gate
    _connect_nodes(graph, gates[0], intersections[0])
    connected_intersections = [intersections[0]]

    # Connect every new intersection to a randomly chosen ALREADY connected intersection
    for i in range(1, len(intersections)):
        curr_node = intersections[i]
        target_node = random.choice(connected_intersections)
        _connect_nodes(graph, curr_node, target_node)
        connected_intersections.append(curr_node)

    # 4. Attach additional gates (if any) to random intersections
    if len(gates) > 1:
        for g in gates[1:]:
            _connect_nodes(graph, g, random.choice(intersections))

    # 5. Add Cross-Connections (Loops) for A* alternate routing
    # Add roughly half as many extra edges as there are intersections
    extra_edges = num_intersections // 2
    for _ in range(extra_edges):
        n1, n2 = random.sample(intersections, 2)
        _connect_nodes(graph, n1, n2)

    # 6. Place Buildings around random intersections
    for index in range(1, num_buildings + 1):
        building_id = f"B{index}"
        anchor = random.choice(intersections)
        anchor_x, anchor_y = nodes[anchor]["coord"]
        
        # Place building near its connected intersection
        offset_x = random.uniform(-1.5, 1.5)
        offset_y = random.uniform(-1.5, 1.5)
        
        nodes[building_id] = {
            "coord": (anchor_x + offset_x, anchor_y + offset_y),
            "kind": "building",
        }
        graph[building_id] = []
        _connect_nodes(graph, anchor, building_id)

    _validate_graph(nodes, graph)
    return graph, nodes


def _connect_nodes(graph: Dict[str, List[str]], source: str, target: str) -> None:
    graph.setdefault(source, [])
    graph.setdefault(target, [])
    if target not in graph[source]:
        graph[source].append(target)
    if source not in graph[target]:
        graph[target].append(source)


def _ensure_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if isinstance(dataframe, pd.DataFrame):
        return dataframe.copy()
    return pd.DataFrame()


def _find_column(dataframe: pd.DataFrame, candidates: List[str]) -> str | None:
    normalized_map = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }
    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]
    return None


def _clean_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_empty_row(row: pd.Series) -> bool:
    for value in row.tolist():
        if _clean_string(value):
            return False
    return True


def _validate_graph(
    nodes: Dict[str, Dict[str, object]],
    graph: Dict[str, List[str]],
) -> None:
    """Ensure every node appears in the adjacency list and edges are consistent."""
    node_names = set(nodes)
    adjacency_names = set(graph)

    if node_names != adjacency_names:
        missing_in_graph = sorted(node_names - adjacency_names)
        missing_in_nodes = sorted(adjacency_names - node_names)
        raise ValueError(
            f"Graph mismatch. Missing in graph: {missing_in_graph}; "
            f"missing in nodes: {missing_in_nodes}"
        )

    for node, node_data in nodes.items():
        if "kind" not in node_data:
            raise ValueError(f"Node '{node}' is missing a kind.")
        if "coord" not in node_data and "pos" not in node_data:
            raise ValueError(f"Node '{node}' is missing coordinates.")

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in nodes:
                raise ValueError(f"Unknown neighbor '{neighbor}' referenced by '{node}'.")
            if node not in graph[neighbor]:
                raise ValueError(f"Edge inconsistency between '{node}' and '{neighbor}'.")
