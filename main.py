from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.runtime as streamlit_runtime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from group1_map.bfs import bfs, generate_random_survivors
from group1_map.graph import create_custom_graph, create_graph, generate_random_graph
from group2_routing.astar import astar
from group3_bayesian.bayesian import infer_people, infer_risk
from group4_planning.assignment import assign_ambulance
from group4_planning.optimization import compute_priority, simulated_annealing
from shared.config import RANDOM_SEED
from visualization.plot import draw_graph

import time
def run_pipeline(
    nodes: Dict[str, Dict[str, object]] | None = None,
    graph: Dict[str, List[str]] | None = None,
    custom_survivors: List[str] | None = None,
) -> Dict[str, object]:
    """Run the optimized single-agent disaster response pipeline."""
    random.seed(RANDOM_SEED)

    if nodes is None or graph is None:
        nodes, graph = create_graph()

    normalized_nodes = _normalize_nodes(nodes)
    start_node = _find_start_node(normalized_nodes)

    # 1. Execute standard BFS to find all reachable nodes
    visited_nodes = bfs(graph, start_node)

    # 2. Handle survivor assignment separately based on the context
    if custom_survivors is not None:
        # Validate that custom survivors chosen via UI are actually reachable
        survivors = [s for s in custom_survivors if s in visited_nodes]
        if not survivors:
            raise ValueError("None of the assigned survivors are reachable from the gate.")
    else:
        # Generate random survivors for default or random maps
        survivors = generate_random_survivors(visited_nodes)

    # Dictionary initialization for routing
    distance_lookup: Dict[Tuple[str, str], float] = {}
    path_lookup: Dict[Tuple[str, str], List[str]] = {}

    relevant_nodes = [start_node] + survivors
    for source in relevant_nodes:
        distance_lookup[(source, source)] = 0.0
        path_lookup[(source, source)] = [source]

    for index, source in enumerate(relevant_nodes):
        for target in relevant_nodes[index + 1:]:
            path, cost, _ = astar(graph, normalized_nodes, source, target)
            distance_lookup[(source, target)] = cost
            distance_lookup[(target, source)] = cost
            path_lookup[(source, target)] = path
            path_lookup[(target, source)] = list(reversed(path))

    survivor_data = {}
    for survivor in survivors:
        camera_reading = random.choice(["High", "Low"])
        detection = random.choices(
            ["Detected", "NotDetected"],
            weights=[0.8, 0.2] if camera_reading == "High" else [0.35, 0.65],
            k=1,
        )[0]

        risk_distribution = infer_risk(camera_reading)
        people_distribution = infer_people(detection)
        vulnerability = round(random.uniform(0, 1), 2)
        priority = compute_priority(
            people_prob=people_distribution["Present"],
            risk_prob=risk_distribution["High"],
            vulnerability=vulnerability,
            distance=distance_lookup[(start_node, survivor)],
        )

        survivor_data[survivor] = {
            "camera_reading": camera_reading,
            "detection": detection,
            "risk_prob": risk_distribution["High"],
            "people_prob": people_distribution["Present"],
            "vulnerability": vulnerability,
            "priority": priority,
            "path": path_lookup[(start_node, survivor)],
            "cost": distance_lookup[(start_node, survivor)],
        }

    priorities = {node: data["priority"] for node, data in survivor_data.items()}
    optimized_order, annealing_score = simulated_annealing(
        targets=survivors,
        priorities=priorities,
        distance_lookup=distance_lookup,
        start_node=start_node,
    )

    assignment = assign_ambulance(
        ordered_targets=optimized_order,
        distance_lookup=distance_lookup,
        start_node=start_node,
    )

    visual_paths = []
    if not assignment["route"]:
        assignment["full_traversal"] = []
    else:
        full_path = [start_node]
        current = start_node
        
        for step in assignment["route"]:
            target = step["target"]
            segment = path_lookup[(current, target)]
            full_path.extend(segment[1:])
            current = target
            
        visual_paths.append(full_path)
        assignment["full_traversal"] = full_path

    risk_map = {node: data["risk_prob"] for node, data in survivor_data.items()}
    figure = draw_graph(
        nodes=normalized_nodes,
        graph=graph,
        risk_map=risk_map,
        paths=visual_paths,
        survivors=survivors,
    )

    return {
        "nodes": normalized_nodes,
        "graph": graph,
        "start_node": start_node,
        "visited_nodes": visited_nodes,
        "survivors": survivors,
        "survivor_data": survivor_data,
        "optimized_order": optimized_order,
        "annealing_score": annealing_score,
        "assignment": assignment,
        "figure": figure,
    }


def main() -> None:
    """Render the Streamlit UI and execute the simulation."""
    if not streamlit_runtime.exists():
        run_pipeline()
        return

    st.set_page_config(page_title="AI Disaster Response System", layout="wide")
    st.title("AI Disaster Response System")
    
    st.sidebar.header("Simulation Settings")
    st.sidebar.markdown("This simulation utilizes a single rescue agent to navigate the optimal path generated by the simulated annealing algorithm.")

    if "simulation_results" not in st.session_state:
        st.session_state["simulation_results"] = None
    if "simulation_error" not in st.session_state:
        st.session_state["simulation_error"] = ""

    preconfigured_tab, custom_tab = st.tabs(["Pre-configured Maps", "Custom Interactive Map"])

    with preconfigured_tab:
        default_col, random_col = st.columns(2)

        with default_col:
            if st.button("Run Default Map", use_container_width=True):
                try:
                    nodes, graph = create_graph()
                    st.session_state["simulation_results"] = run_pipeline(nodes, graph)
                    st.session_state["simulation_error"] = ""
                except Exception as error:
                    st.session_state["simulation_results"] = None
                    st.session_state["simulation_error"] = str(error)

        with random_col:
            if st.button("Run Random Map", use_container_width=True):
                try:
                    random.seed()
                    graph, nodes = generate_random_graph()
                    st.session_state["simulation_results"] = run_pipeline(nodes, graph)
                    st.session_state["simulation_error"] = ""
                except Exception as error:
                    st.session_state["simulation_results"] = None
                    st.session_state["simulation_error"] = str(error)

    with custom_tab:
        st.markdown("Use the interactive tables below to design a custom disaster scenario.")
        left_col, right_col = st.columns(2)

        with left_col:
            st.write("**Node Configuration**")
            edited_nodes = st.data_editor(
                _default_nodes_dataframe(),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="custom_nodes_editor",
                column_config={
                    "Node ID": st.column_config.TextColumn("Node ID", required=True),
                    "Type": st.column_config.SelectboxColumn(
                        "Type", 
                        help="Select the environmental category", 
                        options=["gate", "road", "building"], 
                        required=True
                    ),
                    "Has Survivor": st.column_config.CheckboxColumn(
                        "Has Survivor", 
                        help="Check if this node contains a survivor (Valid for buildings)",
                        default=False
                    )
                }
            )

        with right_col:
            st.write("**Edge Configuration**")
            
            # Dynamically extract current Node IDs to populate the Source/Target dropdowns
            available_nodes = edited_nodes["Node ID"].dropna().tolist()
            
            edited_edges = st.data_editor(
                _default_edges_dataframe(),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="custom_edges_editor",
                column_config={
                    "Source": st.column_config.SelectboxColumn(
                        "Source", 
                        options=available_nodes, 
                        required=True
                    ),
                    "Target": st.column_config.SelectboxColumn(
                        "Target", 
                        options=available_nodes, 
                        required=True
                    )
                }
            )

        if st.button("Run Custom Simulation", use_container_width=True):
            try:
                graph, nodes = create_custom_graph(edited_nodes, edited_edges)
                
                # Filter custom survivors (enforcing that they must be building types)
                custom_survivors = edited_nodes[
                    (edited_nodes["Has Survivor"] == True) & 
                    (edited_nodes["Type"] == "building")
                ]["Node ID"].tolist()
                
                if not custom_survivors:
                    st.session_state["simulation_error"] = "Please assign a survivor to at least one building node."
                    st.session_state["simulation_results"] = None
                else:
                    st.session_state["simulation_results"] = run_pipeline(
                        nodes, graph, custom_survivors=custom_survivors
                    )
                    st.session_state["simulation_error"] = ""
            except Exception as error:
                st.session_state["simulation_results"] = None
                st.session_state["simulation_error"] = str(error)

    if st.session_state["simulation_error"]:
        st.error(st.session_state["simulation_error"])

    if st.session_state["simulation_results"] is not None:
        results = st.session_state["simulation_results"]

        st.header("1. Initial Reconnaissance")
        st.markdown("*Mapping the disaster zone and identifying potential targets.*")
        st.write(f"**Disaster Origin (Gate):** `{results['start_node']}`")
        st.write(f"**Explored Areas:** `{', '.join(results['visited_nodes'])}`")
        st.write(f"**Detected Survivors At:** `{', '.join(results['survivors'])}`")

        st.header("2. Survivor Assessment & Environment Analysis")
        st.markdown("*Inferring risk and human presence using probabilistic sensor data.*")
        
        for survivor, data in results["survivor_data"].items():
            # Translating raw metrics to understandable labels
            risk_pct = data['risk_prob'] * 100
            risk_level = "High" if data['risk_prob'] >= 0.5 else "Low"
            
            people_pct = data['people_prob'] * 100
            people_level = "Very Likely" if data['people_prob'] >= 0.7 else ("Moderate" if data['people_prob'] >= 0.4 else "Unlikely")
            
            vuln_pct = data['vulnerability'] * 100
            vuln_level = "Critical" if data['vulnerability'] >= 0.7 else ("Moderate" if data['vulnerability'] >= 0.4 else "Low")
            
            priority_score = data['priority']
            priority_level = "Urgent" if priority_score >= 0.4 else ("High" if priority_score >= 0.3 else "Normal")

            with st.expander(f"Assessment Profile for Target: {survivor}"):
                st.markdown(f"- **Environmental Risk:** {risk_level} ({risk_pct:.1f}%)")
                st.markdown(f"- **Human Presence:** {people_level} ({people_pct:.1f}%)")
                st.markdown(f"- **Vulnerability Severity:** {vuln_level} ({vuln_pct:.1f}%)")
                st.markdown(f"- **Calculated Priority:** **{priority_level}** (Raw Score: {priority_score:.2f})")

        st.header("3. Navigation & Path Estimates")
        st.markdown("*Calculating optimal approach vectors to each isolated target.*")
        for survivor, data in results["survivor_data"].items():
            path_str = " ➔ ".join(data['path'])
            st.write(f"- **To {survivor}:** {path_str} `(Estimated Cost: {data['cost']:.2f})`")

        st.header("4. Strategic Rescue Execution")
        st.markdown("*Determining the most ethical and efficient global sequence.*")
        
        agent = results["assignment"]
        
        if not agent["route"]:
            st.warning("No survivors found or paths are completely obstructed.")
        else:
            route_summary = " ➔ ".join([stop["target"] for stop in agent["route"]])
            
            if "full_traversal" in agent and agent["full_traversal"]:
                detailed_path = " ➔ ".join(agent["full_traversal"])
            else:
                detailed_path = "No continuous path assigned"
                
            st.info(f"**Optimized Rescue Sequence:** {route_summary}")
            st.success(f"**Complete Traversal Route:** {detailed_path}")
            st.write(f"**Total Mission Distance:** `{agent['distance']:.2f}` units")

        st.header("Visualization")
        st.subheader("5. Mission Visualization")
        
        # Create an empty container to hold the map progression
        map_placeholder = st.empty()
        
        # Render the initial complete map
        map_placeholder.pyplot(results["figure"], clear_figure=True)

        agent = results["assignment"]
        # Display the animation trigger only if a valid path exists
        if agent["route"] and "full_traversal" in agent and agent["full_traversal"]:
            st.markdown("---")
            if st.button("▶️ Show Rescue Path Progression", use_container_width=True):
                full_path = agent["full_traversal"]
                
                # Iterate through the path to create a step-by-step drawing effect
                for i in range(2, len(full_path) + 1):
                    current_step = full_path[:i]
                    is_last_step = (i == len(full_path))
                    
                    step_fig = draw_graph(
                        nodes=results["nodes"],
                        graph=results["graph"],
                        risk_map={node: data["risk_prob"] for node, data in results["survivor_data"].items()},
                        paths=[current_step],
                        survivors=results["survivors"],
                        show_active=not is_last_step  # Hide marker on the final frame
                    )
                    
                    # Replace the previous map with the updated progression map
                    map_placeholder.pyplot(step_fig, clear_figure=True)
                    time.sleep(0.75)  

def _normalize_nodes(
    nodes: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Ensure coordinates are formatted consistently."""
    normalized_nodes: Dict[str, Dict[str, object]] = {}
    for node_id, data in nodes.items():
        if "coord" in data:
            position = data["coord"]
        else:
            position = data["pos"]
        normalized_nodes[node_id] = {"coord": position, "kind": data["kind"]}
    return normalized_nodes


def _find_start_node(nodes: Dict[str, Dict[str, object]]) -> str:
    """Identify the gate node to act as the simulation origin."""
    for node_id, data in nodes.items():
        if data["kind"] == "gate":
            return node_id
    raise ValueError("A gate node is required to start the simulation.")


def _default_nodes_dataframe() -> pd.DataFrame:
    """Provide default values for the custom node editor."""
    return pd.DataFrame(
        [
            {"Node ID": "G1", "X": 0.0, "Y": 0.0, "Type": "gate", "Has Survivor": False},
            {"Node ID": "I1", "X": 2.0, "Y": 0.0, "Type": "road", "Has Survivor": False},
            {"Node ID": "I2", "X": 4.0, "Y": 0.0, "Type": "road", "Has Survivor": False},
            {"Node ID": "B1", "X": 2.0, "Y": 2.0, "Type": "building", "Has Survivor": True},
            {"Node ID": "B2", "X": 4.0, "Y": 2.0, "Type": "building", "Has Survivor": True},
        ]
    )

def _default_edges_dataframe() -> pd.DataFrame:
    """Provide default values for the custom edge editor."""
    return pd.DataFrame(
        [
            {"Source": "G1", "Target": "I1"},
            {"Source": "I1", "Target": "I2"},
            {"Source": "I1", "Target": "B1"},
            {"Source": "I2", "Target": "B2"},
        ]
    )


if __name__ == "__main__":
    main()