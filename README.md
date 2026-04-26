# AI Disaster Response System

This project simulates a modular disaster response pipeline in Python using:

- BFS for map exploration and survivor discovery
- A* for shortest-path routing
- Bayesian inference for uncertain camera and people detection signals
- Simulated annealing and ambulance assignment for ethical rescue planning

## Structure

```text
disaster_ai/
├── main.py
├── shared/
│   ├── config.py
│   └── utils.py
├── group1_map/
│   ├── graph.py
│   └── bfs.py
├── group2_routing/
│   ├── astar.py
│   └── heuristic.py
├── group3_bayesian/
│   ├── bayesian.py
│   └── cpt.py
├── group4_planning/
│   ├── optimization.py
│   └── assignment.py
├── visualization/
│   └── plot.py
└── README.md
```

## How To Run

1. Open a terminal in the `disaster_ai` folder.
2. Run:

```bash
python main.py
```

The program will:

1. Build the disaster map
2. Run BFS to discover survivor locations
3. Use A* to compute path costs
4. Infer risk and people presence probabilities
5. Compute ethical rescue priorities
6. Optimize rescue order for 2 rescue agents
7. Visualize the graph and selected rescue paths

## Notes

- The priority function matches the requested formula exactly.
- The project uses a fixed random seed for reproducible single-run results.
- `matplotlib` is required for visualization.
