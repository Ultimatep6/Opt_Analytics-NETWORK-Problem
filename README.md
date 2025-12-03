# Network Linear Programming Problem: Multi-Layer Goods Transport

## Problem Overview
This project models a complex network linear programming (LP) problem for transporting goods across three layers:

1. **Production Sites (Layer 1):** Nodes with supply limits.
2. **Factories (Layer 2):** Intermediate nodes with both supply and demand constraints.
3. **End Goals (Layer 3):** Nodes with demand requirements.

Goods are transported from production sites to factories, then between factories, and finally to end goals.

## Key Features
- **One-way and Two-way Paths:** Connections between nodes can be unidirectional or bidirectional.
- **Time Interval Activation:** Some connections are only available during specific time intervals.
- **Maximum Throughput:** Each connection has a maximum capacity (throughput) that cannot be exceeded.

## Mathematical Formulation
Let $x_{ijk}^t$ be the amount of goods transported from node $i$ to node $j$ in layer $k$ during time interval $t$.
Let $C_{ijk}^t$ be the maximum throughput for connection $(i, j, k)$ at time $t$.

**Constraints:**
- Supply at production sites
- Demand and supply at factories
- Demand at end goals
- Arc capacity and directionality
- Time-activated arcs: $x_{ijk}^t = 0$ if arc $(i,j)$ is inactive at $t$
- Maximum throughput: $x_{ijk}^t \leq C_{ijk}^t$

**Objective:**
- Minimize total transportation cost or maximize throughput, depending on the scenario.

## Implementation
This structure allows for flexible modeling and can be implemented using Python with libraries such as PuLP or Pyomo. The model can be extended to include additional constraints or objectives as needed.

---

For further details or implementation help, see the `scripts/` directory or contact the project maintainer.