# 2D Warehouse Optimization and Multi-Agent Simulation

## Overview

This project provides a robust, scalable 2D warehouse simulation designed for multi-agent reinforcement learning (MARL) research and logistical optimization. Built upon Pygame and Gymnasium, the environment models a realistic warehouse floor where autonomous robots navigate constrained aisles, retrieve payload boxes from designated shelves, and transport them to drop-off platforms. 

The architecture is explicitly designed to train and evaluate collision avoidance, dynamic pathfinding, and cooperative efficiency in a shared workspace. It serves as a functional prototype for real-world automated guided vehicle (AGV) dispatch systems used in modern logistics centers.

## Prerequisites & Installation

To deploy the simulation, ensure the following dependencies are installed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 pettingzoo pygame numpy gymnasium
```

The core requirements are:
- **Python 3.8+**
- **Pygame**: For 2D rendering and grid visualization.
- **PyTorch**: Used for the deep learning neural networks governing agent policies.
- **Gymnasium**: Provides the reinforcement learning environment API.
- **NumPy**: Handles matrix operations and continuous state representations.

## Core Mechanics and Physics

The project cleanly separates the static physical simulation from the dynamic agent logic and environment tracking.

### 1. Static World Generation (`world.py` & `constants.py`)
The warehouse topography is initialized statically as a discrete grid (22x15).
- **Shelves:** Arranged in structured vertical and horizontal aisles to mimic standard warehouse storage racks. Shelves are the spawn points for payloads (boxes).
- **Drop-off Platforms:** Designated green zones at the bottom of the map where payloads must be delivered to complete a task cycle.
- **Charge Stations:** Starting positions at the top of the map serving as idle zones for agents awaiting dispatch commands.

### 2. Robot Capabilities (`robot.py`)
The `Robot` class handles the physical manifestation of the AGVs:
- **Kinematics:** Movement is restricted to orthogonal grid steps (Up, Down, Left, Right). 
- **Payload Operations:** The `pickup_box` and `drop_box` actions require precise geometric alignment. An agent must be exactly adjacent to the correct shelf or platform and facing the correct direction to successfully trigger a load transfer.
- **Visual State:** The robot dynamically updates its sprite to provide visual feedback (e.g., displaying a box when loaded, rotating to face its movement vector).

## Environment Controller (`unified_env.py`)

`unified_env.py` acts as the core orchestration layer. Inheriting from `gymnasium.Env`, it maintains the global state, processes actions, computes physics, and issues step-wise rewards.

### Task Phases
Each robot cycles through three distinct operational phases:
1. **Homing:** The robot is idle and returns to its designated charge station.
2. **Fetching:** A payload is spawned. The robot is dispatched to navigate to the specific shelf containing the box.
3. **Delivering:** The robot has successfully picked up the payload and must now navigate to its designated drop-off platform.

### Collision Detection & Distance Mapping
The environment utilizes Breadth-First Search (BFS) distance maps dynamically calculated around static obstacles. These maps ensure physically valid simulations by preventing agents from phasing through walls and calculating the shortest possible valid paths. The environment strictly enforces agent-to-agent and agent-to-wall collision detection.

## Multi-Agent Percepts and Policies

The simulation evaluates two distinct robotic agents operating asymmetrically, allowing for the study of an independent tabular agent against an environmentally-aware deep learning agent.

### Robot 1 (Independent Tabular Q-Learning)
Robot 1 operates using a decentralized, pre-trained Q-table policy (`q_learning_agent.py`).
- **Percepts (State Representation):** Robot 1 observes a highly localized 8-feature array, which is mapped to a concise 6-feature discrete tuple:
  1. Current X coordinate
  2. Current Y coordinate
  3. Load status (Boolean: 1 if carrying a box, 0 if empty)
  4. Current directional facing (0-3)
  5. Target shelf X coordinate
  6. Target shelf Y coordinate
- **Behavior:** It is functionally "blind" to the other robot, navigating based purely on its immediate target and falling back to algorithmic BFS routing when encountering unseen states.

### Robot 2 (Spatially-Aware Deep Learning Agent)
Robot 2 acts as the primary learning agent. It requires a much richer understanding of its surroundings to navigate safely and actively avoid Robot 1 while optimizing its delivery times.
- **Percepts (25-Feature Observation Space):** Robot 2 processes a continuous vector encompassing:
  - **Self-State (7 features):** Normalized coordinates, load status, and relative vector distances to both its current target and its drop-off platform.
  - **Surroundings (8 features):** A 360-degree adjacent-cell scan (up, down, left, right, and diagonals) detecting immediate obstacles, walls, boundaries, and the physical presence of Robot 1.
  - **Context (4 features):** The agent's last movement vector, and boolean flags indicating if valid pickup (action 4) or delivery actions are currently possible.
  - **Robot 1 Tracking (6 features):** Relative distance to Robot 1, Robot 1's load status, Robot 1's current phase, and predictive vectors mapping exactly where Robot 1 is expected to step next.

## Training Architecture (`train_unified_agent2.py`)

Robot 2 is trained using a **Deep Q-Network (DQN)**, allowing it to generalize over the continuous 25-feature state space.

### The Deep Q-Network
The model is a PyTorch Feed-Forward Neural Network consisting of:
- An input layer accepting the 25-dimensional state vector.
- Two hidden layers with 128 neurons each, utilizing ReLU activations.
- An output layer producing Q-values for the 6 discrete actions (Up, Down, Left, Right, Interact, Idle).

### The Training Loop
The training process leverages several standard Reinforcement Learning techniques for stability and convergence:
- **Epsilon-Greedy Exploration:** The agent begins exploring randomly (Epsilon = 0.7). Over time, this probability decays (`0.999` multiplier per step) down to a minimum of `0.05`, shifting the agent from exploration to exploitation of the learned policy.
- **Action Masking:** To prevent the model from wasting time learning physically impossible moves, invalid actions (e.g., walking into a wall or picking up air) are masked out (set to negative infinity) before the agent selects an action. Furthermore, during exploration, the agent occasionally forcefully utilizes BFS recommendations to bootstrap its learning.
- **Experience Replay Buffer:** A deque buffer (capacity: 50,000) stores transitions `(state, action, reward, next_state, done_flag, action_masks)`. At each step, a random mini-batch of 128 experiences is sampled to train the network, breaking the correlation between consecutive steps.
- **Target Network Syncing:** Two identical networks are maintained: a Policy Net and a Target Net. The Target Net computes the expected future rewards and is frozen during standard steps. It synchronizes with the Policy Net's weights every 10 episodes to prevent moving-target instability.
- **Optimization:** The network is optimized using Mean Squared Error (MSELoss) between the current predicted Q-values and the Target Q-values, updated via the Adam Optimizer with a learning rate of `1e-4` and gradient clipping (max norm `10.0`). Models are automatically checkpointed based on the highest achieved rewards.
