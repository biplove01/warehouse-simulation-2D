# 2D Warehouse Simulation - Complete Project Documentation

This document provides a comprehensive, deep-dive explanation of the Two-Agent Warehouse Simulation project. By reading this guide, any new developer or reader will gain a complete understanding of the system architecture, agent logic, reinforcement learning approaches, and how the simulation works as a whole.

## 1. Project Overview

This project simulates a warehouse environment where two robotic agents (Robot 1 and Robot 2) navigate a grid to fetch boxes from shelves and deliver them to designated drop-off platforms. The environment is built using Python, with **Gymnasium** acting as the reinforcement learning (RL) framework and **Pygame** used for 2D rendering and interactive testing.

What makes this project unique is the **asymmetric intelligence** of the two robots:
- **Robot 1 (R1)** operates on a pre-trained **Q-Table** (classic Q-learning) or a purely deterministic Breadth-First Search (BFS) algorithm.
- **Robot 2 (R2)** operates on a **Deep Q-Network (DQN)** which is actively trained to optimize its pathing, learn to cooperate (or avoid colliding) with Robot 1, and efficiently complete tasks.

## 2. Environment Architecture (`unified_env.py`)

The core of the simulation is the `TwoAgentWarehouseEnv` class, which inherits from `gym.Env`.

### Grid System and Topography
The world is constructed on a 2D grid defined in `world.py` and `constants.py`.
- **Entities**: The grid contains Shelves (where boxes spawn), Drop-off Platforms (where boxes are delivered), and Charge Stations (visual elements).
- **Phases**: Each robot cycles through three operational phases:
  1. `HOMING`: The robot is waiting for a task at its base position.
  2. `FETCHING`: The robot is navigating to a specific shelf to pick up a box.
  3. `DELIVERING`: The robot has a box and is navigating to the drop-off platform.

### Target Queue and Dispatching
Boxes do not spawn indefinitely everywhere. Instead, there is a global `queue` of tasks. When a box spawns on a shelf, its grid coordinates are added to this queue. The `_dispatch` function checks if a robot is in the `HOMING` phase and assigns it the next target from the queue, immediately transitioning it to `FETCHING`.

## 3. Robot Capabilities & Mechanics (`robot.py`)

The robots can perform 6 distinct actions:
- `0`: Move Up
- `1`: Move Down
- `2`: Move Left
- `3`: Move Right
- `4`: Interact (Pick up a box if adjacent to a loaded shelf, or Drop off a box if adjacent to a platform)
- `5`: Wait (Do nothing / Idle)

Collision detection is strictly enforced. Robots cannot walk through shelves or drop-off platforms, nor can they occupy the same space as the other robot. Collisions heavily penalize the learning agent.

## 4. Robot 1: The Q-Table / BFS Agent

Robot 1 represents the "baseline" or "legacy" worker in this warehouse.

- **Policy Setup**: R1 uses `QTablePolicy` defined in `unified_env.py` and `q_learning_agent.py`.
- **State Representation**: R1 maps its state down to a 6-tuple: `(rx, ry, loaded, direction, bx, by)` (its x/y position, whether it carries a box, its facing direction, and its target's x/y position).
- **Execution**: When picking an action, R1 checks `warehouse_data.pkl` (the saved Q-Table). If a state is known, it picks the best action. If a state is completely unseen, or if the Q-table file doesn't exist, R1 falls back to an algorithmic **Breadth-First Search (BFS)** to guarantee it always finds the shortest path to its current target.
- **Role**: R1 acts independently. It does not actively track Robot 2. It simply does its job, serving as a dynamic, moving obstacle/coworker that Robot 2 must learn to navigate around.

## 5. Robot 2: The Deep Q-Network Agent

Robot 2 is the primary focus of the reinforcement learning loop. It uses a Deep Neural Network (DQN) to map complex environmental states to optimal actions.

### Observation Space
R2 receives a much richer, continuous observation vector (size 25) that includes:
- R2's normalized (x, y) coordinates and load status.
- Normalized directional vectors pointing to its current target and drop-off location.
- **Local Vision**: A surrounding 8-tile radar that flags if an adjacent tile is a wall, out of bounds, or occupied by R1.
- **R1 Awareness**: Vectors pointing toward R1, R1's load status, and predicting R1's *next* intended position.

### Action Masking
To prevent the DQN from wasting time learning obvious physical impossibilities (like walking through a wall), the environment provides an **Action Mask**. The mask disables invalid moves, ensuring the neural network only selects from physically possible actions.

### Reward Shaping
Training R2 requires careful reward shaping. Rewards and penalties include:
- `+10` for picking up a box, `+20` for successfully delivering it.
- `+2.0` for moving closer to its current target, and `-4.0` penalty for moving away.
- `-50.0` for colliding with R1.
- `-35.0` for hitting a wall, and `-25.0` for an invalid interact attempt (e.g., trying to pick up thin air).
- Small idle penalties to discourage standing still.

## 6. Training Pipeline (`train_unified_agent2.py`)

The training script initializes the `TwoAgentWarehouseEnv` and trains Robot 2 using standard Experience Replay and a Target Network approach.

1. **Initialization**: A `policy_net` (the brain being trained) and a `target_net` (a stable reference for Q-value updates) are created. Both are 3-layer Feedforward Neural Networks (MLPs) with ReLU activations.
2. **Action Selection (Epsilon-Greedy)**: During training, R2 uses an epsilon-greedy strategy. It mostly relies on its neural network, but randomly explores. When exploring, it frequently defers to a "Recommended BFS Action" provided by the environment to jump-start the learning process and prevent getting permanently stuck early on.
3. **Experience Replay**: Every step (state, action, reward, next_state) is saved to a buffer. After gathering enough data, random mini-batches are pulled to perform Gradient Descent and update the neural network weights.
4. **Checkpointing**: Every few episodes, the model weights are saved to the `checkpoints/` directory (`latest_model.pth`, `best_model.pth`, `model_episode.pth`).

## 7. Interactive Testing and Evaluation (`test_two_agents.py` & `play.py`)

Once the DQN is trained, you can observe the agents in action.

### Automated Agent Evaluation (`test_two_agents.py`)
This script loads the trained PyTorch model (`model_episode.pth`). 
- **Manual Spawning Feature**: Automatic box spawning is disabled. Instead, an event listener hooks into Pygame's mouse inputs. You can click on any empty shelf in the visual grid to manually spawn a box. The environment detects the click, adds the coordinates to the global queue, and the robots will instantly respond and navigate to the newly spawned box.

### Manual Play Mode (`play.py`)
If you want to test the environment physics yourself, `play.py` provides a manual control mode using `WASD` to move and `E` to interact. This is highly useful for debugging grid alignment, collision boundaries, and BFS pathing.

## 8. Summary of File Structure

- `unified_env.py`: The Gym environment containing all physics, state generation, action masking, reward shaping, and R1's logic.
- `robot.py`: Defines the visual and physical properties of the robots, including pixel-to-grid conversions and interact boundary checks.
- `world.py` & `sprites.py`: Generators and wrappers for the Pygame visual assets (shelves, stations, platforms).
- `constants.py`: Hardcoded grid dimensions, padding, tile sizes, and image loading.
- `train_unified_agent2.py`: The DQN reinforcement learning loop for Robot 2.
- `q_learning_agent.py`: The classical Q-table implementation used by Robot 1.
- `test_two_agents.py`: The visual evaluation script with interactive mouse-click spawning.
- `click_controller.py`: Logic module to handle mapping Pygame pixel clicks back to grid coordinates.
