# Warehouse Simulation 2D - Project Overview

## Project Description

This is a **2D Warehouse Automation Simulation** project that uses reinforcement learning (RL) to train autonomous agents (robots) to efficiently manage warehouse operations. The project implements multi-agent learning where two robots collaborate to pick up boxes from shelves and deliver them to drop-off platforms while optimizing their paths and managing battery charges.

## What This Code Achieves

### Core Objectives

1. **Multi-Agent Reinforcement Learning**
   - Implements dual Q-learning agents that learn optimal warehouse navigation policies
   - Uses both classical Q-Learning and Deep Q-Networks (DQN) for decision-making
   - Agents learn to maximize cumulative rewards while minimizing penalties

2. **Autonomous Robot Navigation**
   - Robots navigate a grid-based 2D warehouse environment
   - Dynamic pathfinding using BFS (Breadth-First Search) for optimal route planning
   - Collision avoidance and obstacle detection
   - Movement control: up, down, left, right actions

3. **Warehouse Task Management**
   - **Picking Phase**: Robots navigate to shelves and pick up boxes
   - **Delivery Phase**: Robots carry boxes to drop-off platforms and deliver them
   - **Homing Phase**: Robots return to charging stations when battery is low
   - **Charging**: Robots dock at charging stations to recharge their batteries

4. **Visual Simulation with Pygame**
   - Real-time 2D graphical interface showing warehouse layout
   - Visual representation of robots, shelves, boxes, and charging stations
   - Interactive gameplay and AI training visualization
   - HUD (Heads-Up Display) showing status information and notifications

5. **Training & Learning**
   - Trains agents over multiple episodes to learn optimal behavior
   - Uses reward shaping to guide agents toward desired behaviors:
     - Positive rewards for successful pickups and deliveries
     - Negative rewards (penalties) for collisions and inefficient moves
   - Saves model checkpoints to track training progress

### Key Features

- **Grid-Based Environment**: 21x15 grid with specific warehouse layout
- **Resource Management**: Battery charging mechanics for realistic constraints
- **Collision Detection**: Prevents robots from moving through walls or each other
- **State Representation**: 8-dimensional observation space including:
  - Robot position (rx, ry)
  - Loaded status (carrying box or not)
  - Target shelf position (bx, by)
  - Drop-off location (dropoff_x, dropoff_y)
  - Direction facing
  
- **Action Space**: 4 discrete actions (move up, down, left, right)

- **Reward System**: 
  - Step penalty: -1 per action
  - Collision penalty: -50 to -35
  - Pickup reward: +10
  - Delivery reward: +20
  - Movement efficiency rewards/penalties based on proximity to targets

## Project Structure

```
warehouse-simulation-2D/
├── world.py                    # Warehouse environment setup and map creation
├── robot.py                    # Robot class definition and movement logic
├── sprites.py                  # Game objects (Shelf, ChargeStation, etc.)
├── constants.py                # Configuration and constants
├── unified_env.py              # Gymnasium environment wrapper for the warehouse
├── q_learning_agent.py         # Classical Q-Learning agent implementation
├── train_unified_agent2.py     # Deep Q-Network (DQN) training script
├── play.py                     # Interactive play/demo mode
├── test_two_agents.py          # Multi-agent testing script
├── click_controller.py         # Click-based control system
├── assets/                     # Game sprites and images
└── checkpoints/                # Saved model weights and training data
```

## Technologies Used

- **PyTorch**: Deep neural networks for DQN agent training
- **Gymnasium**: Reinforcement learning environment framework
- **Pygame**: 2D graphics rendering and game simulation
- **NumPy**: Numerical computations and array operations
- **PettingZoo**: Multi-agent reinforcement learning framework

## Running the Project

### Installation
```bash
pip install -r requirements.txt
```

### Training the Agents
```bash
python train_unified_agent2.py  # Train Deep Q-Network agents
```

### Interactive Play
```bash
python play.py                  # Play in the warehouse simulation
```

### Testing
```bash
python test_two_agents.py       # Test trained agents in environment
```

## Learning Outcomes

This project demonstrates:
- Multi-agent reinforcement learning principles
- Policy optimization through reward shaping
- Neural network-based decision making
- Discrete action space environments
- Real-time simulation and visualization
- Efficient pathfinding in constrained environments
- Collaborative task management between agents
