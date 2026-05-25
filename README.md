# Warehouse Simulation 2D - Multi-Agent Reinforcement Learning

A comprehensive 2D warehouse automation simulation that trains autonomous agents (robots) using reinforcement learning to efficiently manage warehouse operations including picking, delivering, and charging logistics.

---

## 📋 Project Overview

This project implements a **multi-agent reinforcement learning system** where two autonomous robots collaborate in a simulated 2D warehouse environment. The robots learn to optimize their behavior through interaction with the environment, using both classical Q-Learning and Deep Q-Networks (DQN) to solve complex warehouse tasks.

### Core Mission

Train intelligent warehouse robots to:
- 🤖 Autonomously navigate a grid-based warehouse
- 📦 Pick up boxes from shelves efficiently
- 🎯 Deliver boxes to drop-off platforms
- 🔋 Manage battery levels and return for charging
- 🚀 Collaborate without collisions or conflicts
- 💡 Learn optimal policies through reinforcement learning

---

## 🎯 What This Code Achieves

### 1. **Multi-Agent Reinforcement Learning System**
   - Implements **Dual Q-Learning Agents** that independently learn warehouse policies
   - Uses **Deep Q-Networks (DQN)** with neural networks for complex decision-making
   - Trains agents over thousands of episodes to discover optimal behaviors
   - Agents learn state-action value functions (Q-tables) mapping warehouse situations to best actions

### 2. **Autonomous Robot Navigation**
   - Grid-based pathfinding with **BFS (Breadth-First Search)** for optimal routes
   - Dynamic obstacle avoidance and collision detection
   - 4-directional movement (up, down, left, right)
   - Visual feedback showing robot direction and load status
   - Real-time decision-making with action masking for invalid moves

### 3. **Warehouse Task Management Pipeline**
   
   **Fetching Phase:**
   - Robot receives task to pick up box at specific shelf
   - Navigates from current position to target shelf
   - Uses BFS to compute optimal distance-based rewards
   
   **Delivery Phase:**
   - Once box is picked (loaded state = true), robot carries it
   - Navigates to assigned drop-off platform
   - Places box and completes delivery task
   
   **Homing Phase:**
   - Robot periodically returns to charging station
   - Battery status influences decision-making
   - Recharges before accepting new tasks
   - Prevents runaway agent scenario

### 4. **Visual Simulation with Real-Time Graphics**
   - **Pygame-based 2D visualization** of the entire warehouse
   - Warehouse Layout:
     - **Grid Dimensions**: 21×15 cells
     - **5 Charging Stations**: Top-left corner for robot docking
     - **45+ Shelves**: Distributed across warehouse in strategic pattern
     - **4 Drop-off Platforms**: Bottom section for package delivery
   
   - Interactive HUD displaying:
     - Episode progress and rewards
     - Robot status (position, battery, loaded state)
     - Real-time score tracking
     - Notification system for key events

### 5. **Intelligent Reward Shaping**
   Agents learn through carefully designed reward signals:
   
   | Action/Event | Reward/Penalty |
   |---|---|
   | Each step taken | -1 |
   | Move toward target | +2 |
   | Move away from target | -8 |
   | Collision with wall | -35 |
   | Robot-robot collision | -50 |
   | Successfully pick up box | +10 |
   | Successfully deliver box | +20 |
   | Invalid interaction | -25 |
   | Stay at home idle | +0.5 |
   | Wander aimlessly | -1.5 |
   | Wrong drop-off hover | -2 |

### 6. **State-Action Space Definition**

   **Observation Space (8-dimensional):**
   ```
   [robot_x, robot_y, loaded, shelf_x, shelf_y, dropoff_x, dropoff_y, direction]
   ```
   
   **Action Space (4 discrete actions):**
   - 0: Move up
   - 1: Move down
   - 2: Move left
   - 3: Move right

### 7. **Deep Learning Integration**
   - **Neural Network Architecture**: 
     - Input layer: 8 features (state space)
     - Hidden layers: 128 neurons with ReLU activation
     - Output layer: 4 neurons (Q-values for each action)
   - **Training Technique**: Experience replay with target networks
   - **Model Persistence**: Saves best checkpoints during training

### 8. **Collision Avoidance & Physics**
   - Detects collisions with walls, shelves, and other robots
   - Action masking prevents invalid moves before execution
   - Maintains grid-based discrete positioning
   - Ensures deterministic movement mechanics

---

## 📁 Project Architecture

### Core Modules

| File | Purpose |
|------|---------|
| **unified_env.py** | Gymnasium environment wrapper; implements step/reset logic, reward calculation, state management |
| **q_learning_agent.py** | Classical Q-Learning agent with epsilon-greedy exploration |
| **train_unified_agent2.py** | Deep Q-Network training loop; handles DQN agent creation, episode execution, checkpointing |
| **world.py** | Warehouse map generation; places shelves, charging stations, drop-off platforms |
| **robot.py** | Robot entity class; handles movement, direction, loaded state, sprite management |
| **sprites.py** | Game object definitions (Shelf, ChargeStation, DropoffPlatform) |
| **play.py** | Interactive gameplay mode with BFS visualization and manual control |
| **test_two_agents.py** | Evaluation script for testing trained agents in multi-agent scenarios |
| **constants.py** | Global configuration (grid size, colors, dimensions, reward values) |
| **click_controller.py** | Click-based interaction system for warehouse objects |

### Directory Structure

```
warehouse-simulation-2D/
├── README.md                   # Project documentation
├── REQUIREMENTS.md             # Detailed project overview
├── requirements.txt            # Python dependencies
├── constants.py                # Global configuration & constants
├── world.py                    # Warehouse map generation
├── robot.py                    # Robot class definition
├── sprites.py                  # Game object classes
├── unified_env.py              # RL environment implementation
├── q_learning_agent.py         # Q-Learning agent
├── train_unified_agent2.py     # DQN training script
├── play.py                     # Interactive demo mode
├── test_two_agents.py          # Multi-agent testing
├── click_controller.py         # Click interaction handler
├── assets/                     # Sprite images and graphics
│   └── wa/                     # Asset subdirectory
└── checkpoints/                # Model checkpoints & training data
    ├── best_model.pth          # Best performing model weights
    ├── latest_model.pth        # Latest checkpoint
    ├── model_episode.pth       # Episode-specific weights
    └── model_final_exit.pth    # Final training state
```

---

## 🔧 Technology Stack

| Technology | Purpose |
|---|---|
| **PyTorch** | Deep neural networks for DQN training; GPU acceleration support |
| **Gymnasium** | Standard RL environment interface and benchmarking |
| **Pygame** | 2D graphics rendering and real-time visualization |
| **NumPy** | Numerical computations, matrix operations, state arrays |
| **PettingZoo** | Multi-agent RL framework for agent coordination |
| **Python 3.8+** | Core programming language |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Option 1: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Manual Installation with CUDA Support
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pettingzoo pygame numpy gymnasium
```

### Minimum Required Packages
```bash
pip install pygame torch gymnasium numpy
```

---

## 🚀 Usage & Running the Project

### 1. Train Deep Q-Network Agents
```bash
python train_unified_agent2.py
```
**What it does:**
- Initializes two DQN agents in the warehouse environment
- Runs training loop over multiple episodes
- Saves model checkpoints to `checkpoints/` directory
- Displays training progress and reward curves

### 2. Interactive Play Mode
```bash
python play.py
```
**Features:**
- Manual control of robot movement
- Real-time BFS pathfinding visualization
- HUD showing current stats
- Click-based object interaction
- Immediate feedback on actions

### 3. Test Trained Agents
```bash
python test_two_agents.py
```
**Evaluates:**
- Performance of trained agents
- Multi-agent coordination
- Task completion rates
- Collision avoidance effectiveness

### 4. Train Q-Learning Agents (Alternative)
```bash
python train_unified_agent2.py --use-qlearning
```
Uses classical Q-Learning instead of DQN.

---

## 📊 Key Performance Metrics

The system tracks and optimizes:

- **Episode Reward**: Total reward accumulated per episode
- **Task Success Rate**: % of successful pickups and deliveries
- **Collision Count**: Number of collision events
- **Average Path Length**: Efficiency of robot navigation
- **Convergence Speed**: Episodes until policy stabilization
- **Multi-Agent Coordination**: Collision avoidance between robots

---

## 🧠 Learning Mechanism

### Reinforcement Learning Process

1. **Observation**: Agent perceives warehouse state (8-dimensional vector)
2. **Decision**: Agent selects action using epsilon-greedy policy:
   - Exploration (random): probability ε
   - Exploitation (best known): probability 1-ε
3. **Execution**: Action applied; robot moves or interacts
4. **Reward**: Environment provides reward signal based on action outcome
5. **Learning**: Agent updates Q-values or neural network weights
6. **Repeat**: Process continues until episode terminates

### Epsilon Decay Schedule
- Initial epsilon (exploration): 1.0
- Decay per episode: 0.999
- Minimum epsilon: 0.01
- This balances exploration in early training and exploitation later

### Deep Q-Network (DQN) Advantages
- Handles larger state spaces more efficiently than Q-tables
- Learns feature representations automatically
- Generalizes across similar states
- Enables transfer learning potential

---

## 🎓 Educational Value

This project demonstrates:

✅ **Reinforcement Learning Fundamentals**
- Markov Decision Processes (MDPs)
- Q-Learning and Deep Q-Networks
- Policy gradient methods
- Reward shaping techniques

✅ **Multi-Agent Systems**
- Concurrent agent execution
- Resource conflict resolution
- Collaborative task management
- Action masking and constraint satisfaction

✅ **Computer Graphics**
- Sprite rendering and animation
- HUD design and information visualization
- Real-time performance optimization

✅ **Software Engineering**
- Modular architecture
- Clear separation of concerns
- Configuration management
- Experiment tracking and checkpointing

---

## 🔄 Workflow Example: A Single Episode

```
1. [RESET] Environment initializes; robots at home positions
2. [STEP 1] Agent perceives state; chooses move action
3. [UPDATE] Robot moves; no collision detected; reward: -1 (step cost)
4. [STEP 2-5] Agent navigates toward assigned shelf using BFS guidance
5. [STEP 6] Agent reaches shelf; executes pickup action; reward: +10
6. [STEP 7-12] Agent carries box toward drop-off platform
7. [STEP 13] Agent delivers box; reward: +20
8. [STEP 14-15] Agent returns to charging station
9. [EPISODE END] Environment terminates; total reward calculated and logged
```

---

## 💾 Model Checkpoints

The training process saves several checkpoint files:

- **best_model.pth**: Weights from episode with highest cumulative reward
- **latest_model.pth**: Most recent checkpoint (used for resuming training)
- **model_episode.pth**: Periodic snapshots for experiment analysis
- **model_final_exit.pth**: Final state upon training completion

To load and use a checkpoint:
```python
import torch
model = torch.load("checkpoints/best_model.pth")
```

---

## 🛠️ Configuration

Edit `constants.py` to adjust:

| Parameter | Purpose | Default |
|---|---|---|
| `GRID_WIDTH` | Warehouse grid width | 21 |
| `GRID_HEIGHT` | Warehouse grid height | 15 |
| `EPISODE_LIMIT` | Max steps per episode | 500 |
| `EPSILON_DECAY` | Exploration decay rate | 0.999 |
| `LEARNING_RATE` | Neural network learning rate | 0.001 |
| `REWARD_DELIVER` | Delivery completion reward | +20 |
| `PENALTY_COLLISION` | Wall collision penalty | -35 |

---

## 📈 Expected Results

After training for 1000+ episodes, agents typically demonstrate:
- ✅ Reliable navigation without excessive collisions
- ✅ Successful box pickup and delivery completion
- ✅ Intelligent battery management and charging behavior
- ✅ Convergence to stable policies
- ✅ Reduced episode rewards (efficient task completion)

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| ImportError: No module named 'torch' | Run `pip install -r requirements.txt` |
| CUDA out of memory | Set `device='cpu'` in training script |
| Agents not learning | Check reward values in `unified_env.py`; increase episodes |
| Slow rendering | Disable graphics or run headless mode |

---

## 📚 Further Reading & Extensions

Potential enhancements:
- Actor-Critic algorithms for policy learning
- Attention mechanisms for state representation
- Graph neural networks for warehouse topology
- Multi-objective optimization (efficiency vs. fairness)
- Real-world sim-to-real transfer learning
- Decentralized learning without environment server

---

## 📝 Prerequisites to Run This Project

Quick setup:
```bash
# Clone or navigate to project directory
cd warehouse-simulation-2D

# Install dependencies
pip install -r requirements.txt

# Run training
python train_unified_agent2.py

# Or play interactively
python play.py
```

**Core Requirements:**
- Python 3.8+
- pygame - for 2D visualization
- torch - for deep neural networks
- gymnasium - for RL environment framework
- numpy - for numerical operations
- pettingzoo - for multi-agent support

---

## 📄 License & Attribution

This is an educational project demonstrating reinforcement learning principles applied to warehouse automation.

---

## 🎯 Summary

The Warehouse Simulation 2D project successfully demonstrates:
- **Autonomous agent learning** through reinforcement learning
- **Multi-agent coordination** in shared environments  
- **Real-time visualization** of complex AI behavior
- **Practical application** of neural networks and RL algorithms
- **Scalable architecture** for warehouse automation research

Perfect for students and researchers exploring reinforcement learning, game development, and autonomous systems!
