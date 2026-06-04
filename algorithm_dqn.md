# Deep Q-Network (DQN) Algorithm in Dual-Agent Warehouse Simulation

This document explains the advanced reinforcement learning mechanics used to train Robot 2 in the dual-agent environment. Because the state space in the multi-agent setup is far too large for a standard Q-Table, the project uses a **Deep Q-Network (DQN)**.

## 1. Why Deep Q-Learning?

In a simple environment, a Q-Table maps every possible state to an action. However, in this dual-agent environment, Robot 2 needs to track 25 continuous variables (positions, radar for walls, Robot 1's position/vector). 

A 25-dimensional continuous state vector creates an infinite number of possible states. A Q-Table cannot store this. Instead, a **Neural Network** is used as a universal function approximator to estimate the Q-Values.

## 2. Neural Network Fundamentals: How It Works

A Neural Network (NN) consists of artificial "neurons" connected by **Weights** ($W$) and **Biases** ($b$). 
- **Weights** determine the importance of an input (e.g., if a wall is detected on the left, the weight heavily decreases the Q-Value of moving left).
- **Biases** shift the output to help the network fit the data better.

### Forward Propagation (Making a Decision)
The brain of Robot 2 is a Multi-Layer Perceptron (MLP) built with PyTorch. When the robot makes a decision, data flows forward through the network:

1. **Input Layer ($X$)**: The 25-dimensional state vector is fed into the network.
2. **Hidden Layer 1**: The inputs are multiplied by a weight matrix ($W_1$), added to a bias vector ($b_1$), and passed through a **ReLU** (Rectified Linear Unit) activation function.
   - Mathematics: $H_1 = \max(0, X \cdot W_1 + b_1)$
   - Shape: 128 neurons.
3. **Hidden Layer 2**: The outputs of Layer 1 are passed through another linear transformation and ReLU.
   - Mathematics: $H_2 = \max(0, H_1 \cdot W_2 + b_2)$
   - Shape: 128 neurons.
4. **Output Layer**: A final linear transformation computes the estimated Q-Values for all 6 possible actions.
   - Mathematics: $Q = H_2 \cdot W_3 + b_3$

The action with the highest Q-Value output is selected as the robot's next move.

## 3. Finding the Error (The Loss Function)

For the Neural Network to learn, it needs to know how wrong it was. In RL, we don't have labeled data (like "Action 2 was the correct answer"). Instead, we use the **Bellman Equation** to generate a "Target".

To prevent the network from chasing a moving target (which causes catastrophic instability), DQN uses **two separate networks**:
1. `policy_net`: The active network being trained.
2. `target_net`: A frozen, older copy of the `policy_net` used strictly to calculate the target values.

**The Target Calculation:**
$$Target\_Q = R_{t+1} + \gamma \max_{a} Q_{target}(S_{t+1}, a)$$

**The Error (Mean Squared Error):**
The network computes the difference between what it *predicted* (`current_q`) and what the Bellman Equation says it *should have predicted* (`target_q`).
$$Loss = \frac{1}{N} \sum (Target\_Q - Current\_Q)^2$$

In `train_unified_agent2.py`, this is executed as:
```python
loss = nn.MSELoss()(current_q, target_q)
```

## 4. Backpropagation and Gradient Descent

Once the Loss is calculated, the network must adjust its weights to make the error smaller next time.

### Backpropagation
Using a process called Backpropagation, PyTorch calculates the **Gradient** of the Loss with respect to every single weight and bias. The gradient represents the "slope of the error". If a specific weight caused the Q-Value to be too high, the gradient tells the network to lower that weight.

### Gradient Descent (The Adam Optimizer)
**Gradient Descent** is the algorithm used to update the weights. Imagine standing on a hilly landscape blindfolded; to reach the bottom (zero error), you take steps down the steepest slope (the gradient).

Instead of standard Stochastic Gradient Descent (SGD), this project uses the **Adam (Adaptive Moment Estimation)** optimizer. Adam dynamically adjusts the learning rate ($1e-4$) for each individual weight using momentum. It speeds up learning and prevents the network from getting trapped in local minimums.

```python
optimizer.zero_grad() # Clears old gradients
loss.backward()       # BACKPROPAGATION: Computes the new gradients (the slope)

# Prevents "exploding gradients" (where weights change too violently and break the network)
nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0) 

optimizer.step()      # GRADIENT DESCENT: Adam adjusts the weights to lower the Loss
```

## 5. How Training Converges

**Convergence** means the neural network has successfully learned the optimal policy. The weights stop changing significantly, the Loss drops to a stable low value, and the robot efficiently delivers boxes without crashing.

Convergence in this project is achieved through several interacting mechanisms:

1. **Experience Replay**: Instead of learning from sequential steps, experiences are stored in a 50,000-step buffer. Training pulls a random mini-batch of 128 experiences. This breaks correlations and ensures the network learns a generalized policy rather than hyper-fixating on recent events.
2. **Target Network Syncing**: Every 10 episodes, the `target_net` copies the weights of the `policy_net`. This stabilizes the Bellman Targets, allowing the `policy_net` to converge towards a fixed goal.
3. **Epsilon Decay ($\epsilon$)**: 
   - Early in training ($\epsilon = 0.70$), the robot moves randomly to explore the map and fill the Replay Buffer.
   - Over time, $\epsilon$ decays (shrinks) to $0.05$. As the robot stops acting randomly and starts *exploiting* its trained Neural Network, the experiences added to the buffer become high-quality, optimal paths.
   - This shift from random exploration to deterministic exploitation allows the gradients to settle and the network to officially converge on the optimal warehouse routing policy.
