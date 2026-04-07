import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pickle

# Ensure you have your environment file ready
from warehouse_env import WarehouseEnv

# GPU/CPU Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


# ─── ARCHITECTURE ────────────────────────────────────────────────────────────

class QNet(nn.Module):
    """ Dueling DQN Architecture """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        # Advantage stream: How much better is action A than average?
        self.advantage = nn.Linear(128, action_dim)
        # Value stream: How good is it to be in this state?
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature(x)
        val = self.value(x)
        adv = self.advantage(x)
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A))
        return val + (adv - adv.mean(dim=1, keepdim=True))


# ─── TRAINING FUNCTION ───────────────────────────────────────────────────────

def train():
    # 1. Initialize Environment and Networks
    env = WarehouseEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNet(state_dim, action_dim).to(DEVICE)
    target_net = QNet(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = deque(maxlen=50000)

    # 2. Hyperparameters
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update_freq = 10
    save_freq = 50
    total_episodes = 2000

    # 3. Bookkeeping
    os.makedirs("checkpoints", exist_ok=True)
    best_score = -1

    # ─── MAIN EPISODE LOOP ───
    for episode in range(total_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # A. Action Selection (Epsilon-Greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()

            # B. Environment Step
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # C. Store Transition
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # D. Optimization Step
            if len(memory) > batch_size:
                # Sample random batch from buffer
                batch = random.sample(memory, batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

                # Convert to Tensors
                s_t = torch.as_tensor(np.array(s_batch), dtype=torch.float32, device=DEVICE)
                a_t = torch.as_tensor(a_batch, dtype=torch.long, device=DEVICE).unsqueeze(1)
                r_t = torch.as_tensor(r_batch, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                ns_t = torch.as_tensor(np.array(ns_batch), dtype=torch.float32, device=DEVICE)
                d_t = torch.as_tensor(d_batch, dtype=torch.float32, device=DEVICE).unsqueeze(1)

                # Current Q values
                current_q = policy_net(s_t).gather(1, a_t)

                # Double DQN: Policy picks action, Target evaluates
                with torch.no_grad():
                    next_actions = policy_net(ns_t).argmax(dim=1, keepdim=True)
                    next_q_values = target_net(ns_t).gather(1, next_actions)
                    expected_q = r_t + (gamma * next_q_values * (1 - d_t))

                # Loss Calculation & Backprop
                loss = nn.MSELoss()(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ─── POST-EPISODE LOGIC ───

        # Update Target Network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save Best Model (based on deliveries/score)
        if env.score >= best_score:
            best_score = env.score
            torch.save(policy_net.state_dict(), "checkpoints/best_model.pt")
            print(f"  ★ New Best Score: {best_score}! Model Saved.")

        # Regular Checkpoint & Buffer Saving
        if episode % save_freq == 0 and episode > 0:
            ckpt_path = f"checkpoints/model_ep{episode}.pt"
            torch.save({
                'episode': episode,
                'policy': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': epsilon
            }, ckpt_path)

            # Save Buffer for Resuming Training
            buffer_path = f"checkpoints/buffer_ep{episode}.pkl"
            with open(buffer_path, "wb") as f:
                pickle.dump(list(memory), f)
            print(f"  💾 Checkpoint & Buffer saved at Ep {episode}")

        # Epsilon Decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Logging
        if episode % 1 == 0:
            print(
                f"Ep {episode:4d} | Score: {env.score:2d} | Reward: {total_reward:7.2f} | Eps: {epsilon:.3f} | Buf: {len(memory)}")


if __name__ == "__main__":
    train()