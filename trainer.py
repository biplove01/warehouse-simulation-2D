import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pickle

from warehouse_env import WarehouseEnv

# ─── DEVICE SETUP ────────────────────────────────────────────────────────────

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {compute_device}")

RENDER_DURING_TRAINING = False
RENDER_EVERY_N_EPISODES = 1


# ─── ARCHITECTURE ────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Dueling DQN Architecture."""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        # Advantage stream: How much better is action A than the average action?
        self.advantage_stream = nn.Linear(128, action_dim)

        # Value stream: How good is it to simply be in this state?
        self.value_stream = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared_features(state)
        state_value = self.value_stream(features)
        action_advantages = self.advantage_stream(features)

        # Combine streams: Q(s, a) = V(s) + (A(s, a) - mean(A))
        return state_value + (action_advantages - action_advantages.mean(dim=1, keepdim=True))


# ─── TRAINING FUNCTION ───────────────────────────────────────────────────────

def train():

    # 1. Initialize environment and networks
    #    Render mode is set once here based on the toggle above.
    #    If rendering every N episodes, we start with "human" mode on
    #    and let the episode loop control when to actually call render().
    render_mode = "human" if RENDER_DURING_TRAINING else None
    env = WarehouseEnv(render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_network = QNetwork(state_dim, action_dim).to(compute_device)
    target_network = QNetwork(state_dim, action_dim).to(compute_device)
    target_network.load_state_dict(policy_network.state_dict())

    optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)
    replay_buffer = deque(maxlen=50000)

    # 2. Hyperparameters
    batch_size = 128
    discount_factor = 0.98
    epsilon = 0.65
    epsilon_min = 0.05
    epsilon_decay = 0.998
    target_network_update_frequency = 10
    checkpoint_save_frequency = 50
    total_episodes = 2500

    # 3. Bookkeeping
    os.makedirs("checkpoints", exist_ok=True)
    best_delivery_score = -1

    # ─── MAIN EPISODE LOOP ───────────────────────────────────────────────────

    for episode in range(total_episodes):
        current_state, _ = env.reset()
        episode_total_reward = 0
        is_done = False

        # Decide whether to render this episode
        should_render_this_episode = (
            RENDER_DURING_TRAINING and (episode % RENDER_EVERY_N_EPISODES == 0)
        )

        while not is_done:

            # A. Action selection via epsilon-greedy policy
            if random.random() < epsilon:
                if random.random() < 0.7:  # 70% of the time be smart, 30% be random
                    chosen_action = env.heuristic_action()
                else:
                    chosen_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(
                        current_state, dtype=torch.float32, device=compute_device
                    ).unsqueeze(0)
                    chosen_action = policy_network(state_tensor).argmax().item()

            # B. Step the environment
            next_state, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated

            # C. Render this step if visual mode is on for this episode
            if should_render_this_episode:
                env.render()

            # D. Store transition in replay buffer
            replay_buffer.append((current_state, chosen_action, reward, next_state, is_done))
            current_state = next_state
            episode_total_reward += reward

            # E. Optimization step (only when buffer has enough samples)
            if len(replay_buffer) > batch_size:

                # Sample a random mini-batch
                sampled_batch = random.sample(replay_buffer, batch_size)
                (
                    states_batch,
                    actions_batch,
                    rewards_batch,
                    next_states_batch,
                    dones_batch,
                ) = zip(*sampled_batch)

                # Convert to tensors
                states_tensor = torch.as_tensor(
                    np.array(states_batch), dtype=torch.float32, device=compute_device
                )
                actions_tensor = torch.as_tensor(
                    actions_batch, dtype=torch.long, device=compute_device
                ).unsqueeze(1)
                rewards_tensor = torch.as_tensor(
                    rewards_batch, dtype=torch.float32, device=compute_device
                ).unsqueeze(1)
                next_states_tensor = torch.as_tensor(
                    np.array(next_states_batch), dtype=torch.float32, device=compute_device
                )
                dones_tensor = torch.as_tensor(
                    dones_batch, dtype=torch.float32, device=compute_device
                ).unsqueeze(1)

                # Current Q-values for the actions that were actually taken
                current_q_values = policy_network(states_tensor).gather(1, actions_tensor)

                # Double DQN: policy network picks the next action,
                # target network evaluates it
                with torch.no_grad():
                    best_next_actions = policy_network(next_states_tensor).argmax(
                        dim=1, keepdim=True
                    )
                    next_q_values = target_network(next_states_tensor).gather(
                        1, best_next_actions
                    )
                    expected_q_values = rewards_tensor + (
                        discount_factor * next_q_values * (1 - dones_tensor)
                    )

                # Compute loss and backpropagate
                loss = nn.MSELoss()(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ─── POST-EPISODE LOGIC ──────────────────────────────────────────────

        # Periodically sync target network weights with policy network
        if episode % target_network_update_frequency == 0:
            target_network.load_state_dict(policy_network.state_dict())

        # Save model if this episode achieved a new best delivery score
        if env.score > best_delivery_score:
            best_delivery_score = env.score
            torch.save(policy_network.state_dict(), "checkpoints/best_model.pt")
            print(f"  ★ New Best Score: {best_delivery_score}! Model saved.")

        # Save periodic checkpoint and replay buffer for resuming training
        if episode % checkpoint_save_frequency == 0 and episode > 0:
            checkpoint_path = f"checkpoints/model_ep_latest.pt"
            torch.save(
                {
                    "episode": episode,
                    "policy": policy_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epsilon": epsilon,
                },
                checkpoint_path,
            )

            replay_buffer_path = f"checkpoints/buffer_ep_latest.pkl"
            with open(replay_buffer_path, "wb") as buffer_file:
                pickle.dump(list(replay_buffer), buffer_file)

            print(f"  💾 Checkpoint & buffer saved at episode {episode}")

        # Decay epsilon toward its minimum floor
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log episode summary
        render_indicator = " 👁" if should_render_this_episode else ""
        print(
            f"Ep {episode:4d} | "
            f"Score: {env.score:2d} | "
            f"Reward: {episode_total_reward:7.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(replay_buffer)}"
            f"{render_indicator}"
        )


if __name__ == "__main__":
    train()