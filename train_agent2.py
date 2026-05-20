import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pickle
from constants import *

from two_agent_warehouse_env import TwoAgentWarehouseEnv
from train import QNetwork   # reuse the exact same architecture — no changes


if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

print(f"Training on: {compute_device}")


# ─── TRAINING FLAGS ───────────────────────────────────────────────────────────

RENDER_DURING_TRAINING = False
RENDER_EVERY_N_EPISODES = 1

# Agent 1 uses a Q-table trained by DualQAgent — never modified during this run.
AGENT1_QTABLE_FOLDER = "training_data"
AGENT1_QTABLE_FILE   = "warehouse_data.pkl"

# Agent 2's checkpoints are saved here, separate from Agent 1's.
AGENT2_CHECKPOINT_DIR = "checkpoints_agent2"


# ─── TRAINING FUNCTION ────────────────────────────────────────────────────────

def train():
    render_mode = "human" if RENDER_DURING_TRAINING else None

    # ── Environment ───────────────────────────────────────────────────────────
    # TwoAgentWarehouseEnv internally:
    #   • loads Agent 1's Q-table from AGENT1_QTABLE_FOLDER/AGENT1_QTABLE_FILE
    #   • wraps it in QTablePolicy for deterministic (epsilon=0) action selection
    #   • runs Agent 1's TrainingEnv as a silent, deterministic moving obstacle
    #   • exposes Agent 2's Agent2TrainingEnv (obs_size=23) as the training surface
    env = TwoAgentWarehouseEnv(
        agent1_qtable_path=AGENT1_QTABLE_FILE,
        agent1_qtable_folder=AGENT1_QTABLE_FOLDER,
        compute_device=compute_device,
        render_mode=render_mode,
    )

    # state_dim is 23 (19 base + 4 agent1 features).
    # action_dim is still 6 — unchanged.
    state_dim = env.observation_space.shape[0]    # 23
    action_dim = env.action_space.n               # 6

    # ── Agent 2 networks ──────────────────────────────────────────────────────
    # Same Dueling DQN architecture as Agent 1, just with state_dim=23.
    # Agent 1's QNetwork (state_dim=19) is loaded inside TwoAgentWarehouseEnv
    # and is completely separate from these two.
    agent2_policy_network = QNetwork(state_dim, action_dim).to(compute_device)
    agent2_target_network = QNetwork(state_dim, action_dim).to(compute_device)
    agent2_target_network.load_state_dict(agent2_policy_network.state_dict())

    agent2_optimizer = optim.Adam(agent2_policy_network.parameters(), lr=1e-4)
    agent2_replay_buffer = deque(maxlen=50000)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # Kept identical to train.py so the training dynamics are comparable.
    batch_size = 128
    discount_factor = 0.98
    epsilon = 0.7
    epsilon_min = 0.1
    epsilon_decay = 0.998
    target_network_update_frequency = 10
    checkpoint_save_frequency = 50
    total_episodes = 2500

    os.makedirs(AGENT2_CHECKPOINT_DIR, exist_ok=True)
    best_agent2_delivery_score = -1

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode in range(total_episodes):
        current_state, _ = env.reset()
        episode_total_reward = 0.0
        is_done = False

        should_render_this_episode = (
            RENDER_DURING_TRAINING and (episode % RENDER_EVERY_N_EPISODES == 0)
        )

        while not is_done:
            # ── Action selection (ε-greedy with heuristic bias) ───────────────
            # Mirrors train.py exactly: 70% heuristic during exploration,
            # 30% random. Agent 2's heuristic navigates toward its own target
            # via BFS — it knows nothing about Agent 1 at this level.
            if random.random() < epsilon:
                if random.random() < 0.7:
                    chosen_action = env.heuristic_action()
                else:
                    chosen_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(
                        current_state, dtype=torch.float32, device=compute_device
                    ).unsqueeze(0)
                    chosen_action = agent2_policy_network(
                        state_tensor
                    ).argmax().item()

            # ── Environment step ──────────────────────────────────────────────
            # Internally: Agent 1 also steps via its frozen policy.
            # Agent 2 receives collision/proximity-adjusted reward.
            next_state, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated

            if should_render_this_episode:
                env.render()

            # ── Replay buffer (Agent 2 only) ───────────────────────────────────
            agent2_replay_buffer.append(
                (current_state, chosen_action, reward, next_state, is_done)
            )
            current_state = next_state
            episode_total_reward += reward

            # ── Learning step ──────────────────────────────────────────────────
            if len(agent2_replay_buffer) > batch_size:
                sampled_batch = random.sample(agent2_replay_buffer, batch_size)
                (
                    states_batch,
                    actions_batch,
                    rewards_batch,
                    next_states_batch,
                    dones_batch,
                ) = zip(*sampled_batch)

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

                # Double DQN: policy selects action, target evaluates it.
                current_q_values = agent2_policy_network(states_tensor).gather(
                    1, actions_tensor
                )

                with torch.no_grad():
                    best_next_actions = agent2_policy_network(
                        next_states_tensor
                    ).argmax(dim=1, keepdim=True)
                    next_q_values = agent2_target_network(
                        next_states_tensor
                    ).gather(1, best_next_actions)
                    expected_q_values = rewards_tensor + (
                        discount_factor * next_q_values * (1 - dones_tensor)
                    )

                loss = nn.MSELoss()(current_q_values, expected_q_values)
                agent2_optimizer.zero_grad()
                loss.backward()
                agent2_optimizer.step()

        # ── Target network sync ────────────────────────────────────────────────
        if episode % target_network_update_frequency == 0:
            agent2_target_network.load_state_dict(
                agent2_policy_network.state_dict()
            )

        # ── Save best model ────────────────────────────────────────────────────
        # env.score tracks Agent 2's delivery count only (not Agent 1's).
        if env.score > best_agent2_delivery_score:
            best_agent2_delivery_score = env.score
            torch.save(
                agent2_policy_network.state_dict(),
                os.path.join(AGENT2_CHECKPOINT_DIR, "best_model.pt"),
            )
            print(f"  ★ New Best Agent 2 Score: {best_agent2_delivery_score}! Model saved.")

        # ── Periodic checkpoint + buffer ───────────────────────────────────────
        if episode % checkpoint_save_frequency == 0 and episode > 0:
            torch.save(
                {
                    "episode": episode,
                    "policy": agent2_policy_network.state_dict(),
                    "optimizer": agent2_optimizer.state_dict(),
                    "epsilon": epsilon,
                },
                os.path.join(AGENT2_CHECKPOINT_DIR, "model_ep_latest.pt"),
            )
            with open(
                os.path.join(AGENT2_CHECKPOINT_DIR, "buffer_ep_latest.pkl"), "wb"
            ) as buffer_file:
                pickle.dump(list(agent2_replay_buffer), buffer_file)
            print(f"  💾 Agent 2 checkpoint & buffer saved at episode {episode}")

        # ── Epsilon decay ──────────────────────────────────────────────────────
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        render_indicator = " 👁" if should_render_this_episode else ""
        print(
            f"Ep {episode:4d} | "
            f"Agent2 Score: {env.score:2d} | "
            f"Collisions: {env.agent2_collision_count:3d} | "
            f"Reward: {episode_total_reward:7.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(agent2_replay_buffer)}"
            f"{render_indicator}"
        )


if __name__ == "__main__":
    train()