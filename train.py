import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pickle
from constants import *

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y

if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

print(f"Training on: {compute_device}")

RENDER_DURING_TRAINING = False
RENDER_EVERY_N_EPISODES = 1

HOME_WAIT_STEPS_REQUIRED = 4  # consecutive wait steps before next target spawns

render_mode = "human" if RENDER_DURING_TRAINING else None

# ─── TRAINING ENVIRONMENT ────────────────────────────────────────────────────

class TrainingEnv(WarehouseEnv):
    """
    Extends WarehouseEnv with training-specific behavior:
    - Curriculum learning for early episodes
    - Home return phase after every delivery
    - Consecutive wait requirement at home before next target spawns
    """

    # SPAWN_COLUMN_RANGE = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    # SPAWN_ROW_RANGE = {3, 4, 7, 8, 11, 12}
    SPAWN_COLUMN_RANGE = None
    SPAWN_ROW_RANGE = None

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.returning_home = False
        self.consecutive_wait_steps_at_home = 0
        return observation, info

    
    def _spawn_new_target(self):
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        # Filter shelves by configured spawn zone if specified
        filtered_shelves = [
            shelf for shelf in self.shelves
            if (self.SPAWN_COLUMN_RANGE is None or 
                self._to_grid_coords(shelf)[0] in self.SPAWN_COLUMN_RANGE)
            and (self.SPAWN_ROW_RANGE is None or 
                self._to_grid_coords(shelf)[1] in self.SPAWN_ROW_RANGE)
        ]

        # Fall back to all shelves if filter produces empty list
        spawn_pool = filtered_shelves if filtered_shelves else self.shelves

        target_shelf = random.choice(spawn_pool)
        target_shelf.has_box = True
        target_shelf.image = target_shelf.loaded_image
        self.target_grid_x, self.target_grid_y = self._to_grid_coords(target_shelf)
        self.target_distance_map = self._bfs_distance_map(
            self.target_grid_x, self.target_grid_y
        )
        self.returning_home = False


    def _on_delivery(self):
        """After delivery, clear shelves and send robot home before next target."""
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        self.returning_home = True
        self.consecutive_wait_steps_at_home = 0
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map
        # print(f"  🏠 Delivery done, returning to station.")


    def heuristic_action(self):  
        if self.returning_home:
            robot = self.robot
            robot_at_home = (
                robot.grid_x == ROBOT_HOME_GRID_X and
                robot.grid_y == ROBOT_HOME_GRID_Y
            )
            if robot_at_home:
                return 5

            direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            best_action = None
            best_distance = self.home_distance_map.get((robot.grid_x, robot.grid_y), 50)

            for action_index, (delta_x, delta_y) in enumerate(direction_deltas):
                next_x = robot.grid_x + delta_x
                next_y = robot.grid_y + delta_y
                is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
                is_passable = (next_x, next_y) not in self.obstacle_positions
                if is_in_bounds and is_passable:
                    neighbor_distance = self.home_distance_map.get((next_x, next_y), 50)
                    if neighbor_distance < best_distance:
                        best_distance = neighbor_distance
                        best_action = action_index

            if best_action is None:
                return random.randint(0, 3)
            return best_action

        return WarehouseEnv.heuristic_action(self)
    

    def step(self, action):
        robot = self.robot

        # ── HOME RETURN PHASE ─────────────────────────────────────────────────
        if self.returning_home:
            self.steps += 1
            robot_at_home = (
                robot.grid_x == ROBOT_HOME_GRID_X and
                robot.grid_y == ROBOT_HOME_GRID_Y
            )

            if robot_at_home:
                if action == 5:  # Wait — correct behavior at station
                    self.consecutive_wait_steps_at_home += 1
                    reward = 1.0

                    if self.consecutive_wait_steps_at_home >= HOME_WAIT_STEPS_REQUIRED:
                        # Done waiting — spawn next target and resume
                        self.consecutive_wait_steps_at_home = 0
                        self.returning_home = False
                        self._spawn_new_target()
                        # print(f"  ✅ Wait complete, spawning next target.")
                else:
                    # Wrong action at home — reset wait counter
                    self.consecutive_wait_steps_at_home = 0
                    reward = -1.0
            else:
                # Not home yet — navigate using home BFS distance map
                self.consecutive_wait_steps_at_home = 0
                distance_before = self.home_distance_map.get((robot.grid_x, robot.grid_y), 50)

                if action < 4:
                    direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                    delta_x, delta_y = direction_deltas[action]
                    next_x = robot.grid_x + delta_x
                    next_y = robot.grid_y + delta_y

                    is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
                    is_passable = (next_x, next_y) not in self.obstacle_positions

                    if is_in_bounds and is_passable:
                        robot.grid_x, robot.grid_y = next_x, next_y
                        distance_after = self.home_distance_map.get((robot.grid_x, robot.grid_y), 50)
                        distance_delta = distance_before - distance_after
                        if distance_delta >= 0:
                            reward = distance_delta * self.reward_manager.progress_reward_scale
                        else:
                            reward = distance_delta * self.reward_manager.regress_penalty_scale
                        reward += self.reward_manager.step_penalty
                    else:
                        reward = self.reward_manager.collision_penalty
                else:
                    reward = self.reward_manager.step_penalty

            is_done = self.steps >= 500
            self.last_action = action
            return self._get_observation(), reward, is_done, False, {}

        # ── NORMAL STEP ───────────────────────────────────────────────────────
        return super().step(action)
    

    def _handle_pygame_events(self):
        pass


# ─── ARCHITECTURE ────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Dueling DQN Architecture."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.advantage_stream = nn.Linear(128, action_dim)
        self.value_stream = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared_features(state)
        state_value = self.value_stream(features)
        action_advantages = self.advantage_stream(features)
        return state_value + (action_advantages - action_advantages.mean(dim=1, keepdim=True))


# ─── TRAINING FUNCTION ───────────────────────────────────────────────────────

def train():
    env = TrainingEnv(render_mode=render_mode)  # ← TrainingEnv, not WarehouseEnv

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_network = QNetwork(state_dim, action_dim).to(compute_device)
    target_network = QNetwork(state_dim, action_dim).to(compute_device)
    target_network.load_state_dict(policy_network.state_dict())

    optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)
    replay_buffer = deque(maxlen=50000)

    batch_size = 128
    discount_factor = 0.98
    epsilon = 0.7
    epsilon_min = 0.1
    epsilon_decay = 0.998
    target_network_update_frequency = 10
    checkpoint_save_frequency = 50
    total_episodes = 2500

    os.makedirs("checkpoints", exist_ok=True)
    best_delivery_score = -1

    for episode in range(total_episodes):
        current_state, _ = env.reset()
        episode_total_reward = 0
        is_done = False

        should_render_this_episode = (
            RENDER_DURING_TRAINING and (episode % RENDER_EVERY_N_EPISODES == 0)
        )

        while not is_done:
            if random.random() < epsilon:
                if random.random() < 0.7:   # 70% of the time, take heuristic action during exploration 
                    chosen_action = env.heuristic_action()
                else:
                    chosen_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(
                        current_state, dtype=torch.float32, device=compute_device
                    ).unsqueeze(0)
                    chosen_action = policy_network(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated

            if should_render_this_episode:
                env.render()

            replay_buffer.append((current_state, chosen_action, reward, next_state, is_done))
            current_state = next_state
            episode_total_reward += reward

            if len(replay_buffer) > batch_size:
                sampled_batch = random.sample(replay_buffer, batch_size)
                (
                    states_batch, actions_batch, rewards_batch,
                    next_states_batch, dones_batch,
                ) = zip(*sampled_batch)

                states_tensor = torch.as_tensor(np.array(states_batch), dtype=torch.float32, device=compute_device)
                actions_tensor = torch.as_tensor(actions_batch, dtype=torch.long, device=compute_device).unsqueeze(1)
                rewards_tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=compute_device).unsqueeze(1)
                next_states_tensor = torch.as_tensor(np.array(next_states_batch), dtype=torch.float32, device=compute_device)
                dones_tensor = torch.as_tensor(dones_batch, dtype=torch.float32, device=compute_device).unsqueeze(1)

                current_q_values = policy_network(states_tensor).gather(1, actions_tensor)

                with torch.no_grad():
                    best_next_actions = policy_network(next_states_tensor).argmax(dim=1, keepdim=True)
                    next_q_values = target_network(next_states_tensor).gather(1, best_next_actions)
                    expected_q_values = rewards_tensor + (discount_factor * next_q_values * (1 - dones_tensor))

                loss = nn.MSELoss()(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_network_update_frequency == 0:
            target_network.load_state_dict(policy_network.state_dict())

        if env.score > best_delivery_score:
            best_delivery_score = env.score
            torch.save(policy_network.state_dict(), "checkpoints/best_model.pt")
            print(f"  ★ New Best Score: {best_delivery_score}! Model saved.")

        if episode % checkpoint_save_frequency == 0 and episode > 0:
            torch.save(
                {
                    "episode": episode,
                    "policy": policy_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epsilon": epsilon,
                },
                f"checkpoints/model_ep_latest.pt",
            )
            with open(f"checkpoints/buffer_ep_latest.pkl", "wb") as buffer_file:
                pickle.dump(list(replay_buffer), buffer_file)
            print(f"  💾 Checkpoint & buffer saved at episode {episode}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

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