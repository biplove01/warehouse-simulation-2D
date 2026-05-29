import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from collections import deque
from unified_env import TwoAgentWarehouseEnv


class DeepQNetwork(nn.Module):
    def __init__(self, observation_dimension, total_actions):
        super(DeepQNetwork, self).__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(observation_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_actions)
        )

    def forward(self, state_tensor):
        return self.network_layers(state_tensor)


def select_masked_action(evaluation_model, current_state, current_epsilon, action_mask, recommended_bfs_action):
    if random.random() < current_epsilon:
        if random.random() >= 0.30 and action_mask[recommended_bfs_action] == 1.0:
            return recommended_bfs_action
        available_valid_actions = np.where(action_mask == 1.0)[0]
        return random.choice(available_valid_actions) if len(available_valid_actions) > 0 else 5

    state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
    with torch.no_grad():
        predicted_q_values = evaluation_model(state_tensor).cpu().numpy()[0]

    predicted_q_values[action_mask == 0.0] = -float('inf')
    return int(np.argmax(predicted_q_values))


def execute_training_loop():
    # env = TwoAgentWarehouseEnv(render_mode=None)
    env = TwoAgentWarehouseEnv(render_mode="human")
    observation_dim = env.observation_space.shape[0]
    total_actions = env.action_space.n

    policy_net = DeepQNetwork(observation_dim, total_actions)
    target_net = DeepQNetwork(observation_dim, total_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer = deque(maxlen=50000)

    epsilon = 0.7
    epsilon_decay = 0.999
    epsilon_min = 0.05
    batch_size = 128
    gamma = 0.99

    total_episodes = 2000
    checkpoint_directory = "checkpoints"
    os.makedirs(checkpoint_directory, exist_ok=True)

    highest_reward = -9999.0
    best_score_achieved = 0

    try:
        for episode in range(total_episodes):
            should_render = (episode % 20 == 0)
            if should_render:
                env.close()
                env = TwoAgentWarehouseEnv(render_mode=None)
            else:
                if env.render_mode == "human":
                    env.close()
                    env = TwoAgentWarehouseEnv(render_mode=None)

            state, info = env.reset()
            action_mask = info["action_mask"]
            recommended_bfs_action = info["bfs_action"]

            total_r2_reward = 0.0
            final_r1_score = 0
            final_r2_score = 0
            final_collisions = 0

            while True:
                action = select_masked_action(policy_net, state, epsilon, action_mask, recommended_bfs_action)

                next_state, reward, terminated, truncated, next_info = env.step(action)
                next_action_mask = next_info["action_mask"]
                next_bfs_action = next_info["bfs_action"]

                final_r1_score = next_info["r1_score"]
                final_r2_score = next_info["r2_score"]
                final_collisions = next_info["collisions"]

                buffer.append((
                    state, action, reward, next_state,
                    float(terminated or truncated), action_mask, next_action_mask
                ))

                state = next_state
                action_mask = next_action_mask
                recommended_bfs_action = next_bfs_action
                total_r2_reward += reward

                if len(buffer) > batch_size:
                    minibatch = random.sample(buffer, batch_size)
                    b_s, b_a, b_r, b_ns, b_d, b_am, b_nam = zip(*minibatch)

                    t_s = torch.FloatTensor(np.array(b_s))
                    t_a = torch.LongTensor(np.array(b_a)).unsqueeze(1)
                    t_r = torch.FloatTensor(np.array(b_r)).unsqueeze(1)
                    t_ns = torch.FloatTensor(np.array(b_ns))
                    t_d = torch.FloatTensor(np.array(b_d)).unsqueeze(1)
                    t_nam = torch.FloatTensor(np.array(b_nam))

                    current_q = policy_net(t_s).gather(1, t_a)

                    with torch.no_grad():
                        next_q = target_net(t_ns)
                        next_q[t_nam == 0.0] = -float('inf')
                        max_next_q = next_q.max(1)[0].unsqueeze(1)
                        target_q = t_r + (gamma * max_next_q * (1 - t_d))

                    loss = nn.MSELoss()(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()

                if terminated or truncated:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if final_r2_score > best_score_achieved:
                best_score_achieved = final_r2_score

            if episode % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"Ep {episode:4d} | Eps: {epsilon:.2f} | R1 Score (Deliveries): {final_r1_score:2d} | "
                      f"R2 Score (Deliveries): {final_r2_score:2d} | Best R2 Score: {best_score_achieved:2d} | "
                      f"Collisions: {final_collisions:2d} | R2 Reward: {total_r2_reward:.2f}")

                torch.save(policy_net.state_dict(), os.path.join(checkpoint_directory, "latest_model.pth"))
                with open(os.path.join(checkpoint_directory, "replay_buffer.pkl"), "wb") as file_out:
                    pickle.dump(buffer, file_out)

                if total_r2_reward > highest_reward:
                    highest_reward = total_r2_reward
                    torch.save(policy_net.state_dict(), os.path.join(checkpoint_directory, "best_model.pth"))

            if (episode + 1) % 200 == 0:
                periodic_model_path = os.path.join(checkpoint_directory, f"model_episode.pth")
                torch.save(policy_net.state_dict(), periodic_model_path)
                print(f"Saved periodic checkpoint: {periodic_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")

    finally:
        final_model_path = os.path.join(checkpoint_directory, "model_final_exit.pth")
        torch.save(policy_net.state_dict(), final_model_path)
        print(f"Saved final model to: {final_model_path}")
        env.close()


if __name__ == "__main__":
    execute_training_loop()