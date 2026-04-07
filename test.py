import torch
import numpy as np
import time
from warehouse_env import WarehouseEnv
from trainer import QNetwork

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_policy(model_path="checkpoints/model_ep_latest.pt", num_episodes=5, render=True):

    # 1. Setup environment
    env = WarehouseEnv(render_mode="human" if render else None)

    # 2. Load the trained model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_network = QNetwork(state_dim, action_dim).to(compute_device)

    try:
        checkpoint = torch.load(model_path, map_location=compute_device)

        # Handle both raw state dicts and checkpoint dicts saved with a "policy" key
        if isinstance(checkpoint, dict):                            
            model_weights = checkpoint.get("policy", checkpoint)    
        else:
            model_weights = checkpoint

        policy_network.load_state_dict(model_weights)
        policy_network.eval()
        print(f"Successfully loaded model from {model_path}")

    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the robot first!")
        return

    # 3. Evaluation loop
    for episode in range(num_episodes):
        current_state, _ = env.reset()
        episode_total_reward = 0
        is_done = False
        step_count = 0

        print(f"\n--- Episode {episode + 1} Starting ---")

        while not is_done:
            # Always pick the best action greedily (no exploration)
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    current_state, dtype=torch.float32, device=compute_device
                ).unsqueeze(0)
                chosen_action = policy_network(state_tensor).argmax().item()

            current_state, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated
            episode_total_reward += reward
            step_count += 1

             # Log what the robot just did
            action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]
            print(
                f"  Step {step_count:3d} | "
                f"Action: {action_names[chosen_action]:8s} | "
                f"Reward: {reward:6.2f} | "
                f"Loaded: {env.robot.loaded} | "
                f"Pos: ({env.robot.grid_x}, {env.robot.grid_y}) | "
                f"Score: {env.score}"
            )

            if render:
                env.render()

        print(
            f"Episode Finished | "
            f"Deliveries: {env.score} | "
            f"Steps: {step_count} | "
            f"Total Reward: {episode_total_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    test_policy(model_path="checkpoints/best_model.pt", num_episodes=5, render=True)