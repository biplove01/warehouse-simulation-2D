import torch
import numpy as np
import time
from warehouse_env import WarehouseEnv
from trainer import QNet  # Importing the architecture we defined

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_policy(model_path="best_model.pt", num_episodes=5, render=True):
    # 1. Setup Environment
    # If you have a render_mode in your WarehouseEnv, "human" lets you watch it.
    env = WarehouseEnv(render_mode="human" if render else None)

    # 2. Load the Model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = QNet(state_dim, action_dim).to(DEVICE)
    try:
        # Load the weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # Handle cases where you might have saved a dict with "policy" key or just the weights
        state_dict = checkpoint.get("policy", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the robot first!")
        return

    # 3. Evaluation Loop
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\n--- Episode {ep + 1} Starting ---")

        while not done:
            # Always pick the BEST action (Greedy)
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = model(state_t).argmax().item()

            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1

            # Optional: Slow down the render so you can actually see the robot move
            if render:
                time.sleep(0.05)

        print(f"Episode Finished | Deliveries: {env.score} | Steps: {steps} | Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    # Change 'best_model.pt' to whatever your filename is
    test_policy(model_path="best_model.pt", num_episodes=5, render=True)