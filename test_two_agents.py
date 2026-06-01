import torch
import torch.nn as nn
import numpy as np
import pygame
from unified_env import TwoAgentWarehouseEnv
from constants import PADDING_BORDER, GRID_SPACING


# Same architecture as train.py
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


def run_testing():
    # 1. Initialize the EXACT SAME environment used in training
    env = TwoAgentWarehouseEnv(render_mode="human")

    # MONKEY PATCH: Disable automatic spawning of boxes
    env._spawn_target = lambda: None

    # MONKEY PATCH: Intercept pygame events to handle manual mouse clicks for spawning
    original_event_get = pygame.event.get
    
    def custom_event_get(*args, **kwargs):
        events = original_event_get(*args, **kwargs)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                gx = round((mx - PADDING_BORDER) / GRID_SPACING)
                gy = round((my - PADDING_BORDER) / GRID_SPACING)
                for shelf in env.shelves:
                    sgx, sgy = env._gc(shelf)
                    if sgx == gx and sgy == gy:
                        if not shelf.has_box:
                            shelf.has_box = True
                            if hasattr(shelf, 'loaded_image'):
                                shelf.image = shelf.loaded_image
                            if (gx, gy) not in env.queue:
                                env.queue.append((gx, gy))
                                print(f"Manual Spawn: Box placed at ({gx}, {gy})")
                        break
        return events

    pygame.event.get = custom_event_get

    observation_dim = env.observation_space.shape[0]
    total_actions = env.action_space.n

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load the trained PyTorch Model
    policy_net = DeepQNetwork(observation_dim, total_actions).to(compute_device)
    model_path = "checkpoints/model_episode.pth"

    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=compute_device))
        policy_net.eval()
        print(f"✅ Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}. Please run train.py first.")
        # Restore monkey patch before exiting
        pygame.event.get = original_event_get
        return

    # 3. Run the Evaluation Loop
    state, info = env.reset()
    action_mask = info["action_mask"]

    done = False
    truncated = False

    print("\nStarting evaluation... Press Ctrl+C in terminal to stop.")
    print("-> MANUAL MODE ENABLED: Click on any shelf to spawn a box.")

    try:
        while not (done or truncated):
            # Slow down the render for visual debugging
            pygame.time.delay(100)

            # Formulate state tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(compute_device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)[0]

                # APPLY THE ACTION MASK IN TESTING (Crucial Step)
                # Convert the [1.0, 0.0] mask to boolean and apply -infinity to invalid actions
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=compute_device)
                q_values[~mask_tensor] = -float('inf')

                # Select the best VALID action
                best_action = int(q_values.argmax().item())

            # Step the environment
            state, reward, done, truncated, info = env.step(best_action)
            action_mask = info["action_mask"]

            # (Optional) Print the steps to monitor what the network is choosing
            action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]
            # print(f"Chosen Action: {action_names[best_action]:<8} | Reward: {reward:.1f}")

    except KeyboardInterrupt:
        print("\nTesting manually interrupted.")
    finally:
        # Restore monkey patch
        pygame.event.get = original_event_get
        env.close()


if __name__ == "__main__":
    run_testing()