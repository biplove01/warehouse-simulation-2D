import torch
import torch.nn as nn
import numpy as np
import pygame
from unified_env import TwoAgentWarehouseEnv
from constants import PADDING_BORDER, GRID_SPACING

# Import Kafka elements from reader.py
from reader import OrderKafkaReader, KafkaConfig

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
    # Initialize the environment
    env = TwoAgentWarehouseEnv(render_mode="human")

    # Disable automatic spawning of random boxes
    env._spawn_target = lambda: None

    # ---------------------------------------------------------------------------
    # Kafka Integration & Callback Setup
    # ---------------------------------------------------------------------------
    def kafka_order_callback(coordinates):
        """This function runs on the Kafka thread whenever a message is parsed."""
        for gx, gy in coordinates:
            for shelf in env.shelves:
                sgx, sgy = env._get_grid_coords(shelf)
                if sgx == gx and sgy == gy:
                    if not shelf.has_box:
                        shelf.has_box = True
                        if hasattr(shelf, 'loaded_image'):
                            shelf.image = shelf.loaded_image
                        
                        # Add to the unified environment queue safely
                        if (gx, gy) not in env.queue:
                            env.queue.append((gx, gy))
                            print(f"[Kafka Spawn] Box placed at ({gx}, {gy}). Current Queue Size: {len(env.queue)}")
                    break

    # Hardcoded shelf map matching your main block in reader.py
    item_shelf_map = {
        1:  (1,  2),  2:  (2,  6),  3:  (5,  0),  4:  (7,  4),  5:  (9,  8),
        6:  (11, 3),  7:  (13, 7),  8:  (15, 12), 9:  (16, 0),  10: (14, 11),
        11: (19, 2),  12: (20, 5),  13: (19, 8),  14: (20, 11), 15: (10, 12),
    }

    config = KafkaConfig()
    kafka_reader = OrderKafkaReader(config, item_shelf_map, on_order_callback=kafka_order_callback)
    kafka_reader.start()
    # ---------------------------------------------------------------------------

    # Intercept pygame events for manual mouse clicks (Optional backup mechanism)
    original_event_get = pygame.event.get
    
    def custom_event_get(*args, **kwargs):
        events = original_event_get(*args, **kwargs)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                gx = round((mx - PADDING_BORDER) / GRID_SPACING)
                gy = round((my - PADDING_BORDER) / GRID_SPACING)
                for shelf in env.shelves:
                    sgx, sgy = env._get_grid_coords(shelf)
                    if sgx == gx and sgy == gy:
                        if not shelf.has_box:
                            shelf.has_box = True
                            if hasattr(shelf, 'loaded_image'):
                                shelf.image = shelf.loaded_image
                            if (gx, gy) not in env.queue:
                                env.queue.append((gx, gy))
                                print(f"[Manual Spawn] Box placed at ({gx}, {gy})")
                        break
        return events

    pygame.event.get = custom_event_get

    observation_dim = env.observation_space.shape[0]
    total_actions = env.action_space.n

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    policy_net = DeepQNetwork(observation_dim, total_actions).to(compute_device)
    model_path = "checkpoints/model_episode.pth"

    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=compute_device))
        policy_net.eval()
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Run train.py first.")
        pygame.event.get = original_event_get
        kafka_reader.stop()
        return

    # Run the evaluation loop
    state, info = env.reset()
    action_mask = info["action_mask"]

    done = False
    truncated = False

    print("\nStarting evaluation...")
    print("Kafka Stream Active! (You can also click on shelves to manually spawn boxes.)")

    try:
        while not (done or truncated):
            pygame.time.delay(100)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(compute_device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)[0]
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=compute_device)
                q_values[~mask_tensor] = -float('inf')
                best_action = int(q_values.argmax().item())

            # env.step internally runs self._dispatch() which pulls items from the queue updated by Kafka
            state, reward, done, truncated, info = env.step(best_action)
            action_mask = info["action_mask"]

    except KeyboardInterrupt:
        print("\nTesting manually interrupted.")
    finally:
        # Restore clean state and stop background workers
        pygame.event.get = original_event_get
        print("Stopping Kafka Consumer Thread...")
        kafka_reader.stop()
        env.close()
        
if __name__ == "__main__":
    run_testing()