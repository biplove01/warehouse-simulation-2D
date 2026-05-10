import torch
from collections import deque
from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from click_controller import ClickController
from train import QNetwork
from constants import *

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── TESTING ENVIRONMENT ─────────────────────────────────────────────────────
class TestingEnv(WarehouseEnv):
    """
    Testing environment with click‑based target queue.
    - No automatic target spawning on reset.
    - No episode termination (infinite horizon).
    - Boxes appear on shelves immediately when clicked.
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.target_queue = deque()
        self.returning_home = True

    def reset(self, seed=None, options=None):
        # Preserve the queue across resets (though we may never reset in testing)
        preserved_queue = self.target_queue.copy() if hasattr(self, "target_queue") else deque()

        # Clear all shelf boxes
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        # Call parent reset but we will override its automatic spawning
        # First, temporarily disable _spawn_new_target by monkey-patching?
        # Simpler: call the base method but then immediately clear any
        # accidentally spawned target and set returning_home=True.
        obs, info = super().reset(seed=seed, options=options)

        # Undo any random target that WarehouseEnv.reset may have created
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        self.target_queue = preserved_queue
        self.returning_home = True
        self.consecutive_wait_steps_at_home = 0   # if needed

        # Restore boxes for all queued shelves
        for (gx, gy) in self.target_queue:
            for shelf in self.shelves:
                if self._to_grid_coords(shelf) == (gx, gy):
                    shelf.has_box = True
                    shelf.image = shelf.loaded_image
                    break

        # Set navigation target to home
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map

        return obs, info

    def enqueue_target(self, grid_x, grid_y):
        """Called by ClickController when a shelf is clicked."""
        if (grid_x, grid_y) in self.target_queue:
            print(f"  ⚠️ Shelf ({grid_x}, {grid_y}) already in queue, ignored.")
            return

        # Place box on shelf immediately
        for shelf in self.shelves:
            if self._to_grid_coords(shelf) == (grid_x, grid_y):
                shelf.has_box = True
                shelf.image = shelf.loaded_image
                break

        self.target_queue.append((grid_x, grid_y))
        print(f"  📦 Box placed on shelf ({grid_x}, {grid_y}). Queue size: {len(self.target_queue)}")

    def _advance_to_next_target(self):
        """Pop next target from queue and set as active navigation target."""
        if self.target_queue:
            next_x, next_y = self.target_queue.popleft()
            self.target_grid_x, self.target_grid_y = next_x, next_y
            self.target_distance_map = self._bfs_distance_map(next_x, next_y)
            self.returning_home = False
            print(f"  ➡️ Now targeting shelf ({next_x}, {next_y})")
        else:
            self._go_home()

    def _go_home(self):
        self.returning_home = True
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map
        print("  🏠 Queue empty, robot returning home.")

    def _on_delivery(self):
        """After delivery, move to next target (or home). Box already cleared on pickup."""
        self._advance_to_next_target()

    def _spawn_new_target(self):
        """Override to do nothing – never spawn random targets in testing."""
        pass

    def step(self, action):
        robot = self.robot

        # ── HOME WAIT PHASE ───────────────────────────────────────────────────
        if self.returning_home:
            self.steps += 1
            robot_at_home = (robot.grid_x == ROBOT_HOME_GRID_X and
                             robot.grid_y == ROBOT_HOME_GRID_Y)

            if robot_at_home:
                if self.target_queue:
                    # There are pending targets – stop waiting and go to the next one
                    self._advance_to_next_target()
                    # After advancing, return a neutral observation and reward.
                    return self._get_observation(), 0.0, False, False, {}
                else:
                    # No targets – reward waiting, penalise non‑wait actions
                    if action == 5:
                        reward = 0.5
                    else:
                        reward = -0.5
                    # No episode termination
                    return self._get_observation(), reward, False, False, {}
            else:
                # Not home yet – navigate home
                dist_before = self.home_distance_map.get((robot.grid_x, robot.grid_y), 50)
                if action < 4:
                    dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
                    nx, ny = robot.grid_x + dx, robot.grid_y + dy
                    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                        (nx, ny) not in self.obstacle_positions):
                        robot.grid_x, robot.grid_y = nx, ny
                        dist_after = self.home_distance_map.get((robot.grid_x, robot.grid_y), 50)
                        reward = (dist_before - dist_after) * self.reward_manager.progress_reward_scale
                    else:
                        reward = self.reward_manager.collision_penalty
                else:
                    reward = self.reward_manager.step_penalty

                return self._get_observation(), reward, False, False, {}

        # ── NORMAL STEP (robot has an active target shelf) ─────────────────────
        # Use parent step but override termination to always False
        obs, reward, _, _, info = super().step(action)
        # Never terminate
        return obs, reward, False, False, info


# ─── TEST FUNCTION ───────────────────────────────────────────────────────────
def test_policy(model_path="checkpoints/best_model.pt", render=True):
    env = TestingEnv(render_mode="human" if render else None)
    click_controller = ClickController(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_network = QNetwork(state_dim, action_dim).to(compute_device)

    try:
        checkpoint = torch.load(model_path, map_location=compute_device)
        model_weights = checkpoint.get("policy", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        policy_network.load_state_dict(model_weights)
        policy_network.eval()
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the robot first!")
        return

    # Initial reset – no random target, robot starts at home
    current_state, _ = env.reset()
    if render:
        env.render()

    # Action name mapping for display
    action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]

    print("\n🖱️  Click on shelves to queue targets. Robot will deliver them in order.")
    print("    Close the window to stop.\n")

    step_count = 0
    try:
        while True:
            if render:
                click_controller.handle_pygame_events()

            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    current_state, dtype=torch.float32, device=compute_device
                ).unsqueeze(0)
                action = policy_network(state_tensor).argmax().item()

            current_state, reward, done, truncated, _ = env.step(action)
            step_count += 1

            # Only print occasionally to avoid spam – or always if you like
            if step_count % 10 == 0:
                print(f"Step {step_count:5d} | Action: {action_names[action]:8s} | "
                      f"Reward: {reward:6.2f} | Loaded: {env.robot.loaded} | "
                      f"Pos: ({env.robot.grid_x}, {env.robot.grid_y}) | "
                      f"Queue: {len(env.target_queue)}")

            if render:
                env.render()
    except KeyboardInterrupt:
        print("\n\nTest ended by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    test_policy(model_path="checkpoints/best_model.pt", render=True)