import torch
from collections import deque
from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from click_controller import ClickController
from trainer import QNetwork
from constants import *

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── TESTING ENVIRONMENT ─────────────────────────────────────────────────────

class TestingEnv(WarehouseEnv):
    """
    Extends WarehouseEnv with testing-specific behavior:
    - Target queue filled by mouse clicks
    - No random spawning — only user-specified targets
    - Robot returns home and waits when queue is empty
    """

    def reset(self, seed=None, options=None):
        # Preserve the queue across episode resets so clicks aren't lost
        preserved_queue = self.target_queue if hasattr(self, "target_queue") else deque()
        observation, info = super().reset(seed=seed, options=options)
        self.target_queue = preserved_queue
        self.returning_home = True  # always start by going home
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map
        return observation, info

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.target_queue = deque()
        self.returning_home = True

    def enqueue_target(self, grid_x, grid_y):
        """Called by ClickController when user clicks a shelf."""
        self.target_queue.append((grid_x, grid_y))
        print(f"  📦 Shelf ({grid_x}, {grid_y}) added to queue. Queue size: {len(self.target_queue)}")

    def _spawn_new_target(self):
        """
        Pops next target from queue.
        If queue is empty, sends robot home to wait for clicks.
        No random spawning in testing mode.
        """
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        if self.target_queue:
            next_target_grid_x, next_target_grid_y = self.target_queue.popleft()
            target_shelf = next(
                (s for s in self.shelves
                 if self._to_grid_coords(s) == (next_target_grid_x, next_target_grid_y)),
                None
            )
            if target_shelf is None:
                self._go_home()
                return
            target_shelf.has_box = True
            target_shelf.image = target_shelf.loaded_image
            self.target_grid_x, self.target_grid_y = self._to_grid_coords(target_shelf)
            self.target_distance_map = self._bfs_distance_map(
                self.target_grid_x, self.target_grid_y
            )
            self.returning_home = False
        else:
            self._go_home()


    def _go_home(self):
        """Sends robot back to home station to wait for next click."""
        self.returning_home = True
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map
        print("  🏠 Queue empty, robot returning to home station.")

    def _on_delivery(self):
        """After delivery, pop next target from queue or go home."""
        self._spawn_new_target()

    def step(self, action):
        robot = self.robot

        # ── HOME WAIT PHASE ───────────────────────────────────────────────────
        if self.returning_home:
            self.steps += 1
            robot_at_home = (
                robot.grid_x == ROBOT_HOME_GRID_X and
                robot.grid_y == ROBOT_HOME_GRID_Y
            )

            if robot_at_home:
                # Check if a new target has been queued while waiting
                if self.target_queue:
                    self._spawn_new_target()
                # Stay idle at home — force wait action regardless of policy
                reward = 0.0
            else:
                # Navigate home using BFS
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
                    reward = (distance_before - distance_after) * self.reward_manager.progress_reward_scale
                else:
                    reward = self.reward_manager.step_penalty

            is_done = self.steps >= 500
            self.last_action = action
            return self._get_observation(), reward, is_done, False, {}

        # ── NORMAL STEP ───────────────────────────────────────────────────────
        return super().step(action)


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

    if render:
        env.reset()
        env.render()

    print("\n🖱️  Click on shelves to queue targets. Robot will deliver them in order.")
    print("    Close the window to stop.\n")

    episode = 0
    while True:
        current_state, _ = env.reset()
        is_done = False
        step_count = 0
        episode += 1
        print(f"\n--- Episode {episode} Starting ---")

        while not is_done:
            if render:
                click_controller.handle_pygame_events()

            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    current_state, dtype=torch.float32, device=compute_device
                ).unsqueeze(0)
                chosen_action = policy_network(state_tensor).argmax().item()

            current_state, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated
            step_count += 1

            action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]
            print(
                f"  Step {step_count:3d} | "
                f"Action: {action_names[chosen_action]:8s} | "
                f"Reward: {reward:6.2f} | "
                f"Loaded: {env.robot.loaded} | "
                f"Pos: ({env.robot.grid_x}, {env.robot.grid_y}) | "
                f"Score: {env.score} | "
                f"Queue: {len(env.target_queue)}"
            )

            if render:
                env.render()

        print(
            f"Episode Finished | "
            f"Deliveries: {env.score} | "
            f"Steps: {step_count}"
        )


if __name__ == "__main__":
    test_policy(model_path="checkpoints/best_model.pt", render=True)