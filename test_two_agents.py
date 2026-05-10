import torch
import pygame
import sys
import numpy as np
from collections import deque

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from model import QNetwork
from constants import *


compute_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─── PER-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class SingleAgentTestingEnv(WarehouseEnv):
    """
    One robot, one active target at a time.
    Targets are assigned externally by the TwoAgentTestingEnv dispatcher —
    this env never spawns its own targets.

    States:
        idle        : robot is at home, no task assigned
        returning   : robot finished a delivery, heading back to home station
        working     : robot has an active target shelf to collect and deliver
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.returning_home = True
        self.is_idle = True          # True when at home with no pending task

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Clear any randomly spawned target from the base reset.
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        # Start navigation target at home.
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map
        self.returning_home = True
        self.is_idle = True

        return obs, info

    def assign_target(self, target_grid_x, target_grid_y):
        """
        Called by the dispatcher when this agent is free and a task is available.
        Places a box on the shelf and sets it as the active navigation target.
        """
        for shelf in self.shelves:
            if self._to_grid_coords(shelf) == (target_grid_x, target_grid_y):
                shelf.has_box = True
                shelf.image = shelf.loaded_image
                break

        self.target_grid_x = target_grid_x
        self.target_grid_y = target_grid_y
        self.target_distance_map = self._bfs_distance_map(
            target_grid_x, target_grid_y
        )
        self.returning_home = False
        self.is_idle = False
        print(f"  ➡️  Agent assigned to shelf ({target_grid_x}, {target_grid_y})")

    def _spawn_new_target(self):
        """Never auto-spawn — dispatcher controls all task assignment."""
        pass

    def _on_delivery(self):
        """After delivery, go home and signal that this agent is now free."""
        self.returning_home = True
        self.is_idle = False   # not idle yet — still navigating home
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map

    @property
    def is_free(self):
        """
        Agent is free (ready for a new task) when it is at the home station.
        Whether it arrived after a delivery or was idle doesn't matter —
        both cases mean the dispatcher can assign it a new shelf.
        """
        robot_at_home = (
            self.robot.grid_x == ROBOT_HOME_GRID_X and
            self.robot.grid_y == ROBOT_HOME_GRID_Y
        )
        return robot_at_home and (self.returning_home or self.is_idle)

    def step(self, action):
        robot = self.robot

        # ── HOME / IDLE PHASE ─────────────────────────────────────────────────
        if self.returning_home or self.is_idle:
            self.steps += 1
            robot_at_home = (
                robot.grid_x == ROBOT_HOME_GRID_X and
                robot.grid_y == ROBOT_HOME_GRID_Y
            )

            if robot_at_home:
                # Mark as fully idle — dispatcher will assign next task.
                self.returning_home = False
                self.is_idle = True
                reward = 0.1 if action == 5 else -0.1
                return self._get_observation(), reward, False, False, {}
            else:
                # Navigate toward home.
                distance_before = self.home_distance_map.get(
                    (robot.grid_x, robot.grid_y), 50
                )
                if action < 4:
                    direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                    delta_x, delta_y = direction_deltas[action]
                    next_x = robot.grid_x + delta_x
                    next_y = robot.grid_y + delta_y
                    is_in_bounds = (
                        0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
                    )
                    is_passable = (next_x, next_y) not in self.obstacle_positions
                    if is_in_bounds and is_passable:
                        robot.grid_x, robot.grid_y = next_x, next_y
                        distance_after = self.home_distance_map.get(
                            (robot.grid_x, robot.grid_y), 50
                        )
                        reward = (
                            (distance_before - distance_after)
                            * self.reward_manager.progress_reward_scale
                        )
                    else:
                        reward = self.reward_manager.collision_penalty
                else:
                    reward = self.reward_manager.step_penalty

                return self._get_observation(), reward, False, False, {}

        # ── WORKING PHASE (active target assigned) ────────────────────────────
        obs, reward, _, _, info = super().step(action)
        return obs, reward, False, False, info   # never terminate in testing

    def _handle_pygame_events(self):
        """Suppress — event handling is done centrally in TwoAgentTestingEnv."""
        pass


# ─── TWO-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class TwoAgentTestingEnv:
    """
    Shared-queue dispatcher for two agents.

    Click a shelf → it goes into the shared target_queue.
    Whichever agent is free at home gets the next task from the queue.
    Both agents run simultaneously each step.
    One pygame window shows both robots.

    Agent 1 (cyan border)  : loaded from checkpoints/best_model.pt  (19-feature obs)
    Agent 2 (yellow border) : loaded from checkpoints_agent2/best_model.pt (23-feature obs)
    """

    AGENT1_OBS_SIZE = 19
    AGENT2_OBS_SIZE = 23   # 19 base + 4 agent1-awareness features

    def __init__(
        self,
        agent1_model_path: str = "checkpoints/best_model.pt",
        agent2_model_path: str = "checkpoints_agent2/best_model.pt",
    ):
        # ── Sub-environments (one pygame screen, owned by agent1_env) ─────────
        self.agent1_env = SingleAgentTestingEnv(render_mode="human")
        self.agent2_env = SingleAgentTestingEnv(render_mode=None)

        # ── Shared task queue ─────────────────────────────────────────────────
        self.target_queue = deque()

        # ── Load Agent 1 policy (frozen, 19-feature obs) ──────────────────────
        action_dim = self.agent1_env.action_space.n   # 6
        self.agent1_policy = QNetwork(self.AGENT1_OBS_SIZE, action_dim).to(compute_device)
        self._load_weights(self.agent1_policy, agent1_model_path, "Agent 1")

        # ── Load Agent 2 policy (frozen, 23-feature obs) ──────────────────────
        self.agent2_policy = QNetwork(self.AGENT2_OBS_SIZE, action_dim).to(compute_device)
        self._load_weights(self.agent2_policy, agent2_model_path, "Agent 2")

        # ── Internal obs state ────────────────────────────────────────────────
        self._agent1_obs = None
        self._agent2_obs = None

    # ── setup helpers ─────────────────────────────────────────────────────────

    def _load_weights(self, network, model_path, label):
        try:
            checkpoint = torch.load(model_path, map_location=compute_device)
            weights = (
                checkpoint.get("policy", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            network.load_state_dict(weights)
            network.eval()
            print(f"  ✅ {label} loaded from '{model_path}'")
        except FileNotFoundError:
            print(f"  ❌ {label}: '{model_path}' not found — using random weights.")

    def reset(self):
        agent1_obs, _ = self.agent1_env.reset()
        agent2_obs, _ = self.agent2_env.reset()

        # Ensure the two robots start on different cells.
        max_retries = 20
        for _ in range(max_retries):
            if (self.agent1_env.robot.grid_x != self.agent2_env.robot.grid_x or
                    self.agent1_env.robot.grid_y != self.agent2_env.robot.grid_y):
                break
            agent2_obs, _ = self.agent2_env.reset()

        self.target_queue.clear()
        self._agent1_obs = agent1_obs
        # Agent 2 obs is extended with Agent 1 features.
        self._agent2_obs = self._build_agent2_obs(agent2_obs)

    # ── shared queue interface ─────────────────────────────────────────────────

    def enqueue_target(self, target_grid_x, target_grid_y):
        """Called by the click handler when a shelf is clicked."""
        if (target_grid_x, target_grid_y) in self.target_queue:
            print(f"  ⚠️  Shelf ({target_grid_x}, {target_grid_y}) already queued.")
            return
        self.target_queue.append((target_grid_x, target_grid_y))
        print(
            f"  📦 Shelf ({target_grid_x}, {target_grid_y}) queued. "
            f"Queue size: {len(self.target_queue)}"
        )

    # ── dispatcher ────────────────────────────────────────────────────────────

    def _dispatch_tasks(self):
        """
        Assign the next queued shelf to whichever free agent is checked first.
        Agent 1 gets priority when both are free simultaneously.
        """
        for agent_env in (self.agent1_env, self.agent2_env):
            if agent_env.is_free and self.target_queue:
                next_target_x, next_target_y = self.target_queue.popleft()
                agent_env.assign_target(next_target_x, next_target_y)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self):
        """
        Steps both agents simultaneously.
        Returns (agent1_reward, agent2_reward) for optional logging.
        """
        # Dispatch before acting so a newly-free agent gets a task this step.
        self._dispatch_tasks()

        # ── Agent 1 action (19-feature obs, standard policy) ──────────────────
        with torch.no_grad():
            agent1_state_tensor = torch.as_tensor(
                self._agent1_obs, dtype=torch.float32, device=compute_device
            ).unsqueeze(0)
            agent1_action = self.agent1_policy(agent1_state_tensor).argmax().item()

        # ── Agent 2 action (23-feature obs, collision-aware policy) ───────────
        with torch.no_grad():
            agent2_state_tensor = torch.as_tensor(
                self._agent2_obs, dtype=torch.float32, device=compute_device
            ).unsqueeze(0)
            agent2_action = self.agent2_policy(agent2_state_tensor).argmax().item()

        # ── Step both envs ────────────────────────────────────────────────────
        agent1_next_obs, agent1_reward, _, _, _ = self.agent1_env.step(agent1_action)
        agent2_next_obs, agent2_reward, _, _, _ = self.agent2_env.step(agent2_action)

        # ── Update stored observations ─────────────────────────────────────────
        self._agent1_obs = agent1_next_obs
        self._agent2_obs = self._build_agent2_obs(agent2_next_obs)

        return agent1_reward, agent2_reward

    # ── Agent 2 observation builder ───────────────────────────────────────────

    def _build_agent2_obs(self, agent2_base_obs):
        """
        Appends the 4 Agent 1 awareness features to Agent 2's base 19-feature
        observation, matching exactly what Agent2TrainingEnv._get_observation()
        produces during training.
        """
        agent2_robot = self.agent2_env.robot
        agent1_robot = self.agent1_env.robot

        agent1_relative_x = (agent1_robot.grid_x - agent2_robot.grid_x) / GRID_WIDTH
        agent1_relative_y = (agent1_robot.grid_y - agent2_robot.grid_y) / GRID_HEIGHT
        agent1_loaded = float(agent1_robot.loaded)
        agent1_returning_home = float(
            self.agent1_env.returning_home or self.agent1_env.is_idle
        )

        extra_features = np.array(
            [agent1_relative_x, agent1_relative_y, agent1_loaded, agent1_returning_home],
            dtype=np.float32,
        )
        return np.concatenate([agent2_base_obs, extra_features])

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        """
        Agent 1's env owns the pygame screen and draws the full scene.
        Agent 2 is drawn on top as an overlay with a yellow border.
        """
        self.agent1_env.render()

        screen = self.agent1_env.screen
        if screen is None:
            return

        # Draw Agent 2 robot on Agent 1's screen.
        agent2_pixel_x = PADDING_BORDER + self.agent2_env.robot.grid_x * GRID_SPACING
        agent2_pixel_y = PADDING_BORDER + self.agent2_env.robot.grid_y * GRID_SPACING

        agent2_image = (
            ROBOT_IMAGE_VERTICAL_BOX
            if self.agent2_env.robot.loaded
            else ROBOT_IMAGE_VERTICAL
        )
        center_offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
        center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
        screen.blit(
            agent2_image,
            (agent2_pixel_x + center_offset_x, agent2_pixel_y + center_offset_y),
        )

        # Yellow border for Agent 2, cyan border for Agent 1 (drawn by agent1_env.render).
        pygame.draw.rect(
            screen,
            (255, 220, 0),
            (agent2_pixel_x, agent2_pixel_y, TILE_SIZE, TILE_SIZE),
            2,
        )

        # Draw queue size as HUD text so you can see pending tasks.
        font = pygame.font.SysFont("monospace", 15)
        queue_label = font.render(
            f"Queue: {len(self.target_queue)}", True, (255, 255, 255)
        )
        screen.blit(queue_label, (8, 8))

        pygame.display.flip()

    # ── event / click handling ────────────────────────────────────────────────

    def handle_events(self):
        """
        Process pygame events. Returns False when the window is closed.
        Translates shelf clicks into enqueue_target() calls.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                clicked_grid_x = round((mouse_x - PADDING_BORDER) / GRID_SPACING)
                clicked_grid_y = round((mouse_y - PADDING_BORDER) / GRID_SPACING)

                # Check if the click lands on a shelf.
                for shelf in self.agent1_env.shelves:
                    shelf_grid_x, shelf_grid_y = self.agent1_env._to_grid_coords(shelf)
                    if (shelf_grid_x, shelf_grid_y) == (clicked_grid_x, clicked_grid_y):
                        self.enqueue_target(clicked_grid_x, clicked_grid_y)
                        break

        return True


# ─── TEST FUNCTION ────────────────────────────────────────────────────────────

def test_two_agents(
    agent1_model_path: str = "checkpoints/best_model.pt",
    agent2_model_path: str = "checkpoints_agent2/best_model.pt",
):
    env = TwoAgentTestingEnv(
        agent1_model_path=agent1_model_path,
        agent2_model_path=agent2_model_path,
    )
    env.reset()
    env.render()

    action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]

    print("\n🖱️  Click shelves to queue tasks. Both robots will pick them up in order.")
    print("    Agent 1 = cyan border  |  Agent 2 = yellow border")
    print("    Close the window to stop.\n")

    step_count = 0
    agent1_total_score = 0
    agent2_total_score = 0

    try:
        while True:
            still_running = env.handle_events()
            if not still_running:
                break

            agent1_reward, agent2_reward = env.step()
            step_count += 1

            # Track deliveries via env scores.
            agent1_total_score = env.agent1_env.score
            agent2_total_score = env.agent2_env.score

            if step_count % 20 == 0:
                print(
                    f"Step {step_count:5d} | "
                    f"A1 score: {agent1_total_score:2d} | "
                    f"A2 score: {agent2_total_score:2d} | "
                    f"Queue: {len(env.target_queue):2d} | "
                    f"A1 pos: ({env.agent1_env.robot.grid_x}, {env.agent1_env.robot.grid_y}) | "
                    f"A2 pos: ({env.agent2_env.robot.grid_x}, {env.agent2_env.robot.grid_y})"
                )

            env.render()

    except KeyboardInterrupt:
        print("\n\nTest ended by user.")
    finally:
        print(
            f"\nFinal — Agent 1 deliveries: {agent1_total_score} | "
            f"Agent 2 deliveries: {agent2_total_score}"
        )
        pygame.quit()


if __name__ == "__main__":
    test_two_agents(
        agent1_model_path="checkpoints/best_model.pt",
        agent2_model_path="checkpoints_agent2/best_model.pt",
    )