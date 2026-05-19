import torch
import pygame
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

# ─── SPEED CONTROL ────────────────────────────────────────────────────────────
# Dial this down to slow the visualisation. 6 is comfortable to watch.
# Raise toward 20 to run faster. Press +/- at runtime to adjust live.
SIMULATION_FPS = 6

# ─── AGENT 2 HOME STATION ─────────────────────────────────────────────────────
# Agent 1 rests at (ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y) = (2, 0).
# Agent 2 rests at (0, 0) — the top-left charge station, guaranteed distinct.
AGENT2_HOME_GRID_X = 0
AGENT2_HOME_GRID_Y = 0

# Obs sizes must match what the policies were trained with.
AGENT1_OBS_SIZE = 19
AGENT2_OBS_SIZE = 25   # 19 base + 6 agent1-awareness features (from two_agent_warehouse_env.py)


# ─── PER-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class SingleAgentTestingEnv(WarehouseEnv):
    """
    One robot, one active target at a time.
    Never spawns its own targets — the dispatcher assigns them.
    Never calls pygame.display.flip() — the central render() does one flip
    per frame after drawing everything, which eliminates flickering.

    home_grid_x / home_grid_y are set per-agent so Agent 1 and Agent 2
    rest at completely different charge station cells.
    """

    def __init__(self, home_grid_x: int, home_grid_y: int, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.home_grid_x = home_grid_x
        self.home_grid_y = home_grid_y

        # Build a BFS distance map to this agent's own home station.
        self.own_home_distance_map = self._bfs_distance_map(
            home_grid_x, home_grid_y, target_is_walkable=True
        )

        self.returning_home = True
        self.is_idle = True

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Clear any randomly spawned shelf box from base reset.
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        # Start at this agent's own home station.
        self.robot.grid_x = self.home_grid_x
        self.robot.grid_y = self.home_grid_y
        self.robot.loaded = False

        self.target_grid_x = self.home_grid_x
        self.target_grid_y = self.home_grid_y
        self.target_distance_map = self.own_home_distance_map
        self.returning_home = True
        self.is_idle = True

        return obs, info

    @property
    def is_free(self):
        """
        Agent is free (ready for a new task) when it is idle — meaning it just
        finished a delivery and is waiting for the next assignment wherever it
        currently stands. No home-return trip required.
        """
        return self.is_idle

    def assign_target(self, target_grid_x, target_grid_y):
        """
        Dispatcher calls this when a task is available and agent is free.
        The box is already on the shelf — enqueue_target() placed it on click.
        We only update the navigation target here.
        """
        self.target_grid_x = target_grid_x
        self.target_grid_y = target_grid_y
        self.target_distance_map = self._bfs_distance_map(
            target_grid_x, target_grid_y
        )
        self.returning_home = False
        self.is_idle = False
        print(f"  ➡️  Agent ({self.home_grid_x},{self.home_grid_y}) → shelf ({target_grid_x}, {target_grid_y})")

    def _spawn_new_target(self):
        """Never auto-spawn — dispatcher controls all task assignment."""
        pass

    def _on_delivery(self):
        """
        After delivery: do NOT go home. Go idle in place so the dispatcher
        can immediately assign the next queued task this same step.
        The robot stays exactly where it is (adjacent to the dropoff platform)
        and the dispatcher hands it the next shelf within one step cycle.
        Only if the queue is empty will it remain idle until a new task arrives.
        """
        self.returning_home = False
        self.is_idle = True
        # Point nav target at dropoff so the observation stays coherent
        # while idle — avoids stale shelf coordinates in the feature vector.
        self.target_grid_x = self.dropoff_grid_x
        self.target_grid_y = self.dropoff_grid_y

    def step(self, action):
        robot = self.robot

        # ── IDLE PHASE — robot is waiting for next task assignment ────────────
        # Robot stays exactly where it is. The dispatcher will assign the next
        # task at the top of the next TwoAgentTestingEnv.step() call.
        if self.is_idle:
            self.steps += 1
            return self._get_observation(), 0.0, False, False, {}

        # ── HOME-RETURN PHASE — only used on initial startup ──────────────────
        # After the very first reset both robots start at their home stations
        # and returning_home=True. This phase navigates them there if for some
        # reason they are not already there (e.g. after env.reset() mid-session).
        if self.returning_home:
            self.steps += 1
            robot_at_own_home = (
                robot.grid_x == self.home_grid_x and
                robot.grid_y == self.home_grid_y
            )
            if robot_at_own_home:
                self.returning_home = False
                self.is_idle = True
                return self._get_observation(), 0.0, False, False, {}

            distance_before = self.own_home_distance_map.get(
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
                    distance_after = self.own_home_distance_map.get(
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

        # ── WORKING PHASE ─────────────────────────────────────────────────────
        obs, reward, _, _, info = super().step(action)
        return obs, reward, False, False, info   # never terminate in testing

    def _handle_pygame_events(self):
        """
        Suppress base class event handling and display.flip().
        All events and the single flip are handled centrally in
        TwoAgentTestingEnv to eliminate flickering.
        """
        pass


# ─── TWO-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class TwoAgentTestingEnv:
    """
    Shared-queue dispatcher. One pygame window, two robots, zero flicker.

    Flicker fix
    -----------
    agent1_env.render() in the base class calls pygame.display.flip() at the
    end. If we then draw Agent 2 and the HUD on top and call flip() again, the
    screen alternates between two incomplete frames every tick — that's the
    flicker. The fix: override _handle_pygame_events() in SingleAgentTestingEnv
    to suppress the base render entirely, and do all drawing here in one method
    that ends with exactly one pygame.display.flip() per frame.

    Speed control
    -------------
    The clock.tick() call that was buried inside WarehouseEnv.render() is now
    called once per frame from the test loop using the live `self.fps` value.
    Press + / = to speed up, - to slow down at runtime.

    Home stations
    -------------
    Agent 1 → (ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y) = (2, 0)
    Agent 2 → (AGENT2_HOME_GRID_X, AGENT2_HOME_GRID_Y) = (0, 0)
    These are guaranteed distinct cells so the robots never share a station.
    """

    def __init__(
        self,
        agent1_model_path: str = "checkpoints/best_model.pt",
        agent2_model_path: str = "checkpoints_agent2/best_model.pt",
    ):
        # Agent 1 env — owns nothing, just its robot state.
        self.agent1_env = SingleAgentTestingEnv(
            home_grid_x=ROBOT_HOME_GRID_X,
            home_grid_y=ROBOT_HOME_GRID_Y,
            render_mode=None,   # rendering is done centrally
        )

        # Agent 2 env — also owns nothing visual.
        self.agent2_env = SingleAgentTestingEnv(
            home_grid_x=AGENT2_HOME_GRID_X,
            home_grid_y=AGENT2_HOME_GRID_Y,
            render_mode=None,
        )

        # Shared task queue.
        self.target_queue = deque()

        # Load policies.
        action_dim = self.agent1_env.action_space.n   # 6
        self.agent1_policy = QNetwork(AGENT1_OBS_SIZE, action_dim).to(compute_device)
        self._load_weights(self.agent1_policy, agent1_model_path, "Agent 1")

        self.agent2_policy = QNetwork(AGENT2_OBS_SIZE, action_dim).to(compute_device)
        self._load_weights(self.agent2_policy, agent2_model_path, "Agent 2")

        # Stored observations.
        self._agent1_obs = None
        self._agent2_obs = None

        # Pygame — owned centrally, not by any sub-env.
        self.screen = None
        self.clock = None
        self.fps = SIMULATION_FPS

        # Pre-bake HUD font once — creating it every frame causes flicker.
        self._hud_font = None

    # ── weight loader ─────────────────────────────────────────────────────────

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

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        agent1_obs, _ = self.agent1_env.reset()
        agent2_obs, _ = self.agent2_env.reset()

        self.target_queue.clear()
        self._agent1_obs = agent1_obs
        self._agent2_obs = self._build_agent2_obs(agent2_obs)

    # ── shared queue ──────────────────────────────────────────────────────────

    def enqueue_target(self, target_grid_x, target_grid_y):
        if (target_grid_x, target_grid_y) in self.target_queue:
            print(f"  ⚠️  Shelf ({target_grid_x}, {target_grid_y}) already queued.")
            return

        # Place box on the shelf immediately on click so it appears on screen
        # at once — independently of when a robot actually claims the task.
        # Both sub-envs have their own shelf object list, so we update both.
        for shelf in self.agent1_env.shelves:
            if self.agent1_env._to_grid_coords(shelf) == (target_grid_x, target_grid_y):
                shelf.has_box = True
                shelf.image = shelf.loaded_image
                break
        for shelf in self.agent2_env.shelves:
            if self.agent2_env._to_grid_coords(shelf) == (target_grid_x, target_grid_y):
                shelf.has_box = True
                shelf.image = shelf.loaded_image
                break

        self.target_queue.append((target_grid_x, target_grid_y))
        print(
            f"  📦 Shelf ({target_grid_x}, {target_grid_y}) queued. "
            f"Queue: {len(self.target_queue)}"
        )

    # ── dispatcher ────────────────────────────────────────────────────────────

    def _dispatch_tasks(self):
        """
        Assign the next queued shelf to whichever free agent is checked first.
        Agent 1 gets priority when both are simultaneously free.
        The occupancy check ensures a free robot never navigates to the other
        robot's current cell — if the target cell is occupied by the other
        robot right now, we skip this dispatch cycle and retry next step.
        """
        for agent_env, other_env in (
            (self.agent1_env, self.agent2_env),
            (self.agent2_env, self.agent1_env),
        ):
            if agent_env.is_free and self.target_queue:
                candidate_x, candidate_y = self.target_queue[0]
                # Don't dispatch if the other robot is currently sitting on
                # the target shelf (would cause immediate collision on pickup).
                other_at_target = (
                    other_env.robot.grid_x == candidate_x and
                    other_env.robot.grid_y == candidate_y
                )
                if not other_at_target:
                    self.target_queue.popleft()
                    agent_env.assign_target(candidate_x, candidate_y)

    # ── collision prediction ───────────────────────────────────────────────────

    def _predict_next_position(self, robot, action, obstacle_positions):
        """Where will this robot land after action? Read-only, no side effects."""
        if action >= 4:
            return robot.grid_x, robot.grid_y
        direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        delta_x, delta_y = direction_deltas[action]
        next_x = robot.grid_x + delta_x
        next_y = robot.grid_y + delta_y
        is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
        is_passable = (next_x, next_y) not in obstacle_positions
        if is_in_bounds and is_passable:
            return next_x, next_y
        return robot.grid_x, robot.grid_y

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self):
        """
        Steps both agents simultaneously. Agent 1 always moves freely.
        Agent 2 is forced to WAIT if its next cell would equal Agent 1's
        current cell OR Agent 1's next cell (prevents passing through).
        Returns (agent1_reward, agent2_reward).
        """
        self._dispatch_tasks()

        # Select actions via frozen policies.
        with torch.no_grad():
            agent1_action = self.agent1_policy(
                torch.as_tensor(
                    self._agent1_obs, dtype=torch.float32, device=compute_device
                ).unsqueeze(0)
            ).argmax().item()

            agent2_action = self.agent2_policy(
                torch.as_tensor(
                    self._agent2_obs, dtype=torch.float32, device=compute_device
                ).unsqueeze(0)
            ).argmax().item()

        # Predict where each robot will land.
        agent1_next_x, agent1_next_y = self._predict_next_position(
            self.agent1_env.robot, agent1_action,
            self.agent1_env.obstacle_positions,
        )
        agent2_next_x, agent2_next_y = self._predict_next_position(
            self.agent2_env.robot, agent2_action,
            self.agent2_env.obstacle_positions,
        )

        # Agent 2 must yield if it would move onto Agent 1's current cell
        # OR Agent 1's next cell (covers the pass-through case).
        agent2_would_collide = (
            agent2_action < 4
        ) and (
            (agent2_next_x == self.agent1_env.robot.grid_x and
             agent2_next_y == self.agent1_env.robot.grid_y)
            or
            (agent2_next_x == agent1_next_x and
             agent2_next_y == agent1_next_y)
        )

        effective_agent2_action = 5 if agent2_would_collide else agent2_action

        # Step both agents.
        agent1_next_obs, agent1_reward, _, _, _ = self.agent1_env.step(agent1_action)
        agent2_next_obs, agent2_reward, _, _, _ = self.agent2_env.step(effective_agent2_action)

        if agent2_would_collide:
            agent2_reward += -20.0   # collision penalty on top of forced wait

        # Rebuild observations.
        self._agent1_obs = agent1_next_obs
        self._agent2_obs = self._build_agent2_obs(agent2_next_obs)

        return agent1_reward, agent2_reward

    # ── Agent 2 observation builder ───────────────────────────────────────────

    def _build_agent2_obs(self, agent2_base_obs):
        """
        Appends 6 Agent 1 awareness features to Agent 2's base 19-feature obs.
        Must match Agent2TrainingEnv._get_observation() in two_agent_warehouse_env.py.
        """
        agent2_robot = self.agent2_env.robot
        agent1_robot = self.agent1_env.robot

        agent1_relative_x = (agent1_robot.grid_x - agent2_robot.grid_x) / GRID_WIDTH
        agent1_relative_y = (agent1_robot.grid_y - agent2_robot.grid_y) / GRID_HEIGHT
        agent1_loaded = float(agent1_robot.loaded)
        agent1_returning = float(
            self.agent1_env.returning_home or self.agent1_env.is_idle
        )
        agent1_target_relative_x = (
            self.agent1_env.target_grid_x - agent2_robot.grid_x
        ) / GRID_WIDTH
        agent1_target_relative_y = (
            self.agent1_env.target_grid_y - agent2_robot.grid_y
        ) / GRID_HEIGHT

        extra_features = np.array([
            agent1_relative_x,
            agent1_relative_y,
            agent1_loaded,
            agent1_returning,
            agent1_target_relative_x,
            agent1_target_relative_y,
        ], dtype=np.float32)

        return np.concatenate([agent2_base_obs, extra_features])

    # ── render ────────────────────────────────────────────────────────────────

    def _ensure_screen(self):
        """Initialise pygame and the screen the first time render() is called."""
        if self.screen is not None:
            return
        pygame.init()
        window_width = GRID_WIDTH * GRID_SPACING + 2 * PADDING_BORDER
        window_height = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Warehouse — Two Agents  (+/- to change speed)")
        self.clock = pygame.time.Clock()
        # Create font once — never again. Font creation per-frame is a flicker source.
        self._hud_font = pygame.font.SysFont("monospace", 14)

    def render(self):
        """
        Single-pass render: draw everything onto self.screen, then one flip.
        No sub-env render() calls — those would insert extra flip() calls
        which cause the flickering.
        """
        self._ensure_screen()

        # ── Background ────────────────────────────────────────────────────────
        self.screen.fill((30, 30, 30))

        # ── Static world elements (use agent1_env's lists — same world) ───────
        for charge_station in self.agent1_env.charge_stations:
            self.screen.blit(charge_station.image, (charge_station.x, charge_station.y))

        for dropoff_platform in self.agent1_env.dropoff_platforms:
            self.screen.blit(dropoff_platform.image, (dropoff_platform.x, dropoff_platform.y))

        # ── Shelves (draw from agent1_env — both envs share identical shelf state
        #    because they were constructed from the same create_map() call order) ──
        for shelf in self.agent1_env.shelves:
            shelf_grid_x, shelf_grid_y = self.agent1_env._to_grid_coords(shelf)

            is_agent1_target = (
                not self.agent1_env.is_idle and
                not self.agent1_env.robot.loaded and
                (shelf_grid_x, shelf_grid_y) == (
                    self.agent1_env.target_grid_x, self.agent1_env.target_grid_y
                )
            )
            is_agent2_target = (
                not self.agent2_env.is_idle and
                not self.agent2_env.robot.loaded and
                (shelf_grid_x, shelf_grid_y) == (
                    self.agent2_env.target_grid_x, self.agent2_env.target_grid_y
                )
            )

            self.screen.blit(shelf.shadow_image, (shelf.x - 1, shelf.y + 4))

            if is_agent1_target:
                pygame.draw.rect(
                    self.screen, (0, 220, 220),
                    (shelf.x - 2, shelf.y - 2, TILE_SIZE + 4, TILE_SIZE + 4), 2
                )
            if is_agent2_target:
                pygame.draw.rect(
                    self.screen, (255, 220, 0),
                    (shelf.x - 4, shelf.y - 4, TILE_SIZE + 8, TILE_SIZE + 8), 2
                )

            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        center_offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
        center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2

        # ── Agent 1 (cyan border, vertical image) ─────────────────────────────
        a1_px = PADDING_BORDER + self.agent1_env.robot.grid_x * GRID_SPACING
        a1_py = PADDING_BORDER + self.agent1_env.robot.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, (0, 220, 220), (a1_px, a1_py, TILE_SIZE, TILE_SIZE), 2)
        agent1_image = (
            ROBOT_IMAGE_VERTICAL_BOX if self.agent1_env.robot.loaded else ROBOT_IMAGE_VERTICAL
        )
        self.screen.blit(agent1_image, (a1_px + center_offset_x, a1_py + center_offset_y))

        # ── Agent 2 (yellow border, side image) ───────────────────────────────
        a2_px = PADDING_BORDER + self.agent2_env.robot.grid_x * GRID_SPACING
        a2_py = PADDING_BORDER + self.agent2_env.robot.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, (255, 220, 0), (a2_px, a2_py, TILE_SIZE, TILE_SIZE), 2)
        agent2_image = (
            ROBOT_IMAGE_SIDE_BOX if self.agent2_env.robot.loaded else ROBOT_IMAGE_SIDE
        )
        self.screen.blit(agent2_image, (a2_px + center_offset_x, a2_py + center_offset_y))

        # ── HUD (drawn once, stable font object, no per-frame allocation) ──────
        hud_lines = [
            f"FPS: {self.fps:2d}  (+/- to adjust)",
            f"Queue: {len(self.target_queue)}",
            f"A1 score: {self.agent1_env.score}   A2 score: {self.agent2_env.score}",
            f"A1 pos: ({self.agent1_env.robot.grid_x},{self.agent1_env.robot.grid_y})  "
            f"A2 pos: ({self.agent2_env.robot.grid_x},{self.agent2_env.robot.grid_y})",
        ]
        for line_index, line_text in enumerate(hud_lines):
            label = self._hud_font.render(line_text, True, (255, 255, 255))
            self.screen.blit(label, (8, 6 + line_index * 17))

        # ── Single flip — the only one per frame ──────────────────────────────
        pygame.display.flip()
        self.clock.tick(self.fps)

    # ── event / click handling ────────────────────────────────────────────────

    def handle_events(self):
        """
        Returns False when the window is closed.
        Left-click a shelf → enqueue_target().
        + / =  → increase FPS by 1.
        -      → decrease FPS by 1 (minimum 1).
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.fps = min(self.fps + 1, 60)
                    print(f"  ⏩ FPS → {self.fps}")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.fps = max(self.fps - 1, 1)
                    print(f"  ⏪ FPS → {self.fps}")

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                clicked_grid_x = round((mouse_x - PADDING_BORDER) / GRID_SPACING)
                clicked_grid_y = round((mouse_y - PADDING_BORDER) / GRID_SPACING)

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

    print("\n🖱️  Click shelves to queue tasks. Both robots share the queue.")
    print(f"    Agent 1 (cyan)   rests at ({ROBOT_HOME_GRID_X}, {ROBOT_HOME_GRID_Y})")
    print(f"    Agent 2 (yellow) rests at ({AGENT2_HOME_GRID_X}, {AGENT2_HOME_GRID_Y})")
    print(f"    Speed: {SIMULATION_FPS} FPS — press + / - in the window to adjust live.")
    print("    Close the window to stop.\n")

    step_count = 0

    try:
        while True:
            if not env.handle_events():
                break

            env.step()
            step_count += 1

            if step_count % 50 == 0:
                print(
                    f"Step {step_count:5d} | "
                    f"A1: {env.agent1_env.score} deliveries | "
                    f"A2: {env.agent2_env.score} deliveries | "
                    f"Queue: {len(env.target_queue)} | "
                    f"FPS: {env.fps}"
                )

            env.render()

    except KeyboardInterrupt:
        print("\n\nTest ended by user.")
    finally:
        print(
            f"\nFinal — Agent 1 deliveries: {env.agent1_env.score} | "
            f"Agent 2 deliveries: {env.agent2_env.score}"
        )
        pygame.quit()


if __name__ == "__main__":
    test_two_agents(
        agent1_model_path="checkpoints/best_model.pt",
        agent2_model_path="checkpoints_agent2/best_model.pt",
    )