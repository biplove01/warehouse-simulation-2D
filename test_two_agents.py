import torch
import pygame
import numpy as np
from collections import deque
import pickle
import os

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from model import QNetwork
from constants import *


compute_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

SIMULATION_FPS = 6

AGENT2_HOME_GRID_X = 0
AGENT2_HOME_GRID_Y = 0

AGENT1_OBS_SIZE = 19
AGENT2_OBS_SIZE = 25

# ─── DEBUG CONFIG ─────────────────────────────────────────────────────────────
LOG_EVERY_N_STEPS = 1
LOG_COLLISIONS    = True
LOG_DISPATCH      = True
LOG_DELIVERY      = True
LOG_PHASE         = False

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Interact", "Wait"]


def _fmt(action: int, forced: bool = False) -> str:
    name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)
    return f"{name}{'(FORCED)' if forced else ''}"


# ─── Q-TABLE POLICY WRAPPER ───────────────────────────────────────────────────

class QTablePolicy:
    """
    Loads Agent 1's Q-table from a .pkl file and acts greedily (epsilon=0).
    The pkl is expected to be a dict mapping state-tuples → action-value arrays,
    which is the format saved by DualQAgent.save_tables().

    Falls back to action 5 (Wait) for unseen states so the robot never crashes
    on an observation it never encountered during Q-table training.
    """

    def __init__(self, data_folder: str = "training_data",
                 file_name: str = "warehouse_data.pkl"):
        pkl_path = os.path.join(data_folder, file_name)
        try:
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)

            # DualQAgent.save_tables() may store {"q_table": ..., ...} or
            # the raw dict directly — handle both.
            if isinstance(payload, dict) and "q_table" in payload:
                self.q_table = payload["q_table"]
            else:
                self.q_table = payload

            print(
                f"  ✅ Agent 1 Q-table loaded from '{pkl_path}' "
                f"({len(self.q_table)} states)"
            )
        except FileNotFoundError:
            print(f"  ❌ Agent 1: '{pkl_path}' not found — defaulting to Wait.")
            self.q_table = {}

    def select_action(self, obs: np.ndarray) -> int:
        """Greedy lookup. Returns Wait (5) for unseen states."""
        key = tuple(obs.astype(int).tolist())
        if key in self.q_table:
            return int(np.argmax(self.q_table[key]))
        # Unseen state — log once per unique miss then stay still
        return 5


# ─── PER-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class SingleAgentTestingEnv(WarehouseEnv):

    def __init__(self, home_grid_x: int, home_grid_y: int,
                 label: str = "??", render_mode=None):
        super().__init__(render_mode=render_mode)
        self.home_grid_x = home_grid_x
        self.home_grid_y = home_grid_y
        self.label = label

        self.own_home_distance_map = self._bfs_distance_map(
            home_grid_x, home_grid_y, target_is_walkable=True
        )
        self.returning_home = True
        self.is_idle = True

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image
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
        return self.is_idle

    def assign_target(self, target_grid_x, target_grid_y):
        self.target_grid_x = target_grid_x
        self.target_grid_y = target_grid_y
        self.target_distance_map = self._bfs_distance_map(target_grid_x, target_grid_y)
        self.returning_home = False
        self.is_idle = False
        if LOG_DISPATCH:
            print(
                f"  📋 [{self.label}] dispatched → shelf ({target_grid_x},{target_grid_y}) "
                f"| robot at ({self.robot.grid_x},{self.robot.grid_y})"
            )

    def _spawn_new_target(self):
        pass

    def _on_delivery(self):
        self.returning_home = False
        self.is_idle = True
        self.target_grid_x = self.dropoff_grid_x
        self.target_grid_y = self.dropoff_grid_y
        if LOG_DELIVERY:
            print(
                f"  ✅ [{self.label}] delivery complete — score={self.score} "
                f"| now idle at ({self.robot.grid_x},{self.robot.grid_y})"
            )

    def step(self, action):
        robot = self.robot

        if self.is_idle:
            if LOG_PHASE:
                print(f"  💤 [{self.label}] IDLE at ({robot.grid_x},{robot.grid_y})")
            self.steps += 1
            return self._get_observation(), 0.0, False, False, {}

        if self.returning_home:
            self.steps += 1
            if robot.grid_x == self.home_grid_x and robot.grid_y == self.home_grid_y:
                self.returning_home = False
                self.is_idle = True
                if LOG_PHASE:
                    print(f"  🏠 [{self.label}] reached home → idle")
                return self._get_observation(), 0.0, False, False, {}

            if LOG_PHASE:
                print(
                    f"  🔙 [{self.label}] HOME-RETURN "
                    f"({robot.grid_x},{robot.grid_y}) action={_fmt(action)}"
                )
            dist_before = self.own_home_distance_map.get((robot.grid_x, robot.grid_y), 50)
            if action < 4:
                direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                delta_x, delta_y = direction_deltas[action]
                next_x = robot.grid_x + delta_x
                next_y = robot.grid_y + delta_y
                if (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
                        and (next_x, next_y) not in self.obstacle_positions):
                    robot.grid_x, robot.grid_y = next_x, next_y
                    dist_after = self.own_home_distance_map.get((robot.grid_x, robot.grid_y), 50)
                    reward = (dist_before - dist_after) * self.reward_manager.progress_reward_scale
                else:
                    reward = self.reward_manager.collision_penalty
            else:
                reward = self.reward_manager.step_penalty
            return self._get_observation(), reward, False, False, {}

        # WORKING PHASE
        if LOG_PHASE:
            print(
                f"  🔧 [{self.label}] WORKING "
                f"({robot.grid_x},{robot.grid_y}) loaded={int(robot.loaded)} "
                f"target=({self.target_grid_x},{self.target_grid_y}) "
                f"action={_fmt(action)}"
            )
        obs, reward, _, _, info = super().step(action)
        return obs, reward, False, False, info

    def _handle_pygame_events(self):
        pass


# ─── TWO-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class TwoAgentTestingEnv:
    """
    Agent 1 — deterministic Q-table loaded from a .pkl file.
    Agent 2 — DQN policy loaded from a .pt checkpoint.
    """

    def __init__(
        self,
        agent1_data_folder: str = "training_data",
        agent1_file_name: str = "warehouse_data.pkl",
        agent2_model_path: str = "checkpoints_agent2/best_model.pt",
    ):
        self.agent1_env = SingleAgentTestingEnv(
            home_grid_x=ROBOT_HOME_GRID_X,
            home_grid_y=ROBOT_HOME_GRID_Y,
            label="A1",
            render_mode=None,
        )
        self.agent2_env = SingleAgentTestingEnv(
            home_grid_x=AGENT2_HOME_GRID_X,
            home_grid_y=AGENT2_HOME_GRID_Y,
            label="A2",
            render_mode=None,
        )

        self.target_queue = deque()

        # ── Agent 1: Q-table from pkl ─────────────────────────────────────────
        self.agent1_policy = QTablePolicy(
            data_folder=agent1_data_folder,
            file_name=agent1_file_name,
        )

        # ── Agent 2: neural net from .pt ──────────────────────────────────────
        action_dim = self.agent2_env.action_space.n
        self.agent2_policy = QNetwork(AGENT2_OBS_SIZE, action_dim).to(compute_device)
        self._load_nn_weights(self.agent2_policy, agent2_model_path, "Agent 2")

        self._agent1_obs = None
        self._agent2_obs = None

        self.screen = None
        self.clock = None
        self.fps = SIMULATION_FPS
        self._hud_font = None
        self._step_count = 0

        # Track unseen-state misses for Agent 1 so we don't spam the console
        self._a1_unseen_count = 0

    # ── weight loader (neural net only) ──────────────────────────────────────

    def _load_nn_weights(self, network, model_path, label):
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
        self._step_count = 0
        self._a1_unseen_count = 0

    # ── shared queue ──────────────────────────────────────────────────────────

    def enqueue_target(self, target_grid_x, target_grid_y):
        if (target_grid_x, target_grid_y) in self.target_queue:
            print(f"  ⚠️  Shelf ({target_grid_x},{target_grid_y}) already queued.")
            return
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
            f"  📦 Shelf ({target_grid_x},{target_grid_y}) queued. "
            f"Queue depth: {len(self.target_queue)}"
        )

    # ── dispatcher ────────────────────────────────────────────────────────────

    def _dispatch_tasks(self):
        for agent_env, other_env in (
            (self.agent1_env, self.agent2_env),
            (self.agent2_env, self.agent1_env),
        ):
            if agent_env.is_free and self.target_queue:
                candidate_x, candidate_y = self.target_queue[0]
                other_at_target = (
                    other_env.robot.grid_x == candidate_x and
                    other_env.robot.grid_y == candidate_y
                )
                if other_at_target:
                    print(
                        f"  ⏸  [{agent_env.label}] dispatch stalled — "
                        f"{other_env.label} is sitting on target "
                        f"({candidate_x},{candidate_y}), retrying next step"
                    )
                else:
                    self.target_queue.popleft()
                    agent_env.assign_target(candidate_x, candidate_y)

    # ── collision prediction ───────────────────────────────────────────────────

    def _predict_next_position(self, robot, action, obstacle_positions):
        if action >= 4:
            return robot.grid_x, robot.grid_y
        direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        delta_x, delta_y = direction_deltas[action]
        next_x = robot.grid_x + delta_x
        next_y = robot.grid_y + delta_y
        if (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
                and (next_x, next_y) not in obstacle_positions):
            return next_x, next_y
        return robot.grid_x, robot.grid_y

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self):
        self._dispatch_tasks()
        self._step_count += 1
        should_log = (self._step_count % LOG_EVERY_N_STEPS == 0)

        # ── Agent 1 action: Q-table lookup (no GPU, no torch) ────────────────
        key = tuple(self._agent1_obs.astype(int).tolist())
        if key not in self.agent1_policy.q_table:
            self._a1_unseen_count += 1
            if self._a1_unseen_count <= 5 or self._a1_unseen_count % 50 == 0:
                print(
                    f"  🔍 [A1] unseen state at step {self._step_count} "
                    f"(total misses: {self._a1_unseen_count}) — defaulting to Wait"
                )
        agent1_action = self.agent1_policy.select_action(self._agent1_obs)

        # ── Agent 2 action: neural net forward pass ───────────────────────────
        with torch.no_grad():
            agent2_action = self.agent2_policy(
                torch.as_tensor(
                    self._agent2_obs, dtype=torch.float32, device=compute_device
                ).unsqueeze(0)
            ).argmax().item()

        # Predict next positions.
        agent1_next_x, agent1_next_y = self._predict_next_position(
            self.agent1_env.robot, agent1_action,
            self.agent1_env.obstacle_positions,
        )
        agent2_next_x, agent2_next_y = self._predict_next_position(
            self.agent2_env.robot, agent2_action,
            self.agent2_env.obstacle_positions,
        )

        # Collision check.
        agent2_would_collide = (agent2_action < 4) and (
            (agent2_next_x == self.agent1_env.robot.grid_x and
             agent2_next_y == self.agent1_env.robot.grid_y)
            or
            (agent2_next_x == agent1_next_x and
             agent2_next_y == agent1_next_y)
        )
        effective_agent2_action = 5 if agent2_would_collide else agent2_action

        # ── Per-step action table ─────────────────────────────────────────────
        if should_log:
            a1r = self.agent1_env.robot
            a2r = self.agent2_env.robot
            print(
                f"  [{self._step_count:5d}] "
                f"A1 ({a1r.grid_x:2d},{a1r.grid_y:2d}) ld={int(a1r.loaded)} "
                f"→ {_fmt(agent1_action):<10s} next=({agent1_next_x},{agent1_next_y})  |  "
                f"A2 ({a2r.grid_x:2d},{a2r.grid_y:2d}) ld={int(a2r.loaded)} "
                f"→ {_fmt(effective_agent2_action, forced=agent2_would_collide):<18s} "
                f"next=({agent2_next_x},{agent2_next_y})  "
                f"queue={len(self.target_queue)}"
            )

        # ── Collision warning ─────────────────────────────────────────────────
        if agent2_would_collide and LOG_COLLISIONS:
            print(
                f"  ⚠️  [{self._step_count}] COLLISION AVOIDED — "
                f"A2 wanted {_fmt(agent2_action)} → ({agent2_next_x},{agent2_next_y}) "
                f"but A1 occupies/is moving to that cell → A2 forced to Wait"
            )

        # Execute steps.
        agent1_next_obs, agent1_reward, _, _, _ = self.agent1_env.step(agent1_action)
        agent2_next_obs, agent2_reward, _, _, _ = self.agent2_env.step(effective_agent2_action)

        if agent2_would_collide:
            agent2_reward += -20.0

        self._agent1_obs = agent1_next_obs
        self._agent2_obs = self._build_agent2_obs(agent2_next_obs)

        return agent1_reward, agent2_reward

    # ── Agent 2 observation builder ───────────────────────────────────────────

    def _build_agent2_obs(self, agent2_base_obs):
        a2r = self.agent2_env.robot
        a1r = self.agent1_env.robot
        extra = np.array([
            (a1r.grid_x - a2r.grid_x) / GRID_WIDTH,
            (a1r.grid_y - a2r.grid_y) / GRID_HEIGHT,
            float(a1r.loaded),
            float(self.agent1_env.returning_home or self.agent1_env.is_idle),
            (self.agent1_env.target_grid_x - a2r.grid_x) / GRID_WIDTH,
            (self.agent1_env.target_grid_y - a2r.grid_y) / GRID_HEIGHT,
        ], dtype=np.float32)
        return np.concatenate([agent2_base_obs, extra])

    # ── render ────────────────────────────────────────────────────────────────

    def _ensure_screen(self):
        if self.screen is not None:
            return
        pygame.init()
        window_width = GRID_WIDTH * GRID_SPACING + 2 * PADDING_BORDER
        window_height = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(
            "Warehouse — A1:Q-table  A2:DQN  (+/- speed)"
        )
        self.clock = pygame.time.Clock()
        self._hud_font = pygame.font.SysFont("monospace", 14)

    def render(self):
        self._ensure_screen()
        self.screen.fill((30, 30, 30))

        for cs in self.agent1_env.charge_stations:
            self.screen.blit(cs.image, (cs.x, cs.y))
        for dp in self.agent1_env.dropoff_platforms:
            self.screen.blit(dp.image, (dp.x, dp.y))

        for shelf in self.agent1_env.shelves:
            shelf_grid_x, shelf_grid_y = self.agent1_env._to_grid_coords(shelf)
            is_agent1_target = (
                not self.agent1_env.is_idle and not self.agent1_env.robot.loaded and
                (shelf_grid_x, shelf_grid_y) == (
                    self.agent1_env.target_grid_x, self.agent1_env.target_grid_y)
            )
            is_agent2_target = (
                not self.agent2_env.is_idle and not self.agent2_env.robot.loaded and
                (shelf_grid_x, shelf_grid_y) == (
                    self.agent2_env.target_grid_x, self.agent2_env.target_grid_y)
            )
            self.screen.blit(shelf.shadow_image, (shelf.x - 1, shelf.y + 4))
            if is_agent1_target:
                pygame.draw.rect(self.screen, (0, 220, 220),
                                 (shelf.x - 2, shelf.y - 2, TILE_SIZE + 4, TILE_SIZE + 4), 2)
            if is_agent2_target:
                pygame.draw.rect(self.screen, (255, 220, 0),
                                 (shelf.x - 4, shelf.y - 4, TILE_SIZE + 8, TILE_SIZE + 8), 2)
            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        cx = (TILE_SIZE - ROBOT_WIDTH) // 2
        cy = (TILE_SIZE - ROBOT_HEIGHT) // 2

        a1_px = PADDING_BORDER + self.agent1_env.robot.grid_x * GRID_SPACING
        a1_py = PADDING_BORDER + self.agent1_env.robot.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, (0, 220, 220), (a1_px, a1_py, TILE_SIZE, TILE_SIZE), 2)
        a1_img = ROBOT_IMAGE_VERTICAL_BOX if self.agent1_env.robot.loaded else ROBOT_IMAGE_VERTICAL
        self.screen.blit(a1_img, (a1_px + cx, a1_py + cy))

        a2_px = PADDING_BORDER + self.agent2_env.robot.grid_x * GRID_SPACING
        a2_py = PADDING_BORDER + self.agent2_env.robot.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, (255, 220, 0), (a2_px, a2_py, TILE_SIZE, TILE_SIZE), 2)
        a2_img = ROBOT_IMAGE_SIDE_BOX if self.agent2_env.robot.loaded else ROBOT_IMAGE_SIDE
        self.screen.blit(a2_img, (a2_px + cx, a2_py + cy))

        hud_lines = [
            f"FPS: {self.fps:2d}  (+/- to adjust)",
            f"Queue: {len(self.target_queue)}",
            f"A1 (Q-table): {self.agent1_env.score}   A2 (DQN): {self.agent2_env.score}",
            f"A1 pos: ({self.agent1_env.robot.grid_x},{self.agent1_env.robot.grid_y})  "
            f"A2 pos: ({self.agent2_env.robot.grid_x},{self.agent2_env.robot.grid_y})  "
            f"A1 unseen: {self._a1_unseen_count}",
        ]
        for i, line in enumerate(hud_lines):
            surf = self._hud_font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (8, 6 + i * 17))

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ── event / click handling ────────────────────────────────────────────────

    def handle_events(self):
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
    agent1_data_folder: str = "training_data",
    agent1_file_name: str = "warehouse_data.pkl",
    agent2_model_path: str = "checkpoints_agent2/best_model.pt",
):
    env = TwoAgentTestingEnv(
        agent1_data_folder=agent1_data_folder,
        agent1_file_name=agent1_file_name,
        agent2_model_path=agent2_model_path,
    )
    env.reset()
    env.render()

    print("\n🖱️  Click shelves to queue tasks. Both robots share the queue.")
    print(f"    Agent 1 (cyan,   Q-table) rests at ({ROBOT_HOME_GRID_X}, {ROBOT_HOME_GRID_Y})")
    print(f"    Agent 2 (yellow, DQN)     rests at ({AGENT2_HOME_GRID_X}, {AGENT2_HOME_GRID_Y})")
    print(f"    Speed: {SIMULATION_FPS} FPS — press + / - to adjust live.")
    print(
        f"    Logging every {LOG_EVERY_N_STEPS} step(s) | "
        f"collisions={LOG_COLLISIONS} dispatch={LOG_DISPATCH} "
        f"delivery={LOG_DELIVERY} phase={LOG_PHASE}"
    )
    print("    Close the window to stop.\n")
    print(
        f"  {'[step]':7s}  "
        f"{'A1 pos':8s} ld  {'action':<10s} {'→next':<10s}   "
        f"{'A2 pos':8s} ld  {'action':<18s} {'→next':<10s} queue"
    )
    print("  " + "─" * 95)

    step_count = 0
    try:
        while True:
            if not env.handle_events():
                break
            env.step()
            step_count += 1
            if step_count % 50 == 0:
                print(
                    f"\n  ── step {step_count} summary ──  "
                    f"A1: {env.agent1_env.score} deliveries | "
                    f"A2: {env.agent2_env.score} deliveries | "
                    f"queue: {len(env.target_queue)} | "
                    f"A1 unseen states: {env._a1_unseen_count} | "
                    f"fps: {env.fps}\n"
                )
            env.render()
    except KeyboardInterrupt:
        print("\n\nTest ended by user.")
    finally:
        print(
            f"\nFinal — Agent 1 deliveries: {env.agent1_env.score} | "
            f"Agent 2 deliveries: {env.agent2_env.score} | "
            f"Agent 1 unseen-state misses: {env._a1_unseen_count}"
        )
        pygame.quit()


if __name__ == "__main__":
    test_two_agents(
        agent1_data_folder="training_data",
        agent1_file_name="warehouse_data.pkl",
        agent2_model_path="checkpoints_agent2/best_model.pt",
    )