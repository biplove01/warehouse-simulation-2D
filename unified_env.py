import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import os
import pygame

from world import create_map
from constants import *
from robot import Robot
from q_learning_agent import DualQAgent

# ==========================================
# REWARD SHAPING & PENALTY CONSTANTS
# ==========================================
R2_REWARD_STEP = -1  # Default step penalty to encourage speed
R2_PENALTY_COLLISION = -50.0  # Heavy penalty for hitting R1 or R1 hitting R2
R2_REWARD_PICKUP = 10.0  # Reward for successfully picking up a box
R2_REWARD_DELIVER = 20.0  # Reward for successfully dropping off a box
R2_REWARD_MOVE_CLOSER = 2.0  # Multiplier for moving closer to target
R2_PENALTY_MOVE_AWAY_MULT = 4.0  # Multiplier for moving away from target
PENALTY_WALL_COLLISION = -2.0  # Penalty for attempting to walk into walls or shelves

REWARD_STAY_HOME_IDLE = 0.5         # Small positive reward for staying put at home when no task is assigned
PENALTY_WANDERING_IDLE = -1.5       # Penalty for moving away from home or wandering when unassigned
PENALTY_WRONG_DROPOFF_HOVER = -2.0  # Heavy penalty for hovering near Agent 1's drop-off zone
# ==========================================

AGENT1_QTABLE_FOLDER = "checkpoints"
AGENT1_QTABLE_FILE = "warehouse_data.pkl"

AGENT1_HOME_X, AGENT1_HOME_Y = 2, 0
AGENT2_HOME_X, AGENT2_HOME_Y = 0, 0

PHASE_FETCHING = "fetching"
PHASE_DELIVERING = "delivering"
PHASE_HOMING = "homing"

_DIR_INT = {"up": 0, "down": 1, "left": 2, "right": 3}
_ACTION_TO_DIR = {0: "up", 1: "down", 2: "left", 3: "right"}


def bfs_distance_map(start_gx, start_gy, obstacle_positions, target_is_walkable=False):
    dist_map = {}
    queue = deque()
    if target_is_walkable:
        dist_map[(start_gx, start_gy)] = 0
        queue.append((start_gx, start_gy, 0))
    else:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = start_gx + dx, start_gy + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in obstacle_positions):
                dist_map[(nx, ny)] = 0
                queue.append((nx, ny, 0))
    while queue:
        cx, cy, cd = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in obstacle_positions
                    and (nx, ny) not in dist_map):
                dist_map[(nx, ny)] = cd + 1
                queue.append((nx, ny, cd + 1))
    return dist_map


def bfs_best_action(gx, gy, dist_map):
    best_action, best_dist = None, dist_map.get((gx, gy), 999)
    for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
        dist = dist_map.get((gx + dx, gy + dy), 999)
        if dist < best_dist:
            best_dist = dist
            best_action = i
    return best_action


def predict_next(gx, gy, action, obstacle_positions):
    if action >= 4:
        return gx, gy
    dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
    nx, ny = gx + dx, gy + dy
    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
            and (nx, ny) not in obstacle_positions):
        return nx, ny
    return gx, gy


class QTablePolicy:
    def __init__(self):
        self.agent = DualQAgent(action_dim=6)
        try:
            self.agent.load_tables(AGENT1_QTABLE_FOLDER, AGENT1_QTABLE_FILE)
            print(f"✅ R1 Q-table loaded: {len(self.agent.q_table):,} states")
        except FileNotFoundError:
            print(f"⚠️ R1 Q-table not found at {AGENT1_QTABLE_FOLDER}/{AGENT1_QTABLE_FILE}. R1 will use BFS purely.")

        self.agent.epsilon = 0.0
        self.current_shelf_target_x = AGENT1_HOME_X
        self.current_shelf_target_y = AGENT1_HOME_Y

    def select_action(self, obs_8: np.ndarray) -> int:
        state = self.agent._get_state(obs_8)
        if state not in self.agent.q_table:
            return 5
        return int(np.argmax(self.agent.q_table[state]))


class TwoAgentWarehouseEnv(gym.Env):
    OBS_SIZE = 25
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        self.current_fps = self.metadata["render_fps"]

        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()

        self.obstacle_positions = {
            (round((obj.x - PADDING_BORDER) / GRID_SPACING),
             round((obj.y - PADDING_BORDER) / GRID_SPACING))
            for obj in self.shelves + self.dropoff_platforms
        }

        # Keep R1's dropoff exactly the same so its Q-Table doesn't break
        if len(self.dropoff_platforms) > 0:
            central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
            self.r1_dropoff_gx = round((central_platform.x - PADDING_BORDER) / GRID_SPACING)
            self.r1_dropoff_gy = round((central_platform.y - PADDING_BORDER) / GRID_SPACING)

            # Assign a DIFFERENT dropoff platform for R2
            if len(self.dropoff_platforms) > 1:
                for p in self.dropoff_platforms:
                    if p != central_platform:
                        self.r2_dropoff_gx = round((p.x - PADDING_BORDER) / GRID_SPACING)
                        self.r2_dropoff_gy = round((p.y - PADDING_BORDER) / GRID_SPACING)
                        break
            else:
                self.r2_dropoff_gx = self.r1_dropoff_gx + 1
                self.r2_dropoff_gy = self.r1_dropoff_gy
        else:
            self.r1_dropoff_gx = GRID_WIDTH // 2
            self.r1_dropoff_gy = GRID_HEIGHT - 1
            self.r2_dropoff_gx = (GRID_WIDTH // 2) + 1
            self.r2_dropoff_gy = GRID_HEIGHT - 1

        self.r1_dropoff_dist = bfs_distance_map(self.r1_dropoff_gx, self.r1_dropoff_gy, self.obstacle_positions)
        self.r2_dropoff_dist = bfs_distance_map(self.r2_dropoff_gx, self.r2_dropoff_gy, self.obstacle_positions)
        self.r1_home_dist = bfs_distance_map(AGENT1_HOME_X, AGENT1_HOME_Y, self.obstacle_positions, True)
        self.r2_home_dist = bfs_distance_map(AGENT2_HOME_X, AGENT2_HOME_Y, self.obstacle_positions, True)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.a1_policy = QTablePolicy()
        self.queue = deque()

    def _gc(self, obj):
        return (round((obj.x - PADDING_BORDER) / GRID_SPACING),
                round((obj.y - PADDING_BORDER) / GRID_SPACING))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.queue.clear()

        for shelf in self.shelves:
            shelf.has_box = False
            if hasattr(shelf, 'empty_image'):
                shelf.image = shelf.empty_image

        self.robot1 = Robot(start_x=AGENT1_HOME_X, start_y=AGENT1_HOME_Y)
        self.robot2 = Robot(start_x=AGENT2_HOME_X, start_y=AGENT2_HOME_Y)
        self.robot1.loaded = False
        self.robot2.loaded = False

        self._r1_phase = PHASE_HOMING
        self._r2_phase = PHASE_HOMING
        self._r1_target_gx, self._r1_target_gy = AGENT1_HOME_X, AGENT1_HOME_Y
        self._r2_target_gx, self._r2_target_gy = AGENT2_HOME_X, AGENT2_HOME_Y
        self._r1_target_dist = self.r1_home_dist
        self._r2_target_dist = self.r2_home_dist
        self._r1_direction = "up"

        self._r2_last_action = -1
        self._r2_just_picked_up = False
        self._r2_just_delivered = False

        self.r1_score = 0
        self.r2_score = 0
        self.steps = 0
        self.collision_count = 0

        self._spawn_target()
        self._spawn_target()
        self._dispatch()

        if self.render_mode == "human":
            self.render()

        return self._r2_obs(), {"action_mask": self._get_action_mask(), "bfs_action": self._get_r2_bfs_action()}

    def _spawn_target(self):
        available = [s for s in self.shelves if not s.has_box]
        if available:
            chosen = random.choice(available)
            chosen.has_box = True
            if hasattr(chosen, 'loaded_image'):
                chosen.image = chosen.loaded_image
            gx, gy = self._gc(chosen)
            if (gx, gy) not in self.queue:
                self.queue.append((gx, gy))

    def _dispatch(self):
        if self._r1_phase == PHASE_HOMING and self.queue:
            tgt = self.queue.popleft()
            self._r1_target_gx, self._r1_target_gy = tgt
            self._r1_target_dist = bfs_distance_map(tgt[0], tgt[1], self.obstacle_positions)
            self._r1_phase = PHASE_FETCHING
            self.a1_policy.current_shelf_target_x = tgt[0]
            self.a1_policy.current_shelf_target_y = tgt[1]

        if self._r2_phase == PHASE_HOMING and self.queue:
            best_idx = 0
            best_dist = 999
            for i, tgt in enumerate(list(self.queue)[:3]):
                d = abs(self.robot2.grid_x - tgt[0]) + abs(self.robot2.grid_y - tgt[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if len(self.queue) > best_idx:
                tgt = self.queue[best_idx]
                del self.queue[best_idx]
                self._r2_target_gx, self._r2_target_gy = tgt
                self._r2_target_dist = bfs_distance_map(tgt[0], tgt[1], self.obstacle_positions)
                self._r2_phase = PHASE_FETCHING

    def _r1_qtable_obs(self):
        r1 = self.robot1
        return np.array([
            r1.grid_x, r1.grid_y, float(r1.loaded),
            self.a1_policy.current_shelf_target_x, self.a1_policy.current_shelf_target_y,
            self.r1_dropoff_gx, self.r1_dropoff_gy,
            float(_DIR_INT[self._r1_direction]),
        ], dtype=np.float32)

    def _r1_action(self):
        r1 = self.robot1
        if self._r1_phase == PHASE_HOMING:
            a = bfs_best_action(r1.grid_x, r1.grid_y, self.r1_home_dist)
            return a if a is not None else 5
        if self._r1_phase == PHASE_DELIVERING:
            if abs(r1.grid_x - self.r1_dropoff_gx) + abs(r1.grid_y - self.r1_dropoff_gy) == 1:
                return 4
            a = bfs_best_action(r1.grid_x, r1.grid_y, self.r1_dropoff_dist)
            return a if a is not None else 5

        if not r1.loaded and abs(r1.grid_x - self._r1_target_gx) + abs(r1.grid_y - self._r1_target_gy) == 1:
            return 4

        action = self.a1_policy.select_action(self._r1_qtable_obs())

        if action == 5 or action >= 4:
            a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
            return a if a is not None else 0

        cur_d = self._r1_target_dist.get((r1.grid_x, r1.grid_y), 999)
        ddx, ddy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        prop_d = self._r1_target_dist.get((r1.grid_x + ddx, r1.grid_y + ddy), 999)
        if prop_d >= cur_d:
            a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
            return a if a is not None else action

        return action

    def _r1_next_pos(self):
        return predict_next(self.robot1.grid_x, self.robot1.grid_y, self._r1_action(), self.obstacle_positions)

    def _get_r2_bfs_action(self):
        robot = self.robot2
        if self._r2_phase == PHASE_HOMING:
            best_action = bfs_best_action(robot.grid_x, robot.grid_y, self.r2_home_dist)
            return best_action if best_action is not None else 5
        if self._r2_phase == PHASE_DELIVERING:
            if abs(robot.grid_x - self.r2_dropoff_gx) + abs(robot.grid_y - self.r2_dropoff_gy) == 1:
                return 4
            best_action = bfs_best_action(robot.grid_x, robot.grid_y, self.r2_dropoff_dist)
            return best_action if best_action is not None else 5
        if self._r2_phase == PHASE_FETCHING:
            if not robot.loaded and abs(robot.grid_x - self._r2_target_gx) + abs(
                    robot.grid_y - self._r2_target_gy) == 1:
                return 4
            best_action = bfs_best_action(robot.grid_x, robot.grid_y, self._r2_target_dist)
            return best_action if best_action is not None else 5
        return 5

    def _r2_obs(self):
        r2, r1 = self.robot2, self.robot1
        nav_x = self.r2_dropoff_gx if r2.loaded else (
            AGENT2_HOME_X if self._r2_phase == PHASE_HOMING else self._r2_target_gx)
        nav_y = self.r2_dropoff_gy if r2.loaded else (
            AGENT2_HOME_Y if self._r2_phase == PHASE_HOMING else self._r2_target_gy)

        lx, ly = 0.0, 0.0
        if self._r2_last_action == 0:
            ly = -1.0
        elif self._r2_last_action == 1:
            ly = 1.0
        elif self._r2_last_action == 2:
            lx = -1.0
        elif self._r2_last_action == 3:
            lx = 1.0

        can_pickup = float(
            not r2.loaded and self._r2_phase == PHASE_FETCHING and abs(r2.grid_x - self._r2_target_gx) + abs(
                r2.grid_y - self._r2_target_gy) == 1)
        can_deliver = float(
            r2.loaded and abs(r2.grid_x - self.r2_dropoff_gx) + abs(r2.grid_y - self.r2_dropoff_gy) == 1)

        base = [
            r2.grid_x / GRID_WIDTH, r2.grid_y / GRID_HEIGHT, float(r2.loaded),
            (nav_x - r2.grid_x) / GRID_WIDTH, (nav_y - r2.grid_y) / GRID_HEIGHT,
            (self.r2_dropoff_gx - r2.grid_x) / GRID_WIDTH, (self.r2_dropoff_gy - r2.grid_y) / GRID_HEIGHT,
        ]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = r2.grid_x + dx, r2.grid_y + dy
            oob = not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT)
            wall = (nx, ny) in self.obstacle_positions
            is_r1 = (nx == r1.grid_x and ny == r1.grid_y)
            base.append(1.0 if (oob or wall or is_r1) else 0.0)

        base += [lx, ly, can_pickup, can_deliver]

        r1nx, r1ny = self._r1_next_pos()
        extra = [
            (r1.grid_x - r2.grid_x) / GRID_WIDTH, (r1.grid_y - r2.grid_y) / GRID_HEIGHT,
            float(r1.loaded), float(self._r1_phase == PHASE_HOMING),
            (r1nx - r2.grid_x) / GRID_WIDTH, (r1ny - r2.grid_y) / GRID_HEIGHT,
        ]
        return np.array(base + extra, dtype=np.float32)

    def _get_action_mask(self):
        mask = np.ones(6, dtype=np.float32)
        r1nx, r1ny = self._r1_next_pos()

        for a, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = self.robot2.grid_x + dx, self.robot2.grid_y + dy
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                mask[a] = 0.0
            elif (nx, ny) in self.obstacle_positions:
                mask[a] = 0.0
            elif nx == r1nx and ny == r1ny:
                mask[a] = 0.0

        can_pickup = (not self.robot2.loaded and self._r2_phase == PHASE_FETCHING and abs(
            self.robot2.grid_x - self._r2_target_gx) + abs(self.robot2.grid_y - self._r2_target_gy) == 1)
        can_deliver = (self.robot2.loaded and abs(self.robot2.grid_x - self.r2_dropoff_gx) + abs(
            self.robot2.grid_y - self.r2_dropoff_gy) == 1)
        if not can_pickup and not can_deliver:
            mask[4] = 0.0

        return mask

    def step(self, r2_action):
        self.steps += 1
        r1, r2 = self.robot1, self.robot2

        if len(self.queue) < 2 and random.random() < 0.1:
            self._spawn_target()
        self._dispatch()

        a1 = self._r1_action()
        r1nx, r1ny = predict_next(r1.grid_x, r1.grid_y, a1, self.obstacle_positions)
        r2nx, r2ny = predict_next(r2.grid_x, r2.grid_y, r2_action, self.obstacle_positions)

        # Collision logic: R2 hitting R1, or R1 hitting R2
        r2_into_r1 = (r2nx == r1nx and r2ny == r1ny)
        r1_into_r2 = (r1nx == r2.grid_x and r1ny == r2.grid_y)
        collision = r2_into_r1 or r1_into_r2

        if a1 < 4 and (r1nx, r1ny) not in self.obstacle_positions:
            r1.grid_x, r1.grid_y = r1nx, r1ny
            self._r1_direction = _ACTION_TO_DIR[a1]

        dist_map = self.r2_dropoff_dist if r2.loaded else (
            self.r2_home_dist if self._r2_phase == PHASE_HOMING else self._r2_target_dist)
        dist_before = dist_map.get((r2.grid_x, r2.grid_y), 50)

        r2_reward = R2_REWARD_STEP

        # ISSUE 1: Penalize wandering when unassigned (Homing Phase)
        if self._r2_phase == PHASE_HOMING:
            at_home = (r2.grid_x == AGENT2_HOME_X and r2.grid_y == AGENT2_HOME_Y)

            if at_home:
                if r2_action == 5:  # Action 5 is "Wait"
                    r2_reward += REWARD_STAY_HOME_IDLE
                else:
                    # Penalize picking movement actions when it should just wait at home
                    r2_reward += PENALTY_WANDERING_IDLE
            else:
                if r2_action == 5:
                    r2_reward += PENALTY_WANDERING_IDLE

        # ISSUE 2: Penalize hovering near Robot 1's drop-off zone
        if self._r2_phase == PHASE_DELIVERING:
            dist_to_r1_dropoff = self.r1_dropoff_dist.get((r2.grid_x, r2.grid_y), 999)

            if dist_to_r1_dropoff <= 2:
                r2_reward += PENALTY_WRONG_DROPOFF_HOVER


        if collision:
            self.collision_count += 1
            r2_reward += R2_PENALTY_COLLISION
            r2_action = 5
        else:

            if r2_action < 4:
                if r2nx == r2.grid_x and r2ny == r2.grid_y:
                    r2_reward += PENALTY_WALL_COLLISION
                else:
                    r2.grid_x, r2.grid_y = r2nx, r2ny
                    dist_after = dist_map.get((r2nx, r2ny), 50)
                    delta = dist_before - dist_after
                    r2_reward += (delta * R2_REWARD_MOVE_CLOSER if delta >= 0 else delta * R2_PENALTY_MOVE_AWAY_MULT)

            elif r2_action == 4:
                if self._r2_phase == PHASE_FETCHING and not r2.loaded:
                    if abs(r2.grid_x - self._r2_target_gx) + abs(r2.grid_y - self._r2_target_gy) == 1:
                        r2.loaded = True
                        self._r2_just_picked_up = True
                        r2_reward += R2_REWARD_PICKUP
                        for s in self.shelves:
                            if self._gc(s) == (self._r2_target_gx, self._r2_target_gy):
                                s.has_box = False
                                if hasattr(s, 'empty_image'):
                                    s.image = s.empty_image
                                break
                elif self._r2_phase == PHASE_DELIVERING and r2.loaded:
                    if abs(r2.grid_x - self.r2_dropoff_gx) + abs(r2.grid_y - self.r2_dropoff_gy) == 1:
                        r2.loaded = False
                        self.r2_score += 1
                        self._r2_just_delivered = True
                        r2_reward += R2_REWARD_DELIVER


        self._r2_last_action = r2_action

        self._update_r1_phase(a1)
        self._update_r2_phase()

        if self.render_mode == "human":
            self.render()

        done = self.steps >= 1000
        info = {
            "action_mask": self._get_action_mask(),
            "bfs_action": self._get_r2_bfs_action(),
            "r1_score": self.r1_score,
            "r2_score": self.r2_score,
            "collisions": self.collision_count
        }

        return self._r2_obs(), r2_reward, done, False, info

    def _update_r1_phase(self, action):
        if self._r1_phase == PHASE_FETCHING and action == 4:
            if abs(self.robot1.grid_x - self._r1_target_gx) + abs(self.robot1.grid_y - self._r1_target_gy) == 1:
                self.robot1.loaded = True
                self._r1_phase = PHASE_DELIVERING
                for s in self.shelves:
                    if self._gc(s) == (self._r1_target_gx, self._r1_target_gy):
                        s.has_box = False
                        if hasattr(s, 'empty_image'):
                            s.image = s.empty_image
                        break
        elif self._r1_phase == PHASE_DELIVERING and action == 4:
            if abs(self.robot1.grid_x - self.r1_dropoff_gx) + abs(self.robot1.grid_y - self.r1_dropoff_gy) == 1:
                self.robot1.loaded = False
                self.r1_score += 1
                self._r1_phase = PHASE_HOMING
                self._r1_target_gx, self._r1_target_gy = AGENT1_HOME_X, AGENT1_HOME_Y
                self._r1_target_dist = self.r1_home_dist

    def _update_r2_phase(self):
        if self._r2_phase == PHASE_FETCHING and self._r2_just_picked_up:
            self._r2_just_picked_up = False
            self._r2_phase = PHASE_DELIVERING
        elif self._r2_phase == PHASE_DELIVERING and self._r2_just_delivered:
            self._r2_just_delivered = False
            self._r2_phase = PHASE_HOMING
            self._r2_target_gx, self._r2_target_gy = AGENT2_HOME_X, AGENT2_HOME_Y
            self._r2_target_dist = self.r2_home_dist

    def _safe_draw(self, canvas, entity, fallback_color):
        if hasattr(entity, 'draw') and callable(getattr(entity, 'draw')):
            entity.draw(canvas)
        elif hasattr(entity, 'image') and entity.image is not None:
            canvas.blit(entity.image, (entity.x, entity.y))
        else:
            pygame.draw.rect(canvas, fallback_color, (entity.x, entity.y, GRID_SPACING - 2, GRID_SPACING - 2))

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            window_width = GRID_WIDTH * GRID_SPACING + (PADDING_BORDER * 2)
            window_height = GRID_HEIGHT * GRID_SPACING + (PADDING_BORDER * 2)
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Two-Agent Warehouse Optimization Training")
            self.font = pygame.font.SysFont("Arial", 18, bold=True)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    self.current_fps = min(120, self.current_fps + 5)
                    print(f"Speed increased: {self.current_fps} FPS")
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.current_fps = max(5, self.current_fps - 5)
                    print(f"Speed decreased: {self.current_fps} FPS")

        window_width = GRID_WIDTH * GRID_SPACING + (PADDING_BORDER * 2)
        window_height = GRID_HEIGHT * GRID_SPACING + (PADDING_BORDER * 2)
        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((240, 240, 240))

        color_r1 = (220, 40, 40)
        color_r2 = (40, 80, 220)

        for station in self.charge_stations:
            self._safe_draw(canvas, station, (150, 150, 150))
        for platform in self.dropoff_platforms:
            self._safe_draw(canvas, platform, (100, 200, 100))
        for shelf in self.shelves:
            self._safe_draw(canvas, shelf, (200, 150, 100))

        # Highlight Dropoff Platforms
        r1dx, r1dy = PADDING_BORDER + self.r1_dropoff_gx * GRID_SPACING, PADDING_BORDER + self.r1_dropoff_gy * GRID_SPACING
        pygame.draw.rect(canvas, color_r1, (r1dx, r1dy, GRID_SPACING, GRID_SPACING), width=3)

        r2dx, r2dy = PADDING_BORDER + self.r2_dropoff_gx * GRID_SPACING, PADDING_BORDER + self.r2_dropoff_gy * GRID_SPACING
        pygame.draw.rect(canvas, color_r2, (r2dx, r2dy, GRID_SPACING, GRID_SPACING), width=3)

        # Highlight Target Shelves
        if self._r1_phase == PHASE_FETCHING:
            rx, ry = PADDING_BORDER + self._r1_target_gx * GRID_SPACING, PADDING_BORDER + self._r1_target_gy * GRID_SPACING
            pygame.draw.rect(canvas, color_r1, (rx, ry, GRID_SPACING, GRID_SPACING), width=3)

        if self._r2_phase == PHASE_FETCHING:
            bx, by = PADDING_BORDER + self._r2_target_gx * GRID_SPACING, PADDING_BORDER + self._r2_target_gy * GRID_SPACING
            pygame.draw.rect(canvas, color_r2, (bx, by, GRID_SPACING, GRID_SPACING), width=3)

        # Draw R1
        self.robot1.x = PADDING_BORDER + self.robot1.grid_x * GRID_SPACING
        self.robot1.y = PADDING_BORDER + self.robot1.grid_y * GRID_SPACING
        self._safe_draw(canvas, self.robot1, color_r1)
        pygame.draw.rect(canvas, color_r1, (self.robot1.x, self.robot1.y, GRID_SPACING, GRID_SPACING), width=2)
        text_r1 = self.font.render("R1", True, color_r1)
        canvas.blit(text_r1, (self.robot1.x + 5, self.robot1.y - 20))

        if self.robot1.loaded:
            pygame.draw.circle(canvas, color_r1,
                               (int(self.robot1.x + GRID_SPACING // 2), int(self.robot1.y + GRID_SPACING // 2)), 8)

        # Draw R2
        self.robot2.x = PADDING_BORDER + self.robot2.grid_x * GRID_SPACING
        self.robot2.y = PADDING_BORDER + self.robot2.grid_y * GRID_SPACING
        self._safe_draw(canvas, self.robot2, color_r2)
        pygame.draw.rect(canvas, color_r2, (self.robot2.x, self.robot2.y, GRID_SPACING, GRID_SPACING), width=2)
        text_r2 = self.font.render("R2", True, color_r2)
        canvas.blit(text_r2, (self.robot2.x + 5, self.robot2.y - 20))

        if self.robot2.loaded:
            pygame.draw.circle(canvas, color_r2,
                               (int(self.robot2.x + GRID_SPACING // 2), int(self.robot2.y + GRID_SPACING // 2)), 8)

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(self.current_fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None