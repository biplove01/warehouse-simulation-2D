import torch
import torch.nn as nn
import pygame
import numpy as np
from collections import deque

from unified_env import TwoAgentWarehouseEnv, AGENT1_HOME_X, AGENT1_HOME_Y, AGENT2_HOME_X, AGENT2_HOME_Y
from q_learning_agent import DualQAgent
from constants import *

AGENT1_QTABLE_FOLDER = "checkpoints"
AGENT1_QTABLE_FILE = "warehouse_data.pkl"
AGENT2_MODEL_PATH = "checkpoints/best_model.pth"

SIMULATION_FPS = 6

LOG_EVERY_STEP = True
LOG_COLLISIONS = True
LOG_DELIVERY = True
LOG_DISPATCH = True

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Interact", "Wait"]

PHASE_FETCHING = "fetching"
PHASE_DELIVERING = "delivering"
PHASE_HOMING = "homing"

compute_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_DIR_INT = {"up": 0, "down": 1, "left": 2, "right": 3}
_ACTION_TO_DIR = {0: "up", 1: "down", 2: "left", 3: "right"}


class DeepQNetwork(nn.Module):
    def __init__(self, input_dimension, action_dimension):
        super().__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dimension)
        )

    def forward(self, state):
        return self.network_layers(state)


def bfs_distance_map(start_gx, start_gy, obstacle_positions, target_is_walkable=False):
    dist_map = {}
    queue = deque()
    if target_is_walkable:
        dist_map[(start_gx, start_gy)] = 0
        queue.append((start_gx, start_gy, 0))
    else:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = start_gx + dx, start_gy + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in obstacle_positions):
                dist_map[(nx, ny)] = 0
                queue.append((nx, ny, 0))
    while queue:
        cx, cy, cd = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in obstacle_positions and (nx, ny) not in dist_map):
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
    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in obstacle_positions):
        return nx, ny
    return gx, gy


class QTablePolicy:
    def __init__(self):
        self.agent = DualQAgent(action_dim=6)
        self.agent.load_tables(AGENT1_QTABLE_FOLDER, AGENT1_QTABLE_FILE)
        self.agent.epsilon = 0.0
        self.current_shelf_target_x = AGENT1_HOME_X
        self.current_shelf_target_y = AGENT1_HOME_Y
        print(f"  ✅ Q-table loaded: {len(self.agent.q_table):,} states")

    def select_action(self, obs_8: np.ndarray) -> int:
        state = self.agent._get_state(obs_8)
        if state not in self.agent.q_table:
            return 5
        return int(np.argmax(self.agent.q_table[state]))


class SharedTargetQueue:
    def __init__(self, shelves, gc_fn):
        self._shelves = shelves
        self._gc = gc_fn
        self._queue = deque()
        self._assignments = {}

    def enqueue(self, gx, gy):
        if (gx, gy) in self._queue or (gx, gy) in self._assignments.values():
            return
        for s in self._shelves:
            if self._gc(s) == (gx, gy):
                s.has_box = True
                s.image = s.loaded_image
                break
        self._queue.append((gx, gy))

    def try_assign(self, robot_id, other_robot_pos):
        if not self._queue:
            return None
        gx, gy = self._queue[0]
        if other_robot_pos == (gx, gy):
            return None
        self._queue.popleft()
        self._assignments[robot_id] = (gx, gy)
        return gx, gy

    def release(self, robot_id):
        self._assignments.pop(robot_id, None)


class TwoAgentTestingEnv:
    OBS_SIZE = 25

    def __init__(self):
        self._base = TwoAgentWarehouseEnv(render_mode=None)
        self._base.reset()

        self.obstacle_positions = self._base.obstacle_positions
        self.shelves = self._base.shelves
        self.dropoff_platforms = self._base.dropoff_platforms
        self.charge_stations = self._base.charge_stations

        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        if len(self.dropoff_platforms) > 0:
            central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
            self.r1_dropoff_gx = round((central_platform.x - PADDING_BORDER) / GRID_SPACING)
            self.r1_dropoff_gy = round((central_platform.y - PADDING_BORDER) / GRID_SPACING)

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
        self.r1_home_dist = bfs_distance_map(AGENT1_HOME_X, AGENT1_HOME_Y, self.obstacle_positions,
                                             target_is_walkable=True)
        self.r2_home_dist = bfs_distance_map(AGENT2_HOME_X, AGENT2_HOME_Y, self.obstacle_positions,
                                             target_is_walkable=True)

        self.queue = SharedTargetQueue(self.shelves, self._gc)
        self.a1_policy = QTablePolicy()

        self.a2_net = DeepQNetwork(self.OBS_SIZE, 6).to(compute_device)
        self._load_dqn(self.a2_net, AGENT2_MODEL_PATH)

        from robot import Robot
        self.robot1 = Robot(start_x=AGENT1_HOME_X, start_y=AGENT1_HOME_Y)
        self.robot2 = Robot(start_x=AGENT2_HOME_X, start_y=AGENT2_HOME_Y)
        self.robot1.loaded = False
        self.robot2.loaded = False

        self._r1_phase = PHASE_HOMING
        self._r2_phase = PHASE_HOMING
        self._r1_target_gx = AGENT1_HOME_X
        self._r1_target_gy = AGENT1_HOME_Y
        self._r2_target_gx = AGENT2_HOME_X
        self._r2_target_gy = AGENT2_HOME_Y
        self._r1_target_dist = self.r1_home_dist
        self._r2_target_dist = self.r2_home_dist
        self._r1_direction = "up"
        self._r2_last_action = -1
        self._r2_just_picked_up = False
        self._r2_just_delivered = False

        self.r1_score = 0
        self.r2_score = 0
        self.collision_count = 0
        self._step = 0

        self.screen = None
        self.clock = None
        self.fps = SIMULATION_FPS
        self.robot_label_font = None

    def _gc(self, obj):
        return (round((obj.x - PADDING_BORDER) / GRID_SPACING),
                round((obj.y - PADDING_BORDER) / GRID_SPACING))

    def _load_dqn(self, net, path):
        try:
            ckpt = torch.load(path, map_location=compute_device)
            weights = ckpt.get("policy", ckpt) if isinstance(ckpt, dict) else ckpt
            net.load_state_dict(weights)
            net.eval()
            print(f"  ✅ Agent 2 DQN loaded from '{path}'")
        except FileNotFoundError:
            print(f"  ❌ '{path}' not found — Agent 2 uses random weights.")

    def _r1_qtable_obs(self) -> np.ndarray:
        r1 = self.robot1
        return np.array([
            r1.grid_x, r1.grid_y, float(r1.loaded),
            self.a1_policy.current_shelf_target_x, self.a1_policy.current_shelf_target_y,
            self.r1_dropoff_gx, self.r1_dropoff_gy,
            float(_DIR_INT[self._r1_direction]),
        ], dtype=np.float32)

    def _r2_obs(self) -> np.ndarray:
        r2, r1 = self.robot2, self.robot1

        if r2.loaded:
            nav_x, nav_y = self.r2_dropoff_gx, self.r2_dropoff_gy
        elif self._r2_phase == PHASE_HOMING:
            nav_x, nav_y = AGENT2_HOME_X, AGENT2_HOME_Y
        else:
            nav_x, nav_y = self._r2_target_gx, self._r2_target_gy

        lx, ly = 0.0, 0.0
        la = self._r2_last_action
        if la == 0:
            ly = -1.0
        elif la == 1:
            ly = 1.0
        elif la == 2:
            lx = -1.0
        elif la == 3:
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

        r1nx, r1ny = self._r1_next_pos_preview()
        extra = [
            (r1.grid_x - r2.grid_x) / GRID_WIDTH, (r1.grid_y - r2.grid_y) / GRID_HEIGHT,
            float(r1.loaded), float(self._r1_phase == PHASE_HOMING),
            (r1nx - r2.grid_x) / GRID_WIDTH, (r1ny - r2.grid_y) / GRID_HEIGHT,
        ]
        return np.array(base + extra, dtype=np.float32)

    def _r1_action(self) -> int:
        r1 = self.robot1
        if self._r1_phase == PHASE_HOMING:
            if r1.grid_x == AGENT1_HOME_X and r1.grid_y == AGENT1_HOME_Y:
                return 5
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
        if action == 5:
            a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
            return a if a is not None else 0

        if action < 4:
            cur_d = self._r1_target_dist.get((r1.grid_x, r1.grid_y), 999)
            ddx, ddy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            prop_d = self._r1_target_dist.get((r1.grid_x + ddx, r1.grid_y + ddy), 999)
            if prop_d >= cur_d:
                a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
                return a if a is not None else action
        return action

    def _r1_next_pos_preview(self):
        return predict_next(self.robot1.grid_x, self.robot1.grid_y, self._r1_action(), self.obstacle_positions)

    def _dispatch(self):
        for robot_id, robot, other_robot, phase_attr, tgx_attr, tgy_attr, tdist_attr, home_x, home_y in [
            (1, self.robot1, self.robot2, '_r1_phase', '_r1_target_gx', '_r1_target_gy', '_r1_target_dist',
             AGENT1_HOME_X, AGENT1_HOME_Y),
            (2, self.robot2, self.robot1, '_r2_phase', '_r2_target_gx', '_r2_target_gy', '_r2_target_dist',
             AGENT2_HOME_X, AGENT2_HOME_Y),
        ]:
            if getattr(self, phase_attr) == PHASE_HOMING and robot.grid_x == home_x and robot.grid_y == home_y:
                result = self.queue.try_assign(robot_id, (other_robot.grid_x, other_robot.grid_y))
                if result:
                    gx, gy = result
                    setattr(self, tgx_attr, gx)
                    setattr(self, tgy_attr, gy)
                    setattr(self, tdist_attr, bfs_distance_map(gx, gy, self.obstacle_positions))
                    setattr(self, phase_attr, PHASE_FETCHING)
                    if robot_id == 1:
                        self.a1_policy.current_shelf_target_x = gx
                        self.a1_policy.current_shelf_target_y = gy

    def step(self):
        self._step += 1
        r1, r2 = self.robot1, self.robot2

        self._dispatch()

        a1 = self._r1_action()
        r1nx, r1ny = predict_next(r1.grid_x, r1.grid_y, a1, self.obstacle_positions)

        with torch.no_grad():
            a2 = self.a2_net(
                torch.as_tensor(self._r2_obs(), dtype=torch.float32, device=compute_device).unsqueeze(0)
            ).argmax().item()

        r2nx, r2ny = predict_next(r2.grid_x, r2.grid_y, a2, self.obstacle_positions)

        r2_into_r1 = (r2nx == r1nx and r2ny == r1ny)
        r1_into_r2 = (r1nx == r2.grid_x and r1ny == r2.grid_y)
        collision = r2_into_r1 or r1_into_r2

        if a1 < 4 and (r1nx, r1ny) not in self.obstacle_positions:
            r1.grid_x, r1.grid_y = r1nx, r1ny
        if a1 < 4:
            self._r1_direction = _ACTION_TO_DIR[a1]

        if collision:
            self.collision_count += 1
            eff_a2 = 5
        else:
            eff_a2 = a2
            if a2 < 4:
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][a2]
                nx, ny = r2.grid_x + dx, r2.grid_y + dy
                not_r1 = not (nx == r1.grid_x and ny == r1.grid_y)
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx,
                                                                        ny) not in self.obstacle_positions and not_r1):
                    r2.grid_x, r2.grid_y = nx, ny
            elif a2 == 4:
                self._r2_try_interact()

        self._r2_last_action = eff_a2
        self._update_r1_phase(a1)
        self._update_r2_phase(eff_a2)

        if LOG_EVERY_STEP:
            print(f"Step: {self._step:04d} | R1: {ACTION_NAMES[a1]:<8} | R2: {ACTION_NAMES[eff_a2]:<8}")

    def _update_r1_phase(self, action):
        r1 = self.robot1
        if self._r1_phase == PHASE_HOMING:
            return

        if self._r1_phase == PHASE_FETCHING:
            if (action == 4 and not r1.loaded and abs(r1.grid_x - self._r1_target_gx) + abs(
                    r1.grid_y - self._r1_target_gy) == 1):
                r1.loaded = True
                for s in self.shelves:
                    if self._gc(s) == (self._r1_target_gx, self._r1_target_gy):
                        s.has_box = False;
                        s.image = s.empty_image;
                        break
                self._r1_phase = PHASE_DELIVERING
            return

        if self._r1_phase == PHASE_DELIVERING:
            if (action == 4 and r1.loaded and abs(r1.grid_x - self.r1_dropoff_gx) + abs(
                    r1.grid_y - self.r1_dropoff_gy) == 1):
                r1.loaded = False
                self.r1_score += 1
                self.queue.release(robot_id=1)
                self._r1_phase = PHASE_HOMING
                self._r1_target_gx = AGENT1_HOME_X
                self._r1_target_gy = AGENT1_HOME_Y
                self._r1_target_dist = self.r1_home_dist

    def _update_r2_phase(self, action):
        r2 = self.robot2
        if self._r2_phase == PHASE_HOMING:
            return

        if self._r2_phase == PHASE_FETCHING:
            if self._r2_just_picked_up:
                self._r2_just_picked_up = False
                self._r2_phase = PHASE_DELIVERING
            return

        if self._r2_phase == PHASE_DELIVERING:
            if self._r2_just_delivered:
                self._r2_just_delivered = False
                self.queue.release(robot_id=2)
                self._r2_phase = PHASE_HOMING
                self._r2_target_gx = AGENT2_HOME_X
                self._r2_target_gy = AGENT2_HOME_Y
                self._r2_target_dist = self.r2_home_dist

    def _r2_try_interact(self):
        r2 = self.robot2
        if self._r2_phase == PHASE_FETCHING and not r2.loaded:
            if abs(r2.grid_x - self._r2_target_gx) + abs(r2.grid_y - self._r2_target_gy) == 1:
                r2.loaded = True
                self._r2_just_picked_up = True
                for s in self.shelves:
                    if self._gc(s) == (self._r2_target_gx, self._r2_target_gy):
                        s.has_box = False;
                        s.image = s.empty_image;
                        break
            return

        if self._r2_phase == PHASE_DELIVERING and r2.loaded:
            if abs(r2.grid_x - self.r2_dropoff_gx) + abs(r2.grid_y - self.r2_dropoff_gy) == 1:
                r2.loaded = False
                self.r2_score += 1
                self._r2_just_delivered = True

    def _ensure_screen(self):
        if self.screen is not None:
            return
        pygame.init()
        w = GRID_WIDTH * GRID_SPACING + 2 * PADDING_BORDER
        h = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Warehouse Visual Evaluation Panel")
        self.clock = pygame.time.Clock()
        self.robot_label_font = pygame.font.SysFont("monospace", 16, bold=True)

    def render(self):
        self._ensure_screen()
        self.screen.fill((240, 240, 240))

        color_r1 = (0, 255, 255)
        color_r2 = (255, 255, 0)

        for cs in self.charge_stations:
            self.screen.blit(cs.image, (cs.x, cs.y))
        for dp in self.dropoff_platforms:
            self.screen.blit(dp.image, (dp.x, dp.y))

        r1dx, r1dy = PADDING_BORDER + self.r1_dropoff_gx * GRID_SPACING, PADDING_BORDER + self.r1_dropoff_gy * GRID_SPACING
        pygame.draw.rect(self.screen, color_r1, (r1dx, r1dy, TILE_SIZE, TILE_SIZE), width=2)
        r2dx, r2dy = PADDING_BORDER + self.r2_dropoff_gx * GRID_SPACING, PADDING_BORDER + self.r2_dropoff_gy * GRID_SPACING
        pygame.draw.rect(self.screen, color_r2, (r2dx, r2dy, TILE_SIZE, TILE_SIZE), width=2)

        r1, r2 = self.robot1, self.robot2
        for shelf in self.shelves:
            gx, gy = self._gc(shelf)
            is_r1_tgt = (self._r1_phase != PHASE_HOMING and not r1.loaded and (gx, gy) == (self._r1_target_gx,
                                                                                           self._r1_target_gy))
            is_r2_tgt = (self._r2_phase != PHASE_HOMING and not r2.loaded and (gx, gy) == (self._r2_target_gx,
                                                                                           self._r2_target_gy))

            self.screen.blit(shelf.shadow_image, (shelf.x - 1, shelf.y + 4))

            if is_r1_tgt:
                pygame.draw.rect(self.screen, color_r1, (shelf.x - 2, shelf.y - 2, TILE_SIZE + 4, TILE_SIZE + 4), 2)
            if is_r2_tgt:
                pygame.draw.rect(self.screen, color_r2, (shelf.x - 2, shelf.y - 2, TILE_SIZE + 4, TILE_SIZE + 4), 2)

            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        cx = (TILE_SIZE - ROBOT_WIDTH) // 2
        cy = (TILE_SIZE - ROBOT_HEIGHT) // 2

        a1px = PADDING_BORDER + r1.grid_x * GRID_SPACING
        a1py = PADDING_BORDER + r1.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, color_r1, (a1px, a1py, TILE_SIZE, TILE_SIZE), 2)
        self.screen.blit(ROBOT_IMAGE_VERTICAL_BOX if r1.loaded else ROBOT_IMAGE_VERTICAL, (a1px + cx, a1py + cy))

        a2px = PADDING_BORDER + r2.grid_x * GRID_SPACING
        a2py = PADDING_BORDER + r2.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, color_r2, (a2px, a2py, TILE_SIZE, TILE_SIZE), 2)
        self.screen.blit(ROBOT_IMAGE_SIDE_BOX if r2.loaded else ROBOT_IMAGE_SIDE, (a2px + cx, a2py + cy))

        agent_one_label = self.robot_label_font.render("R1", True, color_r1)
        agent_two_label = self.robot_label_font.render("R2", True, color_r2)

        self.screen.blit(agent_one_label, (a1px + 2, a1py - 18))
        self.screen.blit(agent_two_label, (a2px + 2, a2py - 18))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.fps = min(self.fps + 5, 120)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.fps = max(self.fps - 5, 1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                cgx = round((mx - PADDING_BORDER) / GRID_SPACING)
                cgy = round((my - PADDING_BORDER) / GRID_SPACING)
                for shelf in self.shelves:
                    if self._gc(shelf) == (cgx, cgy):
                        self.queue.enqueue(cgx, cgy)
                        break
        return True


def main():
    env = TwoAgentTestingEnv()
    env.render()

    try:
        while True:
            if not env.handle_events():
                break
            env.step()
            env.render()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()