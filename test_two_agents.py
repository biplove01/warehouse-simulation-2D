"""
test_two_agents.py — Interactive visualisation of trained two-agent system.

Agent 1  (cyan border)   — Q-table policy, BFS navigation, same logic as training.
Agent 2  (yellow border) — Trained DQN policy loaded from checkpoints_agent2/.

Click a shelf to queue it.  Both agents share the queue.
Press + / - to change speed.  Close window to quit.
"""

import torch
import pygame
import numpy as np
import random
from collections import deque
import os
import pickle

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from train import QNetwork
from dual_q_learning_agent import DualQAgent
from constants import *

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

AGENT1_QTABLE_FOLDER  = "checkpoints"
AGENT1_QTABLE_FILE    = "warehouse_data.pkl"
AGENT2_MODEL_PATH     = "checkpoints_agent2/best_model.pt"

AGENT1_HOME_X = ROBOT_HOME_GRID_X   # 2
AGENT1_HOME_Y = ROBOT_HOME_GRID_Y   # 0
AGENT2_HOME_X = 0
AGENT2_HOME_Y = 0

SIMULATION_FPS = 6

# Per-step console logging — set False to silence
LOG_EVERY_STEP   = True     # print one line per step
LOG_COLLISIONS   = True     # warn when collision blocked
LOG_DELIVERY     = True     # print on each delivery
LOG_DISPATCH     = True     # print when task is assigned

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Interact", "Wait"]

PHASE_FETCHING   = "fetching"
PHASE_DELIVERING = "delivering"
PHASE_HOMING     = "homing"

compute_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ─── BFS UTILITIES ────────────────────────────────────────────────────────────

def bfs_distance_map(start_gx, start_gy, obstacle_positions,
                     target_is_walkable=False):
    dist_map = {}
    queue = deque()
    if target_is_walkable:
        dist_map[(start_gx, start_gy)] = 0
        queue.append((start_gx, start_gy, 0))
    else:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = start_gx+dx, start_gy+dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx,ny) not in obstacle_positions
                    and (nx,ny) not in dist_map):
                dist_map[(nx,ny)] = 0
                queue.append((nx,ny,0))
    while queue:
        cx, cy, cd = queue.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx+dx, cy+dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx,ny) not in obstacle_positions
                    and (nx,ny) not in dist_map):
                dist_map[(nx,ny)] = cd+1
                queue.append((nx,ny,cd+1))
    return dist_map


def bfs_best_action(gx, gy, dist_map):
    best_a, best_d = None, dist_map.get((gx, gy), 999)
    for i, (dx, dy) in enumerate([(0,-1),(0,1),(-1,0),(1,0)]):
        d = dist_map.get((gx+dx, gy+dy), 999)
        if d < best_d:
            best_d = d
            best_a = i
    return best_a


def predict_next(gx, gy, action, obstacle_positions):
    if action >= 4:
        return gx, gy
    dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
    nx, ny = gx+dx, gy+dy
    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
            and (nx,ny) not in obstacle_positions):
        return nx, ny
    return gx, gy


# ─── Q-TABLE POLICY (exact same wrapper as training) ─────────────────────────

_DIR_INT       = {"up": 0, "down": 1, "left": 2, "right": 3}
_ACTION_TO_DIR = {0: "up", 1: "down", 2: "left", 3: "right"}


class QTablePolicy:
    """Wraps DualQAgent for greedy (epsilon=0) action selection."""

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
            return 5   # unseen → BFS fallback in caller
        return int(np.argmax(self.agent.q_table[state]))


# ─── SHARED TARGET QUEUE ──────────────────────────────────────────────────────

class SharedTargetQueue:
    """Click-driven queue. Boxes appear immediately on click."""

    def __init__(self, shelves, gc_fn):
        self._shelves = shelves
        self._gc = gc_fn
        self._queue: deque = deque()
        self._assignments = {}   # robot_id → (gx, gy)

    def enqueue(self, gx, gy):
        if (gx, gy) in self._queue or (gx, gy) in self._assignments.values():
            print(f"  ⚠️  ({gx},{gy}) already queued or assigned.")
            return
        for s in self._shelves:
            if self._gc(s) == (gx, gy):
                s.has_box = True
                s.image = s.loaded_image
                break
        self._queue.append((gx, gy))
        print(f"  📦 Shelf ({gx},{gy}) queued. Depth: {len(self._queue)}")

    def try_assign(self, robot_id, other_robot_pos):
        """Assign next task to robot_id if available and unblocked."""
        if not self._queue:
            return None
        gx, gy = self._queue[0]
        if other_robot_pos == (gx, gy):
            return None   # other robot is sitting on the target — retry next step
        self._queue.popleft()
        self._assignments[robot_id] = (gx, gy)
        return gx, gy

    def release(self, robot_id):
        self._assignments.pop(robot_id, None)

    @property
    def depth(self):
        return len(self._queue)


# ─── TWO-AGENT TESTING ENVIRONMENT ───────────────────────────────────────────

class TwoAgentTestingEnv:
    """
    Exact same logic as TwoAgentTrainingEnv but:
    - Infinite episodes (no step limit).
    - Tasks come from click-driven SharedTargetQueue.
    - Full pygame render with per-step console logging.
    - Agent 2 uses trained DQN. Agent 1 uses Q-table + BFS (identical to training).
    """

    OBS_SIZE = 25

    def __init__(self):
        # One shared WarehouseEnv for the world — both agents use it.
        self._base = WarehouseEnv(render_mode=None)
        self._base.reset()

        self.obstacle_positions = self._base.obstacle_positions
        self.shelves            = self._base.shelves
        self.dropoff_platforms  = self._base.dropoff_platforms
        self.charge_stations    = self._base.charge_stations

        central = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_gx = round((central.x - PADDING_BORDER) / GRID_SPACING)
        self.dropoff_gy = round((central.y - PADDING_BORDER) / GRID_SPACING)

        self.dropoff_dist  = bfs_distance_map(self.dropoff_gx, self.dropoff_gy,
                                               self.obstacle_positions)
        self.r1_home_dist  = bfs_distance_map(AGENT1_HOME_X, AGENT1_HOME_Y,
                                               self.obstacle_positions,
                                               target_is_walkable=True)
        self.r2_home_dist  = bfs_distance_map(AGENT2_HOME_X, AGENT2_HOME_Y,
                                               self.obstacle_positions,
                                               target_is_walkable=True)

        # Shared queue.
        self.queue = SharedTargetQueue(self.shelves, self._gc)

        # Agent 1 policy — identical to training.
        self.a1_policy = QTablePolicy()

        # Agent 2 policy — trained DQN.
        self.a2_net = QNetwork(self.OBS_SIZE, 6).to(compute_device)
        self._load_dqn(self.a2_net, AGENT2_MODEL_PATH)

        # Robot objects (from robot.py — same as training).
        from robot import Robot
        self.robot1 = Robot(start_x=AGENT1_HOME_X, start_y=AGENT1_HOME_Y)
        self.robot2 = Robot(start_x=AGENT2_HOME_X, start_y=AGENT2_HOME_Y)
        self.robot1.loaded = False
        self.robot2.loaded = False

        # Phase state — mirrors training exactly.
        self._r1_phase           = PHASE_HOMING
        self._r2_phase           = PHASE_HOMING
        self._r1_target_gx       = AGENT1_HOME_X
        self._r1_target_gy       = AGENT1_HOME_Y
        self._r2_target_gx       = AGENT2_HOME_X
        self._r2_target_gy       = AGENT2_HOME_Y
        self._r1_target_dist     = self.r1_home_dist
        self._r2_target_dist     = self.r2_home_dist
        self._r1_direction       = "up"
        self._r2_last_action     = -1
        self._r2_just_picked_up  = False
        self._r2_just_delivered  = False

        # Scores / counters.
        self.r1_score     = 0
        self.r2_score     = 0
        self.collision_count = 0
        self._r1_unseen   = 0
        self._step        = 0

        # Pygame.
        self.screen    = None
        self.clock     = None
        self._hud_font = None
        self.fps       = SIMULATION_FPS

    # ── helpers ───────────────────────────────────────────────────────────────

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

    # ── Q-table obs (8 raw-int features, identical to training) ──────────────

    def _r1_qtable_obs(self) -> np.ndarray:
        r1 = self.robot1
        return np.array([
            r1.grid_x,
            r1.grid_y,
            float(r1.loaded),
            self.a1_policy.current_shelf_target_x,
            self.a1_policy.current_shelf_target_y,
            self.dropoff_gx,
            self.dropoff_gy,
            float(_DIR_INT[self._r1_direction]),
        ], dtype=np.float32)

    # ── Agent 2 obs (25 normalised features, identical to training) ───────────

    def _r2_obs(self) -> np.ndarray:
        r2, r1 = self.robot2, self.robot1

        if r2.loaded:
            nav_x, nav_y = self.dropoff_gx, self.dropoff_gy
        elif self._r2_phase == PHASE_HOMING:
            nav_x, nav_y = AGENT2_HOME_X, AGENT2_HOME_Y
        else:
            nav_x, nav_y = self._r2_target_gx, self._r2_target_gy

        lx, ly = 0.0, 0.0
        la = self._r2_last_action
        if la == 0:    ly = -1.0
        elif la == 1:  ly =  1.0
        elif la == 2:  lx = -1.0
        elif la == 3:  lx =  1.0

        can_pickup = float(
            not r2.loaded and self._r2_phase == PHASE_FETCHING
            and abs(r2.grid_x - self._r2_target_gx)
              + abs(r2.grid_y - self._r2_target_gy) == 1
        )
        can_deliver = float(
            r2.loaded
            and abs(r2.grid_x - self.dropoff_gx)
              + abs(r2.grid_y - self.dropoff_gy) == 1
        )

        base = [
            r2.grid_x / GRID_WIDTH,
            r2.grid_y / GRID_HEIGHT,
            float(r2.loaded),
            (nav_x - r2.grid_x) / GRID_WIDTH,
            (nav_y - r2.grid_y) / GRID_HEIGHT,
            (self.dropoff_gx - r2.grid_x) / GRID_WIDTH,
            (self.dropoff_gy - r2.grid_y) / GRID_HEIGHT,
        ]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = r2.grid_x+dx, r2.grid_y+dy
            oob   = not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT)
            wall  = (nx,ny) in self.obstacle_positions
            is_r1 = (nx == r1.grid_x and ny == r1.grid_y)
            base.append(1.0 if (oob or wall or is_r1) else 0.0)
        base += [lx, ly, can_pickup, can_deliver]

        r1nx, r1ny = self._r1_next_pos_preview()
        extra = [
            (r1.grid_x - r2.grid_x) / GRID_WIDTH,
            (r1.grid_y - r2.grid_y) / GRID_HEIGHT,
            float(r1.loaded),
            float(self._r1_phase == PHASE_HOMING),
            (r1nx - r2.grid_x) / GRID_WIDTH,
            (r1ny - r2.grid_y) / GRID_HEIGHT,
        ]
        return np.array(base + extra, dtype=np.float32)

    # ── Agent 1 action (identical to training) ────────────────────────────────

    def _r1_action(self) -> int:
        r1 = self.robot1

        if self._r1_phase == PHASE_HOMING:
            if r1.grid_x == AGENT1_HOME_X and r1.grid_y == AGENT1_HOME_Y:
                return 5   # wait — dispatcher will assign task
            a = bfs_best_action(r1.grid_x, r1.grid_y, self.r1_home_dist)
            return a if a is not None else 5

        if self._r1_phase == PHASE_DELIVERING:
            if (abs(r1.grid_x - self.dropoff_gx)
              + abs(r1.grid_y - self.dropoff_gy) == 1):
                return 4
            a = bfs_best_action(r1.grid_x, r1.grid_y, self.dropoff_dist)
            return a if a is not None else 5

        # FETCHING: adjacent → interact immediately, then Q-table with BFS override.
        if (not r1.loaded
                and abs(r1.grid_x - self._r1_target_gx)
                  + abs(r1.grid_y - self._r1_target_gy) == 1):
            return 4

        action = self.a1_policy.select_action(self._r1_qtable_obs())

        if action == 5:
            self._r1_unseen += 1
            a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
            return a if a is not None else 0

        # Override if Q-table action makes no BFS progress.
        if action < 4:
            cur_d = self._r1_target_dist.get((r1.grid_x, r1.grid_y), 999)
            ddx, ddy = [(0,-1),(0,1),(-1,0),(1,0)][action]
            prop_d = self._r1_target_dist.get((r1.grid_x+ddx, r1.grid_y+ddy), 999)
            if prop_d >= cur_d:
                self._r1_unseen += 1
                a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
                return a if a is not None else action

        return action

    def _r1_next_pos_preview(self):
        return predict_next(self.robot1.grid_x, self.robot1.grid_y,
                            self._r1_action(), self.obstacle_positions)

    # ── dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(self):
        """Try to assign queued tasks to idle robots. R1 gets priority."""
        for robot_id, robot, other_robot, phase_attr, tgx_attr, tgy_attr, tdist_attr, phase_val, home_x, home_y, home_dist, policy_target in [
            (1, self.robot1, self.robot2, '_r1_phase', '_r1_target_gx', '_r1_target_gy', '_r1_target_dist', PHASE_HOMING, AGENT1_HOME_X, AGENT1_HOME_Y, self.r1_home_dist, True),
            (2, self.robot2, self.robot1, '_r2_phase', '_r2_target_gx', '_r2_target_gy', '_r2_target_dist', PHASE_HOMING, AGENT2_HOME_X, AGENT2_HOME_Y, self.r2_home_dist, False),
        ]:
            if getattr(self, phase_attr) == PHASE_HOMING and robot.grid_x == (AGENT1_HOME_X if robot_id == 1 else AGENT2_HOME_X) and robot.grid_y == (AGENT1_HOME_Y if robot_id == 1 else AGENT2_HOME_Y):
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
                    if LOG_DISPATCH:
                        label = "A1" if robot_id == 1 else "A2"
                        print(f"  📋 [{label}] assigned shelf ({gx},{gy})")

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self):
        self._step += 1
        r1, r2 = self.robot1, self.robot2

        self._dispatch()

        # Agent 1 action.
        a1 = self._r1_action()
        r1nx, r1ny = predict_next(r1.grid_x, r1.grid_y, a1, self.obstacle_positions)

        # Agent 2 action from DQN.
        with torch.no_grad():
            a2 = self.a2_net(
                torch.as_tensor(self._r2_obs(), dtype=torch.float32,
                                device=compute_device).unsqueeze(0)
            ).argmax().item()

        r2nx, r2ny = predict_next(r2.grid_x, r2.grid_y, a2, self.obstacle_positions)

        # Collision detection.
        r2_into_r1 = (r2nx == r1nx and r2ny == r1ny)
        r1_into_r2 = (r1nx == r2.grid_x and r1ny == r2.grid_y)
        collision  = r2_into_r1 or r1_into_r2

        # Move R1 freely.
        if a1 < 4 and (r1nx, r1ny) not in self.obstacle_positions:
            r1.grid_x, r1.grid_y = r1nx, r1ny
        if a1 < 4:
            self._r1_direction = _ACTION_TO_DIR[a1]

        if collision:
            self.collision_count += 1
            if LOG_COLLISIONS:
                print(
                    f"  💥 [step {self._step}] COLLISION — "
                    f"R1 at ({r1.grid_x},{r1.grid_y}) / R2 at ({r2.grid_x},{r2.grid_y}) "
                    f"| R2 forced WAIT"
                )
            eff_a2 = 5   # R2 stays put
        else:
            eff_a2 = a2
            if a2 < 4:
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][a2]
                nx, ny = r2.grid_x + dx, r2.grid_y + dy
                not_r1 = not (nx == r1.grid_x and ny == r1.grid_y)
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                        and (nx, ny) not in self.obstacle_positions and not_r1):
                    r2.grid_x, r2.grid_y = nx, ny
            elif a2 == 4:
                self._r2_try_interact()  # ← THIS WAS MISSING

        self._r2_last_action = eff_a2

        # Console log per step.
        if LOG_EVERY_STEP:
            a2_name = ACTION_NAMES[eff_a2] + ("(BLOCK)" if collision else "")
            print(
                f"  [s{self._step:5d}] "
                f"R1 {self._r1_phase:10s} ({r1.grid_x:2d},{r1.grid_y:2d}) ld={int(r1.loaded)} "
                f"→ {ACTION_NAMES[a1]:<9s} | "
                f"R2 {self._r2_phase:10s} ({r2.grid_x:2d},{r2.grid_y:2d}) ld={int(r2.loaded)} "
                f"→ {a2_name:<15s} "
                f"q={self.queue.depth} sc1={self.r1_score} sc2={self.r2_score}"
            )

        # Update phase machines.
        self._update_r1_phase(a1)
        self._update_r2_phase(eff_a2)

    # ── phase machines (identical to training) ─────────────────────────────────

    def _update_r1_phase(self, action):
        r1 = self.robot1

        if self._r1_phase == PHASE_HOMING:
            # Dispatch handles task assignment — nothing to do here.
            return

        if self._r1_phase == PHASE_FETCHING:
            if (action == 4 and not r1.loaded
                    and abs(r1.grid_x - self._r1_target_gx)
                      + abs(r1.grid_y - self._r1_target_gy) == 1):
                r1.loaded = True
                for s in self.shelves:
                    if self._gc(s) == (self._r1_target_gx, self._r1_target_gy):
                        s.has_box = False; s.image = s.empty_image; break
                self._r1_phase = PHASE_DELIVERING
            return

        if self._r1_phase == PHASE_DELIVERING:
            if (action == 4 and r1.loaded
                    and abs(r1.grid_x - self.dropoff_gx)
                      + abs(r1.grid_y - self.dropoff_gy) == 1):
                r1.loaded = False
                self.r1_score += 1
                self.queue.release(robot_id=1)
                if LOG_DELIVERY:
                    print(f"  ✅ [R1] delivery #{self.r1_score} at ({r1.grid_x},{r1.grid_y})")
                self._r1_phase      = PHASE_HOMING
                self._r1_target_gx  = AGENT1_HOME_X
                self._r1_target_gy  = AGENT1_HOME_Y
                self._r1_target_dist = self.r1_home_dist

    def _update_r2_phase(self, action):
        r2 = self.robot2

        if self._r2_phase == PHASE_HOMING:
            # Dispatch handles task assignment.
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
                if LOG_DELIVERY:
                    print(f"  ✅ [R2] delivery #{self.r2_score} at ({r2.grid_x},{r2.grid_y})")
                self._r2_phase      = PHASE_HOMING
                self._r2_target_gx  = AGENT2_HOME_X
                self._r2_target_gy  = AGENT2_HOME_Y
                self._r2_target_dist = self.r2_home_dist

    # ── R2 interact logic (sets flags used by phase machine) ─────────────────

    def _r2_try_interact(self):
        """Call this when R2 takes action 4. Sets pickup/delivery flags."""
        r2 = self.robot2

        if self._r2_phase == PHASE_FETCHING and not r2.loaded:
            if (abs(r2.grid_x - self._r2_target_gx)
              + abs(r2.grid_y - self._r2_target_gy) == 1):
                r2.loaded = True
                self._r2_just_picked_up = True
                for s in self.shelves:
                    if self._gc(s) == (self._r2_target_gx, self._r2_target_gy):
                        s.has_box = False; s.image = s.empty_image; break
            return

        if self._r2_phase == PHASE_DELIVERING and r2.loaded:
            if (abs(r2.grid_x - self.dropoff_gx)
              + abs(r2.grid_y - self.dropoff_gy) == 1):
                r2.loaded = False
                self.r2_score += 1
                self._r2_just_delivered = True

    # ── render ────────────────────────────────────────────────────────────────

    def _ensure_screen(self):
        if self.screen is not None:
            return
        pygame.init()
        w = GRID_WIDTH  * GRID_SPACING + 2 * PADDING_BORDER
        h = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Warehouse — A1:Q-table(cyan)  A2:DQN(yellow)  +/- speed")
        self.clock     = pygame.time.Clock()
        self._hud_font = pygame.font.SysFont("monospace", 14)

    def render(self):
        self._ensure_screen()
        self.screen.fill((30, 30, 30))

        for cs in self.charge_stations:
            self.screen.blit(cs.image, (cs.x, cs.y))
        for dp in self.dropoff_platforms:
            self.screen.blit(dp.image, (dp.x, dp.y))

        r1, r2 = self.robot1, self.robot2
        for shelf in self.shelves:
            gx, gy = self._gc(shelf)
            is_r1_tgt = (self._r1_phase != PHASE_HOMING and not r1.loaded
                         and (gx, gy) == (self._r1_target_gx, self._r1_target_gy))
            is_r2_tgt = (self._r2_phase != PHASE_HOMING and not r2.loaded
                         and (gx, gy) == (self._r2_target_gx, self._r2_target_gy))
            self.screen.blit(shelf.shadow_image, (shelf.x-1, shelf.y+4))
            if is_r1_tgt:
                pygame.draw.rect(self.screen, (0,220,220),
                                 (shelf.x-2, shelf.y-2, TILE_SIZE+4, TILE_SIZE+4), 2)
            if is_r2_tgt:
                pygame.draw.rect(self.screen, (255,220,0),
                                 (shelf.x-4, shelf.y-4, TILE_SIZE+8, TILE_SIZE+8), 2)
            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        cx = (TILE_SIZE - ROBOT_WIDTH)  // 2
        cy = (TILE_SIZE - ROBOT_HEIGHT) // 2

        a1px = PADDING_BORDER + r1.grid_x * GRID_SPACING
        a1py = PADDING_BORDER + r1.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, (0,220,220), (a1px,a1py,TILE_SIZE,TILE_SIZE), 2)
        self.screen.blit(
            ROBOT_IMAGE_VERTICAL_BOX if r1.loaded else ROBOT_IMAGE_VERTICAL,
            (a1px+cx, a1py+cy)
        )

        a2px = PADDING_BORDER + r2.grid_x * GRID_SPACING
        a2py = PADDING_BORDER + r2.grid_y * GRID_SPACING
        pygame.draw.rect(self.screen, (255,220,0), (a2px,a2py,TILE_SIZE,TILE_SIZE), 2)
        self.screen.blit(
            ROBOT_IMAGE_SIDE_BOX if r2.loaded else ROBOT_IMAGE_SIDE,
            (a2px+cx, a2py+cy)
        )

        hud = [
            f"FPS:{self.fps} (+/-)",
            f"Queue:{self.queue.depth}  Collisions:{self.collision_count}",
            f"A1(cyan) Q-table  score:{self.r1_score}  phase:{self._r1_phase}  "
            f"pos:({r1.grid_x},{r1.grid_y})  unseen:{self._r1_unseen}",
            f"A2(yellow) DQN   score:{self.r2_score}  phase:{self._r2_phase}  "
            f"pos:({r2.grid_x},{r2.grid_y})  loaded:{int(r2.loaded)}",
        ]
        for i, line in enumerate(hud):
            self.screen.blit(
                self._hud_font.render(line, True, (255,255,255)),
                (8, 6 + i*17)
            )

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ── event / click handling ────────────────────────────────────────────────

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.fps = min(self.fps+1, 60)
                    print(f"  ⏩ FPS → {self.fps}")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.fps = max(self.fps-1, 1)
                    print(f"  ⏪ FPS → {self.fps}")
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                cgx = round((mx - PADDING_BORDER) / GRID_SPACING)
                cgy = round((my - PADDING_BORDER) / GRID_SPACING)
                for shelf in self.shelves:
                    if self._gc(shelf) == (cgx, cgy):
                        self.queue.enqueue(cgx, cgy)
                        break
        return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    env = TwoAgentTestingEnv()
    env.render()

    print("\n🖱️  Click shelves to queue tasks.")
    print(f"    A1 (cyan,   Q-table) home=({AGENT1_HOME_X},{AGENT1_HOME_Y})")
    print(f"    A2 (yellow, DQN)     home=({AGENT2_HOME_X},{AGENT2_HOME_Y})")
    print(f"    FPS={SIMULATION_FPS} — press +/- to adjust")
    print("    LOG_EVERY_STEP=True — one line per step in console\n")

    try:
        while True:
            if not env.handle_events():
                break
            env.step()
            env.render()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print(
            f"\nFinal — R1:{env.r1_score} deliveries | "
            f"R2:{env.r2_score} deliveries | "
            f"Collisions:{env.collision_count} | "
            f"R1 unseen states:{env._r1_unseen}"
        )
        pygame.quit()


if __name__ == "__main__":
    main()