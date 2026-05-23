import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pickle
from constants import *

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from train import QNetwork
from dual_q_learning_agent import DualQAgent
from two_agent_training_env import (
    AGENT_COLLISION_PENALTY,
    PROXIMITY_DISTANCE_1_PENALTY,
    PROXIMITY_DISTANCE_2_PENALTY,
    YIELDING_BONUS,
    YIELDING_BONUS_PROXIMITY_THRESHOLD,
)


if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

print(f"Training on: {compute_device}")


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

AGENT1_QTABLE_FOLDER  = "training_data"
AGENT1_QTABLE_FILE    = "warehouse_data.pkl"
AGENT2_CHECKPOINT_DIR = "checkpoints_agent2"

AGENT1_HOME_X = ROBOT_HOME_GRID_X
AGENT1_HOME_Y = ROBOT_HOME_GRID_Y

AGENT2_HOME_X = 0
AGENT2_HOME_Y = 0

HOME_WAIT_FRAMES   = 0     # assign task immediately on home arrival — no wasted steps
RANDOM_ACTION_PROB = 0.30
MAX_STEPS_PER_EPISODE = 2000   # R1 scores ~10/ep; R2 needs room to do the same

_DIR_INT       = {"up": 0, "down": 1, "left": 2, "right": 3}
_ACTION_TO_DIR = {0: "up", 1: "down", 2: "left", 3: "right"}

PHASE_FETCHING   = "fetching"
PHASE_DELIVERING = "delivering"
PHASE_HOMING     = "homing"


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


# ─── Q-TABLE POLICY ───────────────────────────────────────────────────────────

class QTablePolicy:
    """
    Passes the 8-element raw-integer obs directly to DualQAgent._get_state().
    Returns 5 (Wait) for unseen states so the caller triggers BFS fallback.
    """

    def __init__(self, agent: DualQAgent):
        self.agent = agent
        self.current_shelf_target_x: int = AGENT1_HOME_X
        self.current_shelf_target_y: int = AGENT1_HOME_Y

    def select_action(self, obs_8: np.ndarray) -> int:
        state = self.agent._get_state(obs_8)
        if state not in self.agent.q_table:
            return 5
        return int(np.argmax(self.agent.q_table[state]))


# ─── SHARED TARGET QUEUE ──────────────────────────────────────────────────────

class SharedTargetQueue:
    """
    Gives each robot its own independent shelf assignment.
    R1 and R2 never share a shelf target.
    """

    def __init__(self, shelves, to_grid_coords_fn):
        self._shelves = shelves
        self._gc = to_grid_coords_fn
        self._assignments = {}

    def reset(self):
        self._assignments.clear()
        for s in self._shelves:
            s.has_box = False
            s.image = s.empty_image

    def request_target(self, robot_id):
        claimed = set(self._assignments.values())
        available = [s for s in self._shelves if self._gc(s) not in claimed]
        if not available:
            return None
        chosen = random.choice(available)
        chosen.has_box = True
        chosen.image = chosen.loaded_image
        gx, gy = self._gc(chosen)
        self._assignments[robot_id] = (gx, gy)
        return gx, gy

    def release_target(self, robot_id):
        self._assignments.pop(robot_id, None)


# ─── TWO-AGENT TRAINING ENVIRONMENT ──────────────────────────────────────────

class TwoAgentTrainingEnv:

    OBS_SIZE = 25

    def __init__(self, render_mode=None):
        self._base = WarehouseEnv(render_mode=render_mode)
        self._base.reset()

        self.obstacle_positions = self._base.obstacle_positions
        self.shelves            = self._base.shelves
        self.dropoff_platforms  = self._base.dropoff_platforms
        self.charge_stations    = self._base.charge_stations

        central = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_gx = round((central.x - PADDING_BORDER) / GRID_SPACING)
        self.dropoff_gy = round((central.y - PADDING_BORDER) / GRID_SPACING)

        self.dropoff_dist = bfs_distance_map(self.dropoff_gx, self.dropoff_gy,
                                              self.obstacle_positions)
        self.r1_home_dist = bfs_distance_map(AGENT1_HOME_X, AGENT1_HOME_Y,
                                              self.obstacle_positions,
                                              target_is_walkable=True)
        self.r2_home_dist = bfs_distance_map(AGENT2_HOME_X, AGENT2_HOME_Y,
                                              self.obstacle_positions,
                                              target_is_walkable=True)

        self.shared_queue = SharedTargetQueue(self.shelves, self._gc)

        raw = DualQAgent(action_dim=6)
        raw.load_tables(AGENT1_QTABLE_FOLDER, AGENT1_QTABLE_FILE)
        raw.epsilon = 0.0
        self.agent1_policy = QTablePolicy(raw)
        print(f"  ✅ Q-table loaded: {len(raw.q_table):,} states")

        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.robot1 = None
        self.robot2 = None
        self._reset_state()

    def _reset_state(self):
        self._r1_phase        = PHASE_HOMING
        self._r2_phase        = PHASE_HOMING
        self._r1_home_frames  = 0
        self._r2_home_frames  = 0
        self._r1_target_gx    = AGENT1_HOME_X
        self._r1_target_gy    = AGENT1_HOME_Y
        self._r2_target_gx    = AGENT2_HOME_X
        self._r2_target_gy    = AGENT2_HOME_Y
        self._r1_target_dist  = self.r1_home_dist
        self._r2_target_dist  = self.r2_home_dist
        self._r1_direction    = "up"
        self._r2_last_action  = -1
        self.robot1_score     = 0
        self.robot2_score     = 0
        self.collision_count  = 0
        self.steps            = 0
        self._r1_unseen       = 0

    def _gc(self, obj):
        return (round((obj.x - PADDING_BORDER) / GRID_SPACING),
                round((obj.y - PADDING_BORDER) / GRID_SPACING))

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        from robot import Robot

        self.shared_queue.reset()
        self._reset_state()

        blocked = self.obstacle_positions | {self._gc(p) for p in self.dropoff_platforms}
        valid   = [(x,y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)
                   if (x,y) not in blocked]

        # Spawn both robots at their home stations.
        self.robot1 = Robot(start_x=AGENT1_HOME_X, start_y=AGENT1_HOME_Y)
        self.robot2 = Robot(start_x=AGENT2_HOME_X, start_y=AGENT2_HOME_Y)
        self.robot1.loaded = False
        self.robot2.loaded = False

        # Pre-assign tasks immediately — no homing phase wasted at episode start.
        # R1 gets priority (assigned first), R2 gets the next available shelf.
        r1_task = self.shared_queue.request_target(robot_id=1)
        if r1_task:
            gx, gy = r1_task
            self._r1_target_gx   = gx
            self._r1_target_gy   = gy
            self._r1_target_dist = bfs_distance_map(gx, gy, self.obstacle_positions)
            self.agent1_policy.current_shelf_target_x = gx
            self.agent1_policy.current_shelf_target_y = gy
            self._r1_phase = PHASE_FETCHING

        r2_task = self.shared_queue.request_target(robot_id=2)
        if r2_task:
            gx, gy = r2_task
            self._r2_target_gx   = gx
            self._r2_target_gy   = gy
            self._r2_target_dist = bfs_distance_map(gx, gy, self.obstacle_positions)
            self._r2_phase = PHASE_FETCHING

        return self._r2_obs(), {}

    # ── Q-table observation ────────────────────────────────────────────────────

    def _r1_qtable_obs(self) -> np.ndarray:
        r1 = self.robot1
        return np.array([
            r1.grid_x,
            r1.grid_y,
            float(r1.loaded),
            self.agent1_policy.current_shelf_target_x,
            self.agent1_policy.current_shelf_target_y,
            self.dropoff_gx,
            self.dropoff_gy,
            float(_DIR_INT[self._r1_direction]),
        ], dtype=np.float32)

    # ── Robot 2 observation ────────────────────────────────────────────────────

    def _r2_obs(self) -> np.ndarray:
        r2 = self.robot2
        r1 = self.robot1

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

        r1nx, r1ny = self._r1_next_pos()
        extra = [
            (r1.grid_x - r2.grid_x) / GRID_WIDTH,
            (r1.grid_y - r2.grid_y) / GRID_HEIGHT,
            float(r1.loaded),
            float(self._r1_phase == PHASE_HOMING),
            (r1nx - r2.grid_x) / GRID_WIDTH,
            (r1ny - r2.grid_y) / GRID_HEIGHT,
        ]
        return np.array(base + extra, dtype=np.float32)

    # ── Robot 1 action ─────────────────────────────────────────────────────────

    def _r1_action(self) -> int:
        r1 = self.robot1

        if self._r1_phase == PHASE_HOMING:
            if r1.grid_x == AGENT1_HOME_X and r1.grid_y == AGENT1_HOME_Y:
                return 5
            a = bfs_best_action(r1.grid_x, r1.grid_y, self.r1_home_dist)
            return a if a is not None else 5

        if self._r1_phase == PHASE_DELIVERING:
            if (abs(r1.grid_x - self.dropoff_gx)
              + abs(r1.grid_y - self.dropoff_gy) == 1):
                return 4
            a = bfs_best_action(r1.grid_x, r1.grid_y, self.dropoff_dist)
            return a if a is not None else 5

        # FETCHING: Q-table with BFS fallback
        action = self.agent1_policy.select_action(self._r1_qtable_obs())
        if action == 5:
            self._r1_unseen += 1
            a = bfs_best_action(r1.grid_x, r1.grid_y, self._r1_target_dist)
            if a is not None:
                return a

        if (not r1.loaded
                and abs(r1.grid_x - self._r1_target_gx)
                  + abs(r1.grid_y - self._r1_target_gy) == 1):
            return 4

        return action

    def _r1_next_pos(self):
        return predict_next(self.robot1.grid_x, self.robot1.grid_y,
                            self._r1_action(), self.obstacle_positions)

    # ── Robot 2 heuristic ─────────────────────────────────────────────────────

    def heuristic_action(self) -> int:
        r2 = self.robot2

        if self._r2_phase == PHASE_HOMING:
            a = bfs_best_action(r2.grid_x, r2.grid_y, self.r2_home_dist)
            return a if a is not None else random.randint(0, 3)

        if (self._r2_phase == PHASE_FETCHING and not r2.loaded
                and abs(r2.grid_x - self._r2_target_gx)
                  + abs(r2.grid_y - self._r2_target_gy) == 1):
            return 4

        if (r2.loaded
                and abs(r2.grid_x - self.dropoff_gx)
                  + abs(r2.grid_y - self.dropoff_gy) == 1):
            return 4

        dist_map = (self.dropoff_dist if self._r2_phase == PHASE_DELIVERING
                    else self._r2_target_dist)
        a = bfs_best_action(r2.grid_x, r2.grid_y, dist_map)
        return a if a is not None else random.randint(0, 3)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, r2_action: int):
        self.steps += 1
        r1, r2 = self.robot1, self.robot2

        a1 = self._r1_action()
        r1nx, r1ny = predict_next(r1.grid_x, r1.grid_y, a1, self.obstacle_positions)
        r2nx, r2ny = predict_next(r2.grid_x, r2.grid_y, r2_action, self.obstacle_positions)

        r2_into_r1 = (r2nx == r1nx and r2ny == r1ny)
        r1_into_r2 = (r1nx == r2.grid_x and r1ny == r2.grid_y)
        collision  = r2_into_r1 or r1_into_r2
        if collision:
            self.collision_count += 1

        if a1 < 4 and (r1nx, r1ny) not in self.obstacle_positions:
            r1.grid_x, r1.grid_y = r1nx, r1ny
        if a1 < 4:
            self._r1_direction = _ACTION_TO_DIR[a1]

        eff_a2  = (5 if (r2_into_r1 and r2_action < 4) else r2_action)
        r2_base = self._r2_move(eff_a2)
        self._r2_last_action = eff_a2

        r2_extra = AGENT_COLLISION_PENALTY if collision else 0.0
        if not collision:
            md = abs(r2.grid_x - r1.grid_x) + abs(r2.grid_y - r1.grid_y)
            if md == 1:   r2_extra += PROXIMITY_DISTANCE_1_PENALTY
            elif md == 2: r2_extra += PROXIMITY_DISTANCE_2_PENALTY
            if (r2_action in (4,5) and md <= YIELDING_BONUS_PROXIMITY_THRESHOLD
                    and r2_base > 0.0):
                r2_extra += YIELDING_BONUS

        self._update_r1_phase(a1)
        self._update_r2_phase(eff_a2)

        done = self.steps >= MAX_STEPS_PER_EPISODE
        return self._r2_obs(), r2_base + r2_extra, done, False, {}

    # ── Robot 2 movement ──────────────────────────────────────────────────────

    def _r2_move(self, action: int) -> float:
        r2 = self.robot2
        dist_map = (self.r2_home_dist if self._r2_phase == PHASE_HOMING
                    else self.dropoff_dist if r2.loaded
                    else self._r2_target_dist)

        dist_before = dist_map.get((r2.grid_x, r2.grid_y), 50)

        if action < 4:
            dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
            nx, ny = r2.grid_x+dx, r2.grid_y+dy
            ok = (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                  and (nx,ny) not in self.obstacle_positions
                  and not (nx == self.robot1.grid_x and ny == self.robot1.grid_y))
            if ok:
                r2.grid_x, r2.grid_y = nx, ny
                delta = dist_before - dist_map.get((nx,ny), 50)
                return delta * (1.0 if delta >= 0 else 4.0) - 0.05
            return -3.0

        if action == 4:
            if self._r2_phase == PHASE_FETCHING and not r2.loaded:
                if (abs(r2.grid_x-self._r2_target_gx)
                  + abs(r2.grid_y-self._r2_target_gy) == 1):
                    r2.loaded = True
                    for s in self.shelves:
                        if self._gc(s) == (self._r2_target_gx, self._r2_target_gy):
                            s.has_box = False; s.image = s.empty_image; break
                    return 10.0
                return -4.0
            if self._r2_phase == PHASE_DELIVERING and r2.loaded:
                if (abs(r2.grid_x-self.dropoff_gx)
                  + abs(r2.grid_y-self.dropoff_gy) == 1):
                    r2.loaded = False
                    self.robot2_score += 1
                    self.shared_queue.release_target(robot_id=2)
                    return 20.0
                return -4.0
            return -4.0

        return -0.05  # wait

    # ── Phase machines ─────────────────────────────────────────────────────────

    def _update_r1_phase(self, action: int):
        r1 = self.robot1

        if self._r1_phase == PHASE_HOMING:
            if r1.grid_x == AGENT1_HOME_X and r1.grid_y == AGENT1_HOME_Y:
                result = self.shared_queue.request_target(robot_id=1)
                if result:
                    gx, gy = result
                    self._r1_target_gx   = gx
                    self._r1_target_gy   = gy
                    self._r1_target_dist = bfs_distance_map(gx, gy,
                                            self.obstacle_positions)
                    self.agent1_policy.current_shelf_target_x = gx
                    self.agent1_policy.current_shelf_target_y = gy
                    self._r1_phase = PHASE_FETCHING
            return

        if self._r1_phase == PHASE_FETCHING:
            if (action == 4 and not r1.loaded
                    and abs(r1.grid_x-self._r1_target_gx)
                      + abs(r1.grid_y-self._r1_target_gy) == 1):
                r1.loaded = True
                for s in self.shelves:
                    if self._gc(s) == (self._r1_target_gx, self._r1_target_gy):
                        s.has_box = False; s.image = s.empty_image; break
                self._r1_phase = PHASE_DELIVERING
            return

        if self._r1_phase == PHASE_DELIVERING:
            if (action == 4 and r1.loaded
                    and abs(r1.grid_x-self.dropoff_gx)
                      + abs(r1.grid_y-self.dropoff_gy) == 1):
                r1.loaded = False
                self.robot1_score += 1
                self.shared_queue.release_target(robot_id=1)
                self._r1_phase       = PHASE_HOMING
                self._r1_home_frames = 0
                self._r1_target_gx   = AGENT1_HOME_X
                self._r1_target_gy   = AGENT1_HOME_Y
                self._r1_target_dist = self.r1_home_dist

    def _update_r2_phase(self, action: int):
        r2 = self.robot2

        if self._r2_phase == PHASE_HOMING:
            if r2.grid_x == AGENT2_HOME_X and r2.grid_y == AGENT2_HOME_Y:
                # Try to get a task every step while at home — no counter delay.
                result = self.shared_queue.request_target(robot_id=2)
                if result:
                    gx, gy = result
                    self._r2_target_gx   = gx
                    self._r2_target_gy   = gy
                    self._r2_target_dist = bfs_distance_map(gx, gy,
                                            self.obstacle_positions)
                    self._r2_phase = PHASE_FETCHING
                # if result is None, stay in HOMING and retry next step
            return

        if self._r2_phase == PHASE_FETCHING:
            if (action == 4 and not r2.loaded
                    and abs(r2.grid_x-self._r2_target_gx)
                      + abs(r2.grid_y-self._r2_target_gy) == 1):
                self._r2_phase = PHASE_DELIVERING
            return

        if self._r2_phase == PHASE_DELIVERING:
            if action == 4 and not r2.loaded:
                # After delivery, skip home — grab next task immediately if available.
                result = self.shared_queue.request_target(robot_id=2)
                if result:
                    gx, gy = result
                    self._r2_target_gx   = gx
                    self._r2_target_gy   = gy
                    self._r2_target_dist = bfs_distance_map(gx, gy,
                                            self.obstacle_positions)
                    self._r2_phase = PHASE_FETCHING
                else:
                    # No tasks right now — go home and wait
                    self._r2_phase       = PHASE_HOMING
                    self._r2_home_frames = 0
                    self._r2_target_gx   = AGENT2_HOME_X
                    self._r2_target_gy   = AGENT2_HOME_Y
                    self._r2_target_dist = self.r2_home_dist

    @property
    def score(self):
        return self.robot2_score

    @property
    def agent2_collision_count(self):
        return self.collision_count


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train():
    env = TwoAgentTrainingEnv()

    policy_net = QNetwork(env.OBS_SIZE, 6).to(compute_device)
    target_net = QNetwork(env.OBS_SIZE, 6).to(compute_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer    = deque(maxlen=50_000)

    batch_size  = 128
    gamma       = 0.98
    epsilon     = 0.7
    eps_min     = 0.1
    eps_decay   = 0.998
    sync_every  = 10
    save_every  = 50
    total_eps   = 2500

    os.makedirs(AGENT2_CHECKPOINT_DIR, exist_ok=True)
    best_score = -1

    for ep in range(total_eps):
        obs, _  = env.reset()
        ep_rew  = 0.0
        done    = False

        while not done:
            if random.random() < epsilon:
                action = (env.heuristic_action()
                          if random.random() > RANDOM_ACTION_PROB
                          else random.randint(0, 5))
            else:
                with torch.no_grad():
                    action = policy_net(
                        torch.as_tensor(obs, dtype=torch.float32,
                                        device=compute_device).unsqueeze(0)
                    ).argmax().item()

            next_obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((obs, action, rew, next_obs, float(done)))
            obs    = next_obs
            ep_rew += rew

            if len(buffer) > batch_size:
                s, a, r, ns, d = zip(*random.sample(buffer, batch_size))
                s_t  = torch.as_tensor(np.array(s),  dtype=torch.float32, device=compute_device)
                a_t  = torch.as_tensor(a, dtype=torch.long, device=compute_device).unsqueeze(1)
                r_t  = torch.as_tensor(r, dtype=torch.float32, device=compute_device).unsqueeze(1)
                ns_t = torch.as_tensor(np.array(ns), dtype=torch.float32, device=compute_device)
                d_t  = torch.as_tensor(d, dtype=torch.float32, device=compute_device).unsqueeze(1)

                cur_q  = policy_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    best_a = policy_net(ns_t).argmax(1, keepdim=True)
                    tgt_q  = r_t + gamma * target_net(ns_t).gather(1, best_a) * (1-d_t)
                loss = nn.MSELoss()(cur_q, tgt_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

        if ep % sync_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if env.score > best_score:
            best_score = env.score
            torch.save(policy_net.state_dict(),
                       os.path.join(AGENT2_CHECKPOINT_DIR, "best_model.pt"))
            print(f"  ★ New best R2 score: {best_score} (ep {ep})")

        if ep % save_every == 0 and ep > 0:
            torch.save({"episode": ep, "policy": policy_net.state_dict(),
                        "optimizer": optimizer.state_dict(), "epsilon": epsilon},
                       os.path.join(AGENT2_CHECKPOINT_DIR, "model_ep_latest.pt"))
            with open(os.path.join(AGENT2_CHECKPOINT_DIR,
                                   "buffer_ep_latest.pkl"), "wb") as f:
                pickle.dump(list(buffer), f)
            print(f"  💾 Checkpoint ep {ep}")

        epsilon = max(eps_min, epsilon * eps_decay)
        print(f"Ep {ep:4d} | R2:{env.robot2_score:2d} R1:{env.robot1_score:2d} "
              f"| Coll:{env.collision_count:3d} R1unseen:{env._r1_unseen:3d} "
              f"| Rew:{ep_rew:7.2f} eps:{epsilon:.3f} buf:{len(buffer)}")


if __name__ == "__main__":
    train()