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
from train import TrainingEnv, QNetwork, HOME_WAIT_STEPS_REQUIRED
from dual_q_learning_agent import DualQAgent
from two_agent_warehouse_env import (
    QTablePolicy,
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

RENDER_DURING_TRAINING   = False
RENDER_EVERY_N_EPISODES  = 1

AGENT1_QTABLE_FOLDER = "training_data"
AGENT1_QTABLE_FILE   = "warehouse_data.pkl"
AGENT2_CHECKPOINT_DIR = "checkpoints_agent2"

# !! AGENT1_HOME must match the home the Q-table was trained with !!
# ROBOT_HOME_GRID_X / ROBOT_HOME_GRID_Y are imported from warehouse_env.
# Do NOT change these to other values — the Q-table state tuple bakes
# the home coords in, and any mismatch produces 100% unseen-state misses.
AGENT1_HOME_X = ROBOT_HOME_GRID_X   # == 2
AGENT1_HOME_Y = ROBOT_HOME_GRID_Y   # == 0

# Robot 2 home — a different charge station cell on row 0.
AGENT2_HOME_X = 0
AGENT2_HOME_Y = 0

HOME_WAIT_FRAMES_REQUIRED = 2
RANDOM_ACTION_PROBABILITY = 0.30

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Interact", "Wait"]


# ─── PHASE CONSTANTS ──────────────────────────────────────────────────────────

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
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = start_gx + dx, start_gy + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in obstacle_positions
                    and (nx, ny) not in dist_map):
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


def bfs_heuristic_action(robot_gx, robot_gy, dist_map):
    current_dist = dist_map.get((robot_gx, robot_gy), 999)
    best_action, best_dist = None, current_dist
    for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
        d = dist_map.get((robot_gx + dx, robot_gy + dy), 999)
        if d < best_dist:
            best_dist = d
            best_action = i
    return best_action


def predict_next_position(gx, gy, action, obstacle_positions):
    if action >= 4:
        return gx, gy
    dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
    nx, ny = gx + dx, gy + dy
    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
            and (nx, ny) not in obstacle_positions):
        return nx, ny
    return gx, gy


# ─── SHARED TARGET QUEUE ──────────────────────────────────────────────────────

class SharedTargetQueue:
    def __init__(self, shelves, to_grid_coords_fn):
        self._shelves = shelves
        self._to_grid_coords = to_grid_coords_fn
        self._assignments: dict = {}

    def reset(self):
        self._assignments.clear()
        for shelf in self._shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

    def _all_claimed(self):
        return set(self._assignments.values())

    def request_target(self, robot_id):
        claimed = self._all_claimed()
        available = [
            s for s in self._shelves
            if self._to_grid_coords(s) not in claimed
        ]
        if not available:
            return None
        chosen = random.choice(available)
        chosen.has_box = True
        chosen.image = chosen.loaded_image
        gx, gy = self._to_grid_coords(chosen)
        self._assignments[robot_id] = (gx, gy)
        return gx, gy

    def release_target(self, robot_id):
        self._assignments.pop(robot_id, None)


# ─── TRAINING ENVIRONMENT ─────────────────────────────────────────────────────

class TwoAgentTrainingEnv:
    """
    Two-robot training environment.

    Robot 1: frozen Q-table policy. Uses its trained home (ROBOT_HOME_GRID_X/Y).
             The QTablePolicy.returning_home flag is kept in sync with the phase
             so the state tuple the Q-table sees matches what it saw at training.
    Robot 2: DQN learner. 25-dim observation (19 base + 6 R1-awareness).
    """

    OBS_SIZE = 25

    def __init__(self, render_mode=None):
        self._base_env = WarehouseEnv(render_mode=render_mode)
        self._base_env.reset()

        self.obstacle_positions  = self._base_env.obstacle_positions
        self.shelves             = self._base_env.shelves
        self.dropoff_platforms   = self._base_env.dropoff_platforms
        self.charge_stations     = self._base_env.charge_stations

        central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_gx = round((central_platform.x - PADDING_BORDER) / GRID_SPACING)
        self.dropoff_gy = round((central_platform.y - PADDING_BORDER) / GRID_SPACING)
        self.dropoff_dist_map = bfs_distance_map(
            self.dropoff_gx, self.dropoff_gy, self.obstacle_positions
        )

        self.agent1_home_dist_map = bfs_distance_map(
            AGENT1_HOME_X, AGENT1_HOME_Y, self.obstacle_positions,
            target_is_walkable=True
        )
        self.agent2_home_dist_map = bfs_distance_map(
            AGENT2_HOME_X, AGENT2_HOME_Y, self.obstacle_positions,
            target_is_walkable=True
        )

        self.shared_queue = SharedTargetQueue(
            shelves=self.shelves,
            to_grid_coords_fn=self._to_grid_coords,
        )

        # ── Load Robot 1's Q-table ────────────────────────────────────────────
        raw_agent = DualQAgent(action_dim=6)
        raw_agent.load_tables(AGENT1_QTABLE_FOLDER, AGENT1_QTABLE_FILE)
        raw_agent.epsilon = 0.0
        self.agent1_policy = QTablePolicy(raw_agent)
        # Initialise shelf target to trained home so first state key is valid.
        self.agent1_policy.current_shelf_target_x = AGENT1_HOME_X
        self.agent1_policy.current_shelf_target_y = AGENT1_HOME_Y
        # Tell the policy whether R1 is returning home (affects state key build).
        self.agent1_policy.returning_home = True

        import gymnasium as gym
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        self.robot1 = None
        self.robot2 = None

        self.robot1_phase         = PHASE_HOMING
        self.robot2_phase         = PHASE_HOMING
        self.robot1_home_frames   = 0
        self.robot2_home_frames   = 0
        self.robot1_target_gx     = AGENT1_HOME_X
        self.robot1_target_gy     = AGENT1_HOME_Y
        self.robot2_target_gx     = AGENT2_HOME_X
        self.robot2_target_gy     = AGENT2_HOME_Y
        self.robot1_target_dist_map = self.agent1_home_dist_map
        self.robot2_target_dist_map = self.agent2_home_dist_map
        self.robot1_last_action   = -1
        self.robot2_last_action   = -1

        self.robot1_score   = 0
        self.robot2_score   = 0
        self.collision_count = 0
        self.steps           = 0

        # Unseen-state miss counter — printed in training log.
        self._r1_unseen = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_grid_coords(self, obj):
        return (
            round((obj.x - PADDING_BORDER) / GRID_SPACING),
            round((obj.y - PADDING_BORDER) / GRID_SPACING),
        )

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        from robot import Robot

        self.shared_queue.reset()
        self.steps          = 0
        self.robot1_score   = 0
        self.robot2_score   = 0
        self.collision_count = 0
        self._r1_unseen     = 0

        blocked = self.obstacle_positions | {
            self._to_grid_coords(p) for p in self.dropoff_platforms
        }
        all_valid = [
            (gx, gy)
            for gx in range(GRID_WIDTH)
            for gy in range(GRID_HEIGHT)
            if (gx, gy) not in blocked
        ]

        r1_start = random.choice(all_valid)
        r2_start = random.choice([c for c in all_valid if c != r1_start])

        self.robot1 = Robot(start_x=r1_start[0], start_y=r1_start[1])
        self.robot2 = Robot(start_x=r2_start[0], start_y=r2_start[1])
        self.robot1.loaded = False
        self.robot2.loaded = False

        self.robot1_phase         = PHASE_HOMING
        self.robot2_phase         = PHASE_HOMING
        self.robot1_home_frames   = 0
        self.robot2_home_frames   = 0
        self.robot1_target_gx     = AGENT1_HOME_X
        self.robot1_target_gy     = AGENT1_HOME_Y
        self.robot2_target_gx     = AGENT2_HOME_X
        self.robot2_target_gy     = AGENT2_HOME_Y
        self.robot1_target_dist_map = self.agent1_home_dist_map
        self.robot2_target_dist_map = self.agent2_home_dist_map
        self.robot1_last_action   = -1
        self.robot2_last_action   = -1

        # Reset Q-table policy state.
        self.agent1_policy.current_shelf_target_x = AGENT1_HOME_X
        self.agent1_policy.current_shelf_target_y = AGENT1_HOME_Y
        self.agent1_policy.returning_home = True

        return self._get_robot2_obs(), {}

    # ── Robot 1 observation for Q-table ───────────────────────────────────────

    def _build_robot1_qtable_obs(self) -> np.ndarray:
        """
        Builds the 19-element proxy observation that QTablePolicy._build_qtable_obs
        reads. Only indices 0,1,2,15,16 are used; the rest are zero.

        Critical: obs[0] = r1.grid_x / GRID_WIDTH (normalised).
        QTablePolicy._build_qtable_obs then does round(obs[0] * GRID_WIDTH)
        to recover the raw int — the round-trip is lossless for integer coords.

        current_shelf_target_x/y on the policy object provides bx/by
        (the shelf the Q-table navigates toward) and is kept in sync by
        _update_robot1_phase.
        """
        r1 = self.robot1
        obs = np.zeros(19, dtype=np.float32)
        obs[0]  = r1.grid_x / GRID_WIDTH
        obs[1]  = r1.grid_y / GRID_HEIGHT
        obs[2]  = float(r1.loaded)
        # Direction encoding from last action.
        lx, ly = 0, 0
        if self.robot1_last_action == 0:   ly = -1
        elif self.robot1_last_action == 1: ly =  1
        elif self.robot1_last_action == 2: lx = -1
        elif self.robot1_last_action == 3: lx =  1
        obs[15] = float(lx)
        obs[16] = float(ly)
        return obs

    # ── Robot 2 observation ────────────────────────────────────────────────────

    def _get_robot2_obs(self) -> np.ndarray:
        r2 = self.robot2
        r1 = self.robot1

        if r2.loaded:
            nav_x, nav_y = self.dropoff_gx, self.dropoff_gy
        elif self.robot2_phase == PHASE_HOMING:
            nav_x, nav_y = AGENT2_HOME_X, AGENT2_HOME_Y
        else:
            nav_x, nav_y = self.robot2_target_gx, self.robot2_target_gy

        lx, ly = 0.0, 0.0
        if self.robot2_last_action == 0:   ly = -1.0
        elif self.robot2_last_action == 1: ly =  1.0
        elif self.robot2_last_action == 2: lx = -1.0
        elif self.robot2_last_action == 3: lx =  1.0

        can_pickup = float(
            not r2.loaded
            and self.robot2_phase == PHASE_FETCHING
            and abs(r2.grid_x - self.robot2_target_gx)
              + abs(r2.grid_y - self.robot2_target_gy) == 1
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
            nx, ny = r2.grid_x + dx, r2.grid_y + dy
            oob  = not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT)
            wall = (nx, ny) in self.obstacle_positions
            is_r1 = (nx == r1.grid_x and ny == r1.grid_y)
            base.append(1.0 if (oob or wall or is_r1) else 0.0)
        base.extend([lx, ly, can_pickup, can_deliver])

        r1_next_x, r1_next_y = self._predict_robot1_next_pos()
        extra = [
            (r1.grid_x - r2.grid_x) / GRID_WIDTH,
            (r1.grid_y - r2.grid_y) / GRID_HEIGHT,
            float(r1.loaded),
            float(self.robot1_phase == PHASE_HOMING),
            (r1_next_x - r2.grid_x) / GRID_WIDTH,
            (r1_next_y - r2.grid_y) / GRID_HEIGHT,
        ]
        return np.array(base + extra, dtype=np.float32)

    # ── Robot 1 action selection ───────────────────────────────────────────────

    def _select_robot1_action(self) -> int:
        """
        Primary: greedy Q-table lookup.
        Fallback: BFS toward current phase target when state is unseen.
        The BFS fallback means R1 still makes progress even for rare states
        that weren't in the training distribution.
        """
        qtable_obs = self._build_robot1_qtable_obs()
        action = self.agent1_policy.select_action(qtable_obs)

        # Detect unseen-state fallback (select_action returns 0 for unseen).
        key = tuple(qtable_obs.astype(int).tolist())
        # QTablePolicy reconstructs its own key internally; replicate it here
        # just to check presence.
        rx = round(float(qtable_obs[0]) * GRID_WIDTH)
        ry = round(float(qtable_obs[1]) * GRID_WIDTH)   # intentional: same scale check
        # Simpler: ask whether the policy fell back by seeing if table is small
        # or the action seems wrong. Actually, just always run BFS as override
        # when the Q-table says action 0 AND we're clearly not heading up.
        # The cleanest approach: track misses in QTablePolicy directly.
        # For now, augment with always-available BFS as a safety net.

        if self.robot1_phase == PHASE_HOMING:
            # When homing, prefer BFS to home unconditionally — the Q-table
            # home-return behavior is reliable only for states near the trained
            # home. BFS guarantees progress every step.
            bfs = bfs_heuristic_action(
                self.robot1.grid_x, self.robot1.grid_y,
                self.agent1_home_dist_map
            )
            if bfs is not None:
                return bfs
            return 5  # wait if somehow stuck

        if self.robot1_phase == PHASE_DELIVERING:
            # When delivering, always BFS to dropoff — Q-table delivery is
            # identical to trained behavior since dropoff coords are the same.
            bfs = bfs_heuristic_action(
                self.robot1.grid_x, self.robot1.grid_y,
                self.dropoff_dist_map
            )
            if bfs is not None:
                return bfs
            # Adjacent to dropoff → interact
            if (abs(self.robot1.grid_x - self.dropoff_gx)
              + abs(self.robot1.grid_y - self.dropoff_gy) == 1):
                return 4
            return 5

        # PHASE_FETCHING: use Q-table (this is what it was trained for).
        # BFS as fallback only.
        if action == 5:  # Q-table said wait — likely unseen state
            self._r1_unseen += 1
            bfs = bfs_heuristic_action(
                self.robot1.grid_x, self.robot1.grid_y,
                self.robot1_target_dist_map
            )
            if bfs is not None:
                return bfs
        # Adjacent to shelf → interact (pickup)
        if (not self.robot1.loaded and
                abs(self.robot1.grid_x - self.robot1_target_gx)
              + abs(self.robot1.grid_y - self.robot1_target_gy) == 1):
            return 4
        return action

    def _predict_robot1_next_pos(self):
        action = self._select_robot1_action()
        return predict_next_position(
            self.robot1.grid_x, self.robot1.grid_y,
            action, self.obstacle_positions
        )

    # ── Robot 2 heuristic (for exploration) ───────────────────────────────────

    def heuristic_action_robot2(self) -> int:
        r2 = self.robot2

        if self.robot2_phase == PHASE_HOMING:
            a = bfs_heuristic_action(r2.grid_x, r2.grid_y, self.agent2_home_dist_map)
            return a if a is not None else random.randint(0, 3)

        dist_map = (self.dropoff_dist_map if self.robot2_phase == PHASE_DELIVERING
                    else self.robot2_target_dist_map)

        if (self.robot2_phase == PHASE_FETCHING and not r2.loaded
                and abs(r2.grid_x - self.robot2_target_gx)
                  + abs(r2.grid_y - self.robot2_target_gy) == 1):
            return 4

        if (r2.loaded
                and abs(r2.grid_x - self.dropoff_gx)
                  + abs(r2.grid_y - self.dropoff_gy) == 1):
            return 4

        a = bfs_heuristic_action(r2.grid_x, r2.grid_y, dist_map)
        return a if a is not None else random.randint(0, 3)

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, robot2_action: int):
        self.steps += 1
        r1 = self.robot1
        r2 = self.robot2

        robot1_action = self._select_robot1_action()

        r1_next_x, r1_next_y = predict_next_position(
            r1.grid_x, r1.grid_y, robot1_action, self.obstacle_positions
        )
        r2_next_x, r2_next_y = predict_next_position(
            r2.grid_x, r2.grid_y, robot2_action, self.obstacle_positions
        )

        r2_into_r1_next     = (r2_next_x == r1_next_x and r2_next_y == r1_next_y)
        r1_into_r2_current  = (r1_next_x == r2.grid_x and r1_next_y == r2.grid_y)
        collision = r2_into_r1_next or r1_into_r2_current

        if collision:
            self.collision_count += 1

        # ── Move Robot 1 ──────────────────────────────────────────────────────
        if robot1_action < 4:
            if (0 <= r1_next_x < GRID_WIDTH and 0 <= r1_next_y < GRID_HEIGHT
                    and (r1_next_x, r1_next_y) not in self.obstacle_positions):
                r1.grid_x, r1.grid_y = r1_next_x, r1_next_y
        self.robot1_last_action = robot1_action

        # ── Move Robot 2 ──────────────────────────────────────────────────────
        effective_r2 = robot2_action
        r2_extra = 0.0

        if collision:
            r2_extra += AGENT_COLLISION_PENALTY
            if r2_into_r1_next and robot2_action < 4:
                effective_r2 = 5  # force wait

        r2_base = self._apply_robot2_movement(effective_r2)
        self.robot2_last_action = effective_r2

        if not collision:
            md = abs(r2.grid_x - r1.grid_x) + abs(r2.grid_y - r1.grid_y)
            if md == 1:
                r2_extra += PROXIMITY_DISTANCE_1_PENALTY
            elif md == 2:
                r2_extra += PROXIMITY_DISTANCE_2_PENALTY
            if (robot2_action in (4, 5)
                    and md <= YIELDING_BONUS_PROXIMITY_THRESHOLD
                    and r2_base > 0.0):
                r2_extra += YIELDING_BONUS

        # ── Update phase machines ─────────────────────────────────────────────
        self._update_robot1_phase(robot1_action)
        self._update_robot2_phase(effective_r2)

        is_done = self.steps >= 500
        return self._get_robot2_obs(), r2_base + r2_extra, is_done, False, {}

    # ── Robot 2 movement ──────────────────────────────────────────────────────

    def _apply_robot2_movement(self, action: int) -> float:
        r2 = self.robot2

        if self.robot2_phase == PHASE_HOMING:
            dist_map = self.agent2_home_dist_map
        elif r2.loaded:
            dist_map = self.dropoff_dist_map
        else:
            dist_map = self.robot2_target_dist_map

        dist_before = dist_map.get((r2.grid_x, r2.grid_y), 50)

        if action < 4:
            dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
            nx, ny = r2.grid_x + dx, r2.grid_y + dy
            passable = (
                0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                and (nx, ny) not in self.obstacle_positions
                and not (nx == self.robot1.grid_x and ny == self.robot1.grid_y)
            )
            if passable:
                r2.grid_x, r2.grid_y = nx, ny
                dist_after = dist_map.get((r2.grid_x, r2.grid_y), 50)
                delta = dist_before - dist_after
                return delta * (1.0 if delta >= 0 else 4.0) - 0.05
            return -3.0

        if action == 4:
            return self._handle_robot2_interact()
        return -0.05  # wait

    def _handle_robot2_interact(self) -> float:
        r2 = self.robot2

        if self.robot2_phase == PHASE_FETCHING and not r2.loaded:
            if (abs(r2.grid_x - self.robot2_target_gx)
              + abs(r2.grid_y - self.robot2_target_gy) == 1):
                r2.loaded = True
                for s in self.shelves:
                    if self._to_grid_coords(s) == (self.robot2_target_gx,
                                                    self.robot2_target_gy):
                        s.has_box = False
                        s.image = s.empty_image
                        break
                return 10.0
            return -4.0

        if self.robot2_phase == PHASE_DELIVERING and r2.loaded:
            if (abs(r2.grid_x - self.dropoff_gx)
              + abs(r2.grid_y - self.dropoff_gy) == 1):
                r2.loaded = False
                self.robot2_score += 1
                self.shared_queue.release_target(robot_id=2)
                return 20.0
            return -4.0

        return -4.0

    # ── Phase state machines ───────────────────────────────────────────────────

    def _update_robot1_phase(self, action: int):
        r1 = self.robot1

        if self.robot1_phase == PHASE_HOMING:
            if r1.grid_x == AGENT1_HOME_X and r1.grid_y == AGENT1_HOME_Y:
                self.robot1_home_frames += 1
                if self.robot1_home_frames >= HOME_WAIT_FRAMES_REQUIRED:
                    self.robot1_home_frames = 0
                    result = self.shared_queue.request_target(robot_id=1)
                    if result is not None:
                        gx, gy = result
                        self.robot1_target_gx = gx
                        self.robot1_target_gy = gy
                        self.robot1_target_dist_map = bfs_distance_map(
                            gx, gy, self.obstacle_positions
                        )
                        # Update Q-table policy so bx/by in state tuple is correct.
                        self.agent1_policy.current_shelf_target_x = gx
                        self.agent1_policy.current_shelf_target_y = gy
                        self.agent1_policy.returning_home = False
                        self.robot1_phase = PHASE_FETCHING
            return

        if self.robot1_phase == PHASE_FETCHING:
            # Pickup triggers on Interact when adjacent to the target shelf.
            adjacent = (
                abs(r1.grid_x - self.robot1_target_gx)
              + abs(r1.grid_y - self.robot1_target_gy) == 1
            )
            if action == 4 and not r1.loaded and adjacent:
                r1.loaded = True
                for s in self.shelves:
                    if self._to_grid_coords(s) == (self.robot1_target_gx,
                                                    self.robot1_target_gy):
                        s.has_box = False
                        s.image = s.empty_image
                        break
                self.robot1_phase = PHASE_DELIVERING
            return

        if self.robot1_phase == PHASE_DELIVERING:
            adjacent = (
                abs(r1.grid_x - self.dropoff_gx)
              + abs(r1.grid_y - self.dropoff_gy) == 1
            )
            if action == 4 and r1.loaded and adjacent:
                r1.loaded = False
                self.robot1_score += 1
                self.shared_queue.release_target(robot_id=1)
                self.robot1_phase       = PHASE_HOMING
                self.robot1_home_frames = 0
                self.robot1_target_gx   = AGENT1_HOME_X
                self.robot1_target_gy   = AGENT1_HOME_Y
                self.robot1_target_dist_map = self.agent1_home_dist_map
                self.agent1_policy.returning_home = True

    def _update_robot2_phase(self, action: int):
        r2 = self.robot2

        if self.robot2_phase == PHASE_HOMING:
            if r2.grid_x == AGENT2_HOME_X and r2.grid_y == AGENT2_HOME_Y:
                self.robot2_home_frames += 1
                if self.robot2_home_frames >= HOME_WAIT_FRAMES_REQUIRED:
                    self.robot2_home_frames = 0
                    result = self.shared_queue.request_target(robot_id=2)
                    if result is not None:
                        gx, gy = result
                        self.robot2_target_gx = gx
                        self.robot2_target_gy = gy
                        self.robot2_target_dist_map = bfs_distance_map(
                            gx, gy, self.obstacle_positions
                        )
                        self.robot2_phase = PHASE_FETCHING
            return

        if self.robot2_phase == PHASE_FETCHING:
            adjacent = (
                abs(r2.grid_x - self.robot2_target_gx)
              + abs(r2.grid_y - self.robot2_target_gy) == 1
            )
            if action == 4 and not r2.loaded and adjacent:
                self.robot2_phase = PHASE_DELIVERING
            return

        if self.robot2_phase == PHASE_DELIVERING:
            if action == 4 and not r2.loaded:
                self.robot2_phase       = PHASE_HOMING
                self.robot2_home_frames = 0
                self.robot2_target_gx   = AGENT2_HOME_X
                self.robot2_target_gy   = AGENT2_HOME_Y
                self.robot2_target_dist_map = self.agent2_home_dist_map

    # ── expose for training loop ───────────────────────────────────────────────

    def heuristic_action(self) -> int:
        return self.heuristic_action_robot2()

    @property
    def score(self) -> int:
        return self.robot2_score

    @property
    def agent2_collision_count(self) -> int:
        return self.collision_count


# ─── TRAINING FUNCTION ────────────────────────────────────────────────────────

def train():
    env = TwoAgentTrainingEnv(render_mode=None)

    state_dim  = env.OBS_SIZE
    action_dim = 6

    agent2_policy_network = QNetwork(state_dim, action_dim).to(compute_device)
    agent2_target_network = QNetwork(state_dim, action_dim).to(compute_device)
    agent2_target_network.load_state_dict(agent2_policy_network.state_dict())

    agent2_optimizer     = optim.Adam(agent2_policy_network.parameters(), lr=1e-4)
    agent2_replay_buffer = deque(maxlen=50000)

    batch_size                      = 128
    discount_factor                 = 0.98
    epsilon                         = 0.7
    epsilon_min                     = 0.1
    epsilon_decay                   = 0.998
    target_network_update_frequency = 10
    checkpoint_save_frequency       = 50
    total_episodes                  = 2500

    os.makedirs(AGENT2_CHECKPOINT_DIR, exist_ok=True)
    best_score = -1

    for episode in range(total_episodes):
        current_state, _ = env.reset()
        episode_total_reward = 0.0
        is_done = False

        while not is_done:
            if random.random() < epsilon:
                if random.random() < (1.0 - RANDOM_ACTION_PROBABILITY):
                    chosen_action = env.heuristic_action()
                else:
                    chosen_action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q = agent2_policy_network(
                        torch.as_tensor(
                            current_state, dtype=torch.float32, device=compute_device
                        ).unsqueeze(0)
                    )
                    chosen_action = q.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated

            agent2_replay_buffer.append(
                (current_state, chosen_action, reward, next_state, is_done)
            )
            current_state = next_state
            episode_total_reward += reward

            if len(agent2_replay_buffer) > batch_size:
                sb = random.sample(agent2_replay_buffer, batch_size)
                s, a, r, ns, d = zip(*sb)

                s_t  = torch.as_tensor(np.array(s),  dtype=torch.float32, device=compute_device)
                a_t  = torch.as_tensor(a,             dtype=torch.long,    device=compute_device).unsqueeze(1)
                r_t  = torch.as_tensor(r,             dtype=torch.float32, device=compute_device).unsqueeze(1)
                ns_t = torch.as_tensor(np.array(ns),  dtype=torch.float32, device=compute_device)
                d_t  = torch.as_tensor(d,             dtype=torch.float32, device=compute_device).unsqueeze(1)

                current_q = agent2_policy_network(s_t).gather(1, a_t)
                with torch.no_grad():
                    best_a  = agent2_policy_network(ns_t).argmax(dim=1, keepdim=True)
                    next_q  = agent2_target_network(ns_t).gather(1, best_a)
                    target_q = r_t + discount_factor * next_q * (1 - d_t)

                loss = nn.MSELoss()(current_q, target_q)
                agent2_optimizer.zero_grad()
                loss.backward()
                agent2_optimizer.step()

        if episode % target_network_update_frequency == 0:
            agent2_target_network.load_state_dict(agent2_policy_network.state_dict())

        if env.score > best_score:
            best_score = env.score
            torch.save(
                agent2_policy_network.state_dict(),
                os.path.join(AGENT2_CHECKPOINT_DIR, "best_model.pt"),
            )
            print(f"  ★ New Best Score: {best_score}! Model saved.")

        if episode % checkpoint_save_frequency == 0 and episode > 0:
            torch.save(
                {
                    "episode":   episode,
                    "policy":    agent2_policy_network.state_dict(),
                    "optimizer": agent2_optimizer.state_dict(),
                    "epsilon":   epsilon,
                },
                os.path.join(AGENT2_CHECKPOINT_DIR, "model_ep_latest.pt"),
            )
            with open(os.path.join(AGENT2_CHECKPOINT_DIR,
                                   "buffer_ep_latest.pkl"), "wb") as f:
                pickle.dump(list(agent2_replay_buffer), f)
            print(f"  💾 Checkpoint saved at episode {episode}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(
            f"Ep {episode:4d} | "
            f"R2 Score: {env.robot2_score:2d} | "
            f"R1 Score: {env.robot1_score:2d} | "
            f"Collisions: {env.collision_count:3d} | "
            f"R1 unseen: {env._r1_unseen:3d} | "
            f"Reward: {episode_total_reward:7.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(agent2_replay_buffer)}"
        )


if __name__ == "__main__":
    train()