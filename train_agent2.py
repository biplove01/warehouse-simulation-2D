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
from train import QNetwork, TrainingEnv
from dual_q_learning_agent import DualQAgent


if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

print(f"Training on: {compute_device}")


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

AGENT2_CHECKPOINT_DIR = "checkpoints_agent2"

AGENT1_HOME_X = ROBOT_HOME_GRID_X   # 2
AGENT1_HOME_Y = ROBOT_HOME_GRID_Y   # 0
AGENT2_HOME_X = 0
AGENT2_HOME_Y = 0

RANDOM_ACTION_PROB    = 0.30
MAX_STEPS_PER_EPISODE = 2000

PHASE_FETCHING   = "fetching"
PHASE_DELIVERING = "delivering"
PHASE_HOMING     = "homing"

# ─── REWARD / PENALTY CONSTANTS ───────────────────────────────────────────────
# All tunable values in one place. Adjust here, never dig into methods.
#
# Tuning guide:
#   Collisions stay high past ep 300  → raise AGENT_COLLISION_PENALTY (e.g. -30)
#   R2 score stays 0 past ep 400     → lower PROXIMITY penalties (they're blocking it)
#   R2 delivers but still collides   → raise AGENT_COLLISION_PENALTY further
#   Reward wildly unstable           → lower YIELDING_BONUS_PROXIMITY_THRESHOLD to 1

AGENT_COLLISION_PENALTY          = -20.0   # R2 collides with R1 → heavy penalty, episode continues
PROXIMITY_DISTANCE_1_PENALTY     = -2.0    # R2 is 1 cell from R1
PROXIMITY_DISTANCE_2_PENALTY     = -0.5    # R2 is 2 cells from R1
YIELDING_BONUS                   = 1.0     # R2 waits/yields near R1 while progressing
YIELDING_BONUS_PROXIMITY_THRESHOLD = 2     # "near" = within this Manhattan distance

# ─── VISUALIZATION TOGGLE ─────────────────────────────────────────────────────
# Set RENDER_TRAINING = True to watch training live in a pygame window.
# Set RENDER_TRAINING = False (default) for fast headless training.
RENDER_TRAINING    = False   # ← flip to True to visualise
RENDER_EVERY_N_EPS = 1       # render every N episodes (1 = every episode)
RENDER_FPS         = 6       # frames per second during visualisation


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


# ─── TWO-AGENT TRAINING ENVIRONMENT ──────────────────────────────────────────

class TwoAgentTrainingEnv:
    """
    Robot 1 runs inside its own TrainingEnv — the exact same environment it
    runs in during single-agent training. It handles its own target spawning,
    pickup, delivery, home-return, and score tracking internally.

    We only READ Robot 1's position each step for:
      - collision detection against Robot 2
      - building Robot 2's 25-feature observation (R1 position + next position)

    Robot 2 has its own phase machine and BFS navigation.
    """

    OBS_SIZE = 25

    def __init__(self, render_mode=None):
        # Robot 1: self-contained, identical to single-agent training.
        self.r1_env = TrainingEnv(render_mode=None)

        # Robot 2: separate env for world geometry only.
        self._r2_base = WarehouseEnv(render_mode=render_mode)
        self._r2_base.reset()

        self.obstacle_positions = self._r2_base.obstacle_positions
        self.shelves            = self._r2_base.shelves
        self.dropoff_platforms  = self._r2_base.dropoff_platforms
        self.charge_stations    = self._r2_base.charge_stations

        central = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_gx = round((central.x - PADDING_BORDER) / GRID_SPACING)
        self.dropoff_gy = round((central.y - PADDING_BORDER) / GRID_SPACING)

        # BFS maps for Robot 2 only.
        self.dropoff_dist = bfs_distance_map(self.dropoff_gx, self.dropoff_gy,
                                              self.obstacle_positions)
        self.r2_home_dist = bfs_distance_map(AGENT2_HOME_X, AGENT2_HOME_Y,
                                              self.obstacle_positions,
                                              target_is_walkable=True)

        print("  ℹ️  Robot 1: TrainingEnv — identical to single-agent training")
        print("  ℹ️  Robot 2: DQN learner")

        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        # Robot 2 state.
        self.robot2             = None
        self._r2_phase          = PHASE_HOMING
        self._r2_home_frames    = 0
        self._r2_target_gx      = AGENT2_HOME_X
        self._r2_target_gy      = AGENT2_HOME_Y
        self._r2_target_dist    = self.r2_home_dist
        self._r2_last_action    = -1
        self._r2_just_picked_up = False
        self._r2_just_delivered = False

        self.robot2_score    = 0
        self.collision_count = 0
        self.steps           = 0
        self._debug_ep       = -1

        # Pygame screen (only used when RENDER_TRAINING = True).
        self._screen    = None
        self._clock     = None
        self._hud_font  = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _gc(self, obj):
        return (round((obj.x - PADDING_BORDER) / GRID_SPACING),
                round((obj.y - PADDING_BORDER) / GRID_SPACING))

    @property
    def robot1(self):
        return self.r1_env.robot

    @property
    def robot1_score(self):
        return self.r1_env.score

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        from robot import Robot

        r1_obs, _ = self.r1_env.reset(seed=seed, options=options)

        self.robot2_score    = 0
        self.collision_count = 0
        self.steps           = 0
        self._debug_ep      += 1

        self.robot2 = Robot(start_x=AGENT2_HOME_X, start_y=AGENT2_HOME_Y)
        self.robot2.loaded = False

        self._r2_phase          = PHASE_HOMING
        self._r2_home_frames    = 0
        self._r2_target_gx      = AGENT2_HOME_X
        self._r2_target_gy      = AGENT2_HOME_Y
        self._r2_target_dist    = self.r2_home_dist
        self._r2_last_action    = -1
        self._r2_just_picked_up = False
        self._r2_just_delivered = False

        # Immediately assign R2 a target — pick a shelf R1 is not targeting.
        self._assign_r2_next_target()

        return self._r2_obs(), {}

    def _assign_r2_next_target(self):
        """Pick a shelf for R2 that R1 is not already targeting."""
        r1_target = (self.r1_env.target_grid_x, self.r1_env.target_grid_y)
        available = [
            s for s in self.shelves
            if not s.has_box and self._gc(s) != r1_target
        ]
        if not available:
            return   # all shelves claimed — R2 stays HOMING until one frees up
        chosen = random.choice(available)
        chosen.has_box = True
        chosen.image   = chosen.loaded_image
        gx, gy = self._gc(chosen)
        self._r2_target_gx   = gx
        self._r2_target_gy   = gy
        self._r2_target_dist = bfs_distance_map(gx, gy, self.obstacle_positions)
        self._r2_phase       = PHASE_FETCHING

    # ── Robot 2 observation ────────────────────────────────────────────────────

    def _r2_obs(self) -> np.ndarray:
        r2 = self.robot2
        r1 = self.r1_env.robot

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

        # R1 look-ahead: where will R1 be next step?
        r1_action_preview = self.r1_env.heuristic_action()
        r1nx, r1ny = predict_next(r1.grid_x, r1.grid_y,
                                   r1_action_preview, self.obstacle_positions)
        extra = [
            (r1.grid_x - r2.grid_x) / GRID_WIDTH,
            (r1.grid_y - r2.grid_y) / GRID_HEIGHT,
            float(r1.loaded),
            float(self.r1_env.returning_home),
            (r1nx - r2.grid_x) / GRID_WIDTH,
            (r1ny - r2.grid_y) / GRID_HEIGHT,
        ]
        return np.array(base + extra, dtype=np.float32)

    # ── Robot 2 heuristic (BFS-guided, used during exploration) ───────────────

    def heuristic_action(self) -> int:
        r2 = self.robot2

        if self._r2_phase == PHASE_HOMING:
            a = bfs_best_action(r2.grid_x, r2.grid_y, self.r2_home_dist)
            return a if a is not None else random.randint(0, 3)

        if (self._r2_phase == PHASE_FETCHING and not r2.loaded
                and abs(r2.grid_x - self._r2_target_gx)
                  + abs(r2.grid_y - self._r2_target_gy) == 1):
            return 4   # adjacent to shelf → interact

        if (r2.loaded and abs(r2.grid_x - self.dropoff_gx)
                       + abs(r2.grid_y - self.dropoff_gy) == 1):
            return 4   # adjacent to dropoff → interact

        dist_map = self.dropoff_dist if r2.loaded else self._r2_target_dist
        a = bfs_best_action(r2.grid_x, r2.grid_y, dist_map)
        return a if a is not None else random.randint(0, 3)

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, r2_action: int):
        self.steps += 1
        r1 = self.r1_env.robot
        r2 = self.robot2

        # Robot 1 acts through its own TrainingEnv — identical to single-agent.
        r1_action = self.r1_env.heuristic_action()
        r1nx, r1ny = predict_next(r1.grid_x, r1.grid_y, r1_action,
                                   self.r1_env.obstacle_positions)

        # Robot 2 predicted next position.
        r2nx, r2ny = predict_next(r2.grid_x, r2.grid_y, r2_action,
                                   self.obstacle_positions)

        # Collision detection — both conditions.
        r2_into_r1 = (r2nx == r1nx and r2ny == r1ny)
        r1_into_r2 = (r1nx == r2.grid_x and r1ny == r2.grid_y)
        collision  = r2_into_r1 or r1_into_r2

        # Robot 1 always steps freely through its own env.
        r1_obs_next, _, r1_done, r1_trunc, _ = self.r1_env.step(r1_action)
        if r1_done or r1_trunc:
            r1_obs_next, _ = self.r1_env.reset()

        # Collision → R2 is blocked and penalized, but episode CONTINUES.
        # Ending on collision fills the buffer with 1-step episodes and prevents
        # R2 from ever learning a full pickup→delivery→home cycle.
        # R2 stays in place, takes the penalty, and keeps learning.
        if collision:
            self.collision_count += 1
            r2_base = -0.05          # step penalty for being blocked
            self._r2_last_action = 5  # treat as wait
            r2_extra = AGENT_COLLISION_PENALTY  # heavy collision penalty

            self._update_r2_phase(5)  # phase machine sees a wait action

            done = self.steps >= MAX_STEPS_PER_EPISODE
            return self._r2_obs(), r2_base + r2_extra, done, False, {}

        # No collision — normal Robot 2 movement and reward.
        r2_base  = self._r2_move(r2_action)
        self._r2_last_action = r2_action

        r2_extra = 0.0
        md = abs(r2.grid_x - r1.grid_x) + abs(r2.grid_y - r1.grid_y)
        if md == 1:   r2_extra += PROXIMITY_DISTANCE_1_PENALTY
        elif md == 2: r2_extra += PROXIMITY_DISTANCE_2_PENALTY
        if (r2_action in (4, 5) and md <= YIELDING_BONUS_PROXIMITY_THRESHOLD
                and r2_base > 0.0):
            r2_extra += YIELDING_BONUS

        self._update_r2_phase(r2_action)

        # Debug trace for first 5 episodes.
        if self._debug_ep < 5 and self.steps % 20 == 0:
            print(
                f"  [ep{self._debug_ep} s{self.steps:4d}] "
                f"R1 ret={self.r1_env.returning_home} "
                f"pos=({r1.grid_x:2d},{r1.grid_y:2d}) ld={int(r1.loaded)} "
                f"sc={self.r1_env.score} | "
                f"R2 {self._r2_phase:10s} "
                f"pos=({r2.grid_x:2d},{r2.grid_y:2d}) ld={int(r2.loaded)} "
                f"tgt=({self._r2_target_gx},{self._r2_target_gy}) "
                f"act={r2_action} rew={r2_base+r2_extra:.2f} sc={self.robot2_score}"
            )

        done = self.steps >= MAX_STEPS_PER_EPISODE
        return self._r2_obs(), r2_base + r2_extra, done, False, {}

    # ── Robot 2 movement ──────────────────────────────────────────────────────

    def _r2_move(self, action: int) -> float:
        r2 = self.robot2
        r1 = self.r1_env.robot

        if self._r2_phase == PHASE_HOMING:
            dist_map = self.r2_home_dist
        elif r2.loaded:
            dist_map = self.dropoff_dist
        else:
            dist_map = self._r2_target_dist

        dist_before = dist_map.get((r2.grid_x, r2.grid_y), 50)

        if action < 4:
            dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
            nx, ny = r2.grid_x+dx, r2.grid_y+dy
            ok = (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                  and (nx,ny) not in self.obstacle_positions
                  and not (nx == r1.grid_x and ny == r1.grid_y))
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
                    self._r2_just_picked_up = True
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
                    self._r2_just_delivered = True
                    return 20.0
                return -4.0
            return -4.0

        return -0.05   # wait

    # ── Robot 2 phase machine ─────────────────────────────────────────────────

    def _update_r2_phase(self, action: int):
        r2 = self.robot2

        if self._r2_phase == PHASE_HOMING:
            if r2.grid_x == AGENT2_HOME_X and r2.grid_y == AGENT2_HOME_Y:
                self._assign_r2_next_target()
            return

        if self._r2_phase == PHASE_FETCHING:
            if self._r2_just_picked_up:
                self._r2_just_picked_up = False
                self._r2_phase = PHASE_DELIVERING
            return

        if self._r2_phase == PHASE_DELIVERING:
            if self._r2_just_delivered:
                self._r2_just_delivered = False
                self._r2_phase       = PHASE_HOMING
                self._r2_home_frames = 0
                self._r2_target_gx   = AGENT2_HOME_X
                self._r2_target_gy   = AGENT2_HOME_Y
                self._r2_target_dist = self.r2_home_dist

    # ── render (called only when RENDER_TRAINING = True) ──────────────────────

    def render(self, step_count=0, episode=0, epsilon=0.0):
        import pygame

        if self._screen is None:
            pygame.init()
            w = GRID_WIDTH  * GRID_SPACING + 2 * PADDING_BORDER
            h = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER
            self._screen   = pygame.display.set_mode((w, h))
            self._clock    = pygame.time.Clock()
            self._hud_font = pygame.font.SysFont("monospace", 13)
            pygame.display.set_caption("Training — R1:BFS(cyan)  R2:DQN(yellow)")

        # Handle window close during training.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys; sys.exit()

        self._screen.fill((30, 30, 30))

        for cs in self.charge_stations:
            self._screen.blit(cs.image, (cs.x, cs.y))
        for dp in self.dropoff_platforms:
            self._screen.blit(dp.image, (dp.x, dp.y))

        r1, r2 = self.r1_env.robot, self.robot2
        for shelf in self.shelves:
            gx, gy = self._gc(shelf)
            is_r1_tgt = (not self.r1_env.returning_home and not r1.loaded
                         and (gx,gy) == (self.r1_env.target_grid_x,
                                          self.r1_env.target_grid_y))
            is_r2_tgt = (self._r2_phase == PHASE_FETCHING and not r2.loaded
                         and (gx,gy) == (self._r2_target_gx, self._r2_target_gy))
            self._screen.blit(shelf.shadow_image, (shelf.x-1, shelf.y+4))
            if is_r1_tgt:
                pygame.draw.rect(self._screen, (0,220,220),
                                 (shelf.x-2, shelf.y-2, TILE_SIZE+4, TILE_SIZE+4), 2)
            if is_r2_tgt:
                pygame.draw.rect(self._screen, (255,220,0),
                                 (shelf.x-4, shelf.y-4, TILE_SIZE+8, TILE_SIZE+8), 2)
            self._screen.blit(shelf.image, (shelf.x, shelf.y))

        cx = (TILE_SIZE - ROBOT_WIDTH)  // 2
        cy = (TILE_SIZE - ROBOT_HEIGHT) // 2

        # R1 — cyan border.
        a1px = PADDING_BORDER + r1.grid_x * GRID_SPACING
        a1py = PADDING_BORDER + r1.grid_y * GRID_SPACING
        pygame.draw.rect(self._screen, (0,220,220), (a1px,a1py,TILE_SIZE,TILE_SIZE), 2)
        self._screen.blit(
            ROBOT_IMAGE_VERTICAL_BOX if r1.loaded else ROBOT_IMAGE_VERTICAL,
            (a1px+cx, a1py+cy)
        )

        # R2 — yellow border.
        a2px = PADDING_BORDER + r2.grid_x * GRID_SPACING
        a2py = PADDING_BORDER + r2.grid_y * GRID_SPACING
        pygame.draw.rect(self._screen, (255,220,0), (a2px,a2py,TILE_SIZE,TILE_SIZE), 2)
        self._screen.blit(
            ROBOT_IMAGE_SIDE_BOX if r2.loaded else ROBOT_IMAGE_SIDE,
            (a2px+cx, a2py+cy)
        )

        hud = [
            f"Ep:{episode}  Step:{step_count}  ε:{epsilon:.3f}  FPS:{RENDER_FPS}",
            f"R1(cyan)  score:{self.r1_env.score:2d}  "
            f"{'returning' if self.r1_env.returning_home else 'working':10s}  "
            f"pos:({r1.grid_x},{r1.grid_y})",
            f"R2(yellow) score:{self.robot2_score:2d}  {self._r2_phase:10s}  "
            f"pos:({r2.grid_x},{r2.grid_y})  loaded:{int(r2.loaded)}",
            f"Collisions:{self.collision_count}",
        ]
        for i, line in enumerate(hud):
            self._screen.blit(
                self._hud_font.render(line, True, (255,255,255)),
                (8, 6 + i*16)
            )

        pygame.display.flip()
        self._clock.tick(RENDER_FPS)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def score(self):
        return self.robot2_score

    @property
    def agent2_collision_count(self):
        return self.collision_count


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train():
    render_mode = "human" if RENDER_TRAINING else None
    env = TwoAgentTrainingEnv(render_mode=render_mode)

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
        obs, _     = env.reset()
        ep_rew     = 0.0
        done       = False
        step_count = 0
        should_render = RENDER_TRAINING and (ep % RENDER_EVERY_N_EPS == 0)

        while not done:
            step_count += 1

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
            obs     = next_obs
            ep_rew += rew

            if should_render:
                env.render(step_count=step_count, episode=ep, epsilon=epsilon)

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
              f"| Coll:{env.collision_count:3d} "
              f"| Rew:{ep_rew:7.2f} eps:{epsilon:.3f} buf:{len(buffer)}")


if __name__ == "__main__":
    train()