I am making a warehouse simulation project where robots go from charge station or resting place to Shelves where items spawn. The items are stored in a queue. A robot is assigned a task to pick the item from the shelf. and then drop it in a drop off location. I initially trained and finetuned one robot in that environment till it was perfect. import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pygame
from world import create_map
from constants import *
from robot import Robot
import random

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([
                GRID_WIDTH - 1,
                GRID_HEIGHT - 1,
                1,
                GRID_WIDTH - 1,
                GRID_HEIGHT - 1,
                GRID_WIDTH - 1,
                GRID_HEIGHT - 1,
                3
            ], dtype=np.float32)
        )
        self.window = None
        self.clock = None
        self.current_step = 0
        self.reward = 0
        self.max_steps = 10000
        self.score = 0

        # 10 deliveries for training episodes, very high for human testing
        self.goal_deliveries = 10 if render_mode is None else 100

    def _get_shelf_at(self, gx, gy):
        for s in self.shelves:
            sgx = round((s.x - PADDING_BORDER) / GRID_SPACING)
            sgy = round((s.y - PADDING_BORDER) / GRID_SPACING)
            if sgx == gx and sgy == gy:
                return s
        return None

    def _is_adjacent(self, rx, ry, tx, ty):
        return abs(rx - tx) + abs(ry - ty) == 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.current_step = 0
        self.reward = 0
        self.state_history = deque(maxlen=8)
        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()

        # All shelves start empty
        for s in self.shelves:
            if hasattr(s, 'has_box'):
                s.has_box = False

        # Target queue
        self.target_queue = deque()

        # Central dropoff target
        central_drop = self.dropoff_platforms[2]
        self.dropoff_gx = round((central_drop.x - PADDING_BORDER) / GRID_SPACING)
        self.dropoff_gy = round((central_drop.y - PADDING_BORDER) / GRID_SPACING)

        # Auto-spawn 10 random boxes only during training (render_mode=None)
        if self.render_mode is None:
            shelf_positions = []
            for s in self.shelves:
                gx = round((s.x - PADDING_BORDER) / GRID_SPACING)
                gy = round((s.y - PADDING_BORDER) / GRID_SPACING)
                shelf_positions.append((s, gx, gy))
            random.shuffle(shelf_positions)
            num_to_spawn = 10  # Fixed typo here
            for i in range(min(num_to_spawn, len(shelf_positions))):
                s, gx, gy = shelf_positions[i]
                s.has_box = True
                self.target_queue.append((gx, gy))

        # Blocked cells for robot spawn
        blocked = set()
        for obj in self.shelves + self.dropoff_platforms + self.charge_stations:
            gx = round((obj.x - PADDING_BORDER) / GRID_SPACING)
            gy = round((obj.y - PADDING_BORDER) / GRID_SPACING)
            blocked.add((gx, gy))

        while True:
            random_x = np.random.randint(0, GRID_WIDTH)
            random_y = np.random.randint(0, GRID_HEIGHT)
            if (random_x, random_y) not in blocked:
                break

        self.robot = Robot(start_x=random_x, start_y=random_y)

        observation = self._get_obs()
        info = {}
        if self.render_mode == 'human':
            self._render_frame()
        return observation, info

    def _get_obs(self):
        if self.target_queue:
            box_pos = [self.target_queue[0][0], self.target_queue[0][1]]
        else:
            box_pos = [self.robot.grid_x, self.robot.grid_y]

        return np.array([
            self.robot.grid_x,
            self.robot.grid_y,
            float(self.robot.loaded),
            box_pos[0], box_pos[1],
            self.dropoff_gx, self.dropoff_gy,
            {'up': 0, 'down': 1, 'left': 2, 'right': 3}[self.robot.direction]
        ], dtype=np.float32)



    def step(self, action):
        self.reward = -0.3
        terminated = False
        truncated = False
        obstacles = self.shelves + self.dropoff_platforms
        moved = False

        # Current target for shaping
        if self.robot.loaded:
            target_x, target_y = self.dropoff_gx, self.dropoff_gy
        else:
            if self.target_queue:
                target_x, target_y = self.target_queue[0]
            else:
                target_x, target_y = self.robot.grid_x, self.robot.grid_y

        curr_distance = self.shortest_path_dist(self.robot.grid_x, self.robot.grid_y, target_x, target_y)

        if action < 4:
            dirs = ['up', 'down', 'left', 'right']
            moved = self.robot.handle_inputs_single(dirs[action], obstacles)
            if not moved:
                self.reward -= 2

        elif action == 4:
            if self.robot.loaded:
                success = self.robot.drop_box(self.dropoff_platforms)
                if success:
                    self.reward += 200
                    self.score += 1
                    if self.target_queue:
                        self.target_queue.popleft()
                    if self.score >= self.goal_deliveries:
                        terminated = True
                else:
                    self.reward -= 25
            else:
                if self.target_queue:
                    tx, ty = self.target_queue[0]
                    target_shelf = self._get_shelf_at(tx, ty)
                    if (target_shelf and
                        self._is_adjacent(self.robot.grid_x, self.robot.grid_y, tx, ty) and
                        target_shelf.has_box):
                        self.robot.loaded = True
                        target_shelf.has_box = False
                        target_shelf.image = target_shelf.empty_image
                        self.reward += 50
                    else:
                        self.reward -= 50
                else:
                    self.reward -= 50

        elif action == 5:
            moved = True 
            if len(self.target_queue) > 0:
                self.reward -= 5

        # Next target/distance
        if self.robot.loaded:
            next_target_x, next_target_y = self.dropoff_gx, self.dropoff_gy
        else:
            if self.target_queue:
                next_target_x, next_target_y = self.target_queue[0]
            else:
                next_target_x, next_target_y = self.robot.grid_x, self.robot.grid_y

        next_distance = self.shortest_path_dist(self.robot.grid_x, self.robot.grid_y, next_target_x, next_target_y)
        self.reward += 0.99 * (curr_distance - next_distance)

        # Looping penalty
        next_obs = self._get_obs()
        state_tuple = tuple(next_obs.astype(int).tolist())
        self.state_history.append(state_tuple)
        if len(self.state_history) == self.state_history.maxlen:
            if list(self.state_history).count(state_tuple) >= 3:
                self.reward -= 100

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            self.reward -= 300

        if self.render_mode == 'human':
            self._render_frame()

        return next_obs, terminated, truncated, {}

    def shortest_path_dist(self, start_x, start_y, goal_x, goal_y):
        obstacles = set()
        for obj in self.shelves + self.dropoff_platforms:
            gx = round((obj.x - PADDING_BORDER) / GRID_SPACING)
            gy = round((obj.y - PADDING_BORDER) / GRID_SPACING)
            obstacles.add((gx, gy))

        queue = deque([(start_x, start_y, 0)])
        visited = set([(start_x, start_y)])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y,  dist = queue.popleft()

            if (x == goal_x and y == goal_y) or (abs(x - goal_x) + abs(y - goal_y) == 1):
                return dist

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                    (nx, ny) not in visited and (nx, ny) not in obstacles):
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

        return 9999


    def _render_frame(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((GRID_WIDTH * GRID_SPACING, GRID_HEIGHT * GRID_SPACING))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get  ():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                gx = pos[0] // GRID_SPACING
                gy = pos[1] // GRID_SPACING
                shelf = self._get_shelf_at(gx, gy)
                if shelf and not shelf.has_box:
                    shelf.has_box = True
                    shelf.image = shelf.loaded_image
                    self.target_queue.append((gx, gy))

        canvas = pygame.Surface((GRID_WIDTH * GRID_SPACING, GRID_HEIGHT * GRID_SPACING))
        canvas.fill("#5FCB9B")

        for obj in self.charge_stations + self.dropoff_platforms:
            canvas.blit(obj.image, obj)

        for obj in self.shelves:
            canvas.blit(obj.shadow_image, (obj.x - 3, obj.y + 12))

        r_rect = self.robot.get_pixel_rect()
        canvas.blit(self.robot.image, (r_rect.x, r_rect.y))

        for obj in self.shelves:
            canvas.blit(obj.image, obj)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = Nonefrom warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent
from collections import deque

env = WarehouseEnv(render_mode=None)
agent = DualQAgent(env.action_space.n)


SAVE_INTERVAL = 500
EPISODES = 10_000

DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"
agent.load_tables(DATA_FOLDER, FILE_NAME)


robot_prev_state = env.reset()


for ep in range(1, EPISODES + 1):

    print(f"episode: {ep}")
    obs, _ = env.reset()
    steps = 0
    done = False

    total_reward = 0.0
    successfully_dropped = False


    while not done:
      action = agent.select_action(obs)

      next_obs, term, trunc, _ = env.step(action)

      total_reward += env.reward

      agent.update(obs, action, env.reward, next_obs)

      obs = next_obs
      done = term or trunc
      successfully_dropped = term     # just for debugging
      steps += 1


    if ep % 2 == 0:
      print(f"Ep {ep} | total_reward: {total_reward:.4f} | steps: {steps} | "
      f"states: {len(agent.q_table)} Sucess: {successfully_dropped}")

    if ep % SAVE_INTERVAL == 0:
      print(f"Episode {ep}: Saving checkpoint...")
      agent.save_tables(DATA_FOLDER, FILE_NAME)

      print(f"Current States {len(agent.q_table)}")

# Final save
agent.save_tables(DATA_FOLDER, FILE_NAME)

# Things stable for now
import pygame
import os

PADDING_BORDER = 10
TILE_SIZE = 50
TILE_GAP = 3
GRID_SPACING = TILE_SIZE + TILE_GAP

GRID_WIDTH = 22
GRID_HEIGHT = 15

# SHELF_IMAGE_HEIGHT = 64
SHELF_IMAGE_HEIGHT = 50
CHARGE_STATION_HEIGHT = 58

ROBOT_WIDTH = 40
ROBOT_HEIGHT = 48

NUMBER_OF_ROBOTS = 5

# load images
def load_img(image_name, scale=None):
  image = pygame.image.load(os.path.join("assets", image_name))
  if scale is not None:
    image = pygame.transform.scale(image, scale)
  return image


# Images
ROBOT_IMAGE_SIDE = load_img("robot-side.png", ( ROBOT_HEIGHT, ROBOT_WIDTH))
ROBOT_IMAGE_SIDE_BOX = load_img("robot-side-box.png", ( ROBOT_HEIGHT, ROBOT_WIDTH))
ROBOT_IMAGE_VERTICAL = load_img("robot-vertical.png", ( ROBOT_WIDTH, ROBOT_HEIGHT))
ROBOT_IMAGE_VERTICAL_BOX = load_img("robot-vertical-box.png", (ROBOT_WIDTH, ROBOT_HEIGHT))
SHELF_IMAGE_EMPTY = load_img("shelf-empty.png", (TILE_SIZE, SHELF_IMAGE_HEIGHT))
SHELF_IMAGE_FILLED = load_img("shelf-filled.png", (TILE_SIZE, SHELF_IMAGE_HEIGHT))
SHELF_SHADOW = load_img("shelf-shadow.png", (TILE_SIZE + 3, SHELF_IMAGE_HEIGHT))

ROBOT_CHARGE_STATION_IMAGE = load_img("robot-charging.png", (TILE_SIZE, CHARGE_STATION_HEIGHT))
DROP_OFF_PLATFORM_IMAGE = load_img("drop-off.png", (TILE_SIZE, CHARGE_STATION_HEIGHT))
from constants import *
from sprites import *
import random


def create_map():
  charge_stations = []
  shelves = []
  dropoff_platforms = []

  def grid_to_pixel(gx, gy):
      return PADDING_BORDER + gx * GRID_SPACING, PADDING_BORDER + gy * GRID_SPACING

  # Charge stations: row 0, cols 0–4
  for i in range(5):
      x, y = grid_to_pixel(i, 0)
      charge_stations.append(ChargeStation(x, y, ROBOT_CHARGE_STATION_IMAGE))

  # Horizontal shelves (vertical stacks)
  for col in range(5, 17):  # 12 columns starting at col 5
      for row_offset in [0, 3, 4, 7, 8, 11, 12]:
          x, y = grid_to_pixel(col, row_offset)
          shelf = Shelf(x, y, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED)
          shelves.append(shelf)


  # Left vertical shelves
  for row in range(2, 11):
      for col in [1, 2]:
          x, y = grid_to_pixel(col, row)
          shelves.append(Shelf(x, y, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED))

  # Right vertical shelves
  for row in range(2, 12):
      for col in [19, 20]:
          x, y = grid_to_pixel(col, row)
          shelves.append(Shelf(x, y, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED))

  # Drop-off platform
  for i in range(4):
    x, y = grid_to_pixel(i, 14)
    dropoff_platform = DropoffPlatform(x, y, DROP_OFF_PLATFORM_IMAGE)
    dropoff_platforms.append(dropoff_platform)

  # Random box
#   box_shelf = random.choice(shelves)
#   box_shelf.has_box = True
#   box_shelf.image = box_shelf.loaded_image

  return shelves, charge_stations, dropoff_platforms
This used just QTable for learning. But then i introduced another robot which is now responsible for completing its task while avoiding robot 1. RObot 2 should learn to not come across the path of robot 1. Since robot 1 is based on QTable, it is deterministic now. RObot 2 uses NN to train and learn its ways. I will provide you with Robot2's information next. import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from train import TrainingEnv, QNetwork, HOME_WAIT_STEPS_REQUIRED
from dual_q_learning_agent import DualQAgent
from constants import *


# ─── Q-TABLE POLICY WRAPPER ───────────────────────────────────────────────────

class QTablePolicy:
    """
    Wraps DualQAgent so it presents the same interface as the NN policy:
        action = policy.select_action(obs_19)

    The Q-table was trained on a 6-field state tuple built from an 8-feature
    observation produced by a different WarehouseEnv.  We reconstruct exactly
    those 6 fields from the 19-feature observation produced by the current env.

    State tuple: (rx, ry, loaded, robot_direction, bx, by)
        rx, ry           — robot grid position (integer)
        loaded           — 0 or 1
        robot_direction  — 0=up 1=down 2=left 3=right (derived from last move)
        bx, by           — current TARGET SHELF grid position (integer, always
                           the shelf — never the dropoff — because that is how
                           DualQAgent._get_state() was defined during training)

    Critical detail: the 19-feature obs switches nav_target to the dropoff
    when the robot is loaded (obs[3,4] become relative-to-dropoff).  We must
    NOT use obs[3,4] directly when loaded — instead we keep a separate
    current_shelf_target that is set at spawn and cleared on delivery.

    Direction mapping (from last_move_x/y in obs[15], obs[16]):
        last_move_y == -1  →  up    (0)
        last_move_y ==  1  →  down  (1)
        last_move_x == -1  →  left  (2)
        last_move_x ==  1  →  right (3)
        both zero           →  up    (0)  (no move yet — matches Robot init)
    """

    # Direction integer mapping used by DualQAgent._get_state()
    _DIRECTION_FROM_LAST_MOVE = {
        ( 0, -1): 0,   # up
        ( 0,  1): 1,   # down
        (-1,  0): 2,   # left
        ( 1,  0): 3,   # right
        ( 0,  0): 0,   # no move yet
    }

    def __init__(self, agent: DualQAgent):
        self.agent = agent
        # Track the current shelf target so bx,by stays correct when loaded.
        # Set externally by TwoAgentWarehouseEnv whenever Agent 1 gets a new target.
        self.current_shelf_target_x: int = ROBOT_HOME_GRID_X
        self.current_shelf_target_y: int = ROBOT_HOME_GRID_Y

    def select_action(self, obs_19: np.ndarray) -> int:
        """
        Converts the 19-feature observation to the 6-field Q-table state tuple
        and returns the greedy action (epsilon is 0 — always exploit).
        Falls back to action 0 (up) for unseen states.
        """
        qtable_obs = self._build_qtable_obs(obs_19)
        state = self.agent._get_state(qtable_obs)

        if state not in self.agent.q_table:
            # Unseen state — fall back to BFS heuristic action index 0.
            return 0

        return int(np.argmax(self.agent.q_table[state]))

    def _build_qtable_obs(self, obs_19: np.ndarray) -> np.ndarray:
        """
        Reconstructs the 8-element observation array that DualQAgent._get_state()
        expects from the 19-feature observation of the current warehouse env.

        Layout expected by DualQAgent._get_state():
            obs[0] = rx      (raw int)
            obs[1] = ry      (raw int)
            obs[2] = loaded  (0 or 1)
            obs[3] = bx      (target shelf x, raw int — always the SHELF)
            obs[4] = by      (target shelf y, raw int — always the SHELF)
            obs[7] = direction (0-3)
        """
        robot_x = round(float(obs_19[0]) * GRID_WIDTH)
        robot_y = round(float(obs_19[1]) * GRID_HEIGHT)
        loaded  = int(round(float(obs_19[2])))

        # bx, by must always be the SHELF, regardless of loaded state.
        # We use the externally maintained current_shelf_target for this.
        box_x = self.current_shelf_target_x
        box_y = self.current_shelf_target_y

        # Direction from last_move_x (obs[15]) and last_move_y (obs[16]).
        last_move_x = int(round(float(obs_19[15])))
        last_move_y = int(round(float(obs_19[16])))
        direction = self._DIRECTION_FROM_LAST_MOVE.get((last_move_x, last_move_y), 0)

        # Build an 8-element array matching DualQAgent's expected layout.
        # Indices 5,6 (dropoff) are not used by _get_state() so we set them 0.
        qtable_obs = np.array([
            robot_x,    # obs[0]  rx
            robot_y,    # obs[1]  ry
            loaded,     # obs[2]  loaded
            box_x,      # obs[3]  bx
            box_y,      # obs[4]  by
            0,          # obs[5]  dropoff_x (unused by _get_state)
            0,          # obs[6]  dropoff_y (unused by _get_state)
            direction,  # obs[7]  robot_direction
        ], dtype=np.float32)

        return qtable_obs


# ─── REWARD / PENALTY TUNING ─────────────────────────────────────────────────
#
#  All reward shaping values for Agent 2 live here so you never need to
#  dig into the class methods to tune them.
#
#  Quick tuning guide (based on training curves):
#
#  Collisions stay high (>20/ep) past ep 300
#      → raise AGENT_COLLISION_PENALTY (e.g. -30) and PROXIMITY_DISTANCE_1_PENALTY (e.g. -4)
#
#  Agent 2 score stays 0 past ep 400  (paralysed, hiding from Agent 1)
#      → lower PROXIMITY_DISTANCE_1_PENALTY (e.g. -1) and remove PROXIMITY_DISTANCE_2_PENALTY (0)
#
#  Agent 2 delivers but still collides
#      → raise AGENT_COLLISION_PENALTY further (e.g. -40); it is too weak vs delivery bonus (+20)
#
#  Episode reward wildly unstable between episodes
#      → tighten YIELDING_BONUS_PROXIMITY_THRESHOLD from 2 to 1
#
# ─────────────────────────────────────────────────────────────────────────────

# Agent 2 tried to step onto Agent 1's cell — hardest constraint.
AGENT_COLLISION_PENALTY        = -20.0

# Agent 2 is adjacent to Agent 1 after stepping (Manhattan distance == 1).
PROXIMITY_DISTANCE_1_PENALTY   = -2.0

# Agent 2 is two cells away from Agent 1 after stepping (Manhattan distance == 2).
PROXIMITY_DISTANCE_2_PENALTY   = -0.5

# Bonus for deliberately yielding (wait/interact) while Agent 1 is close
# AND Agent 2 is still making net progress toward its own delivery goal.
YIELDING_BONUS                 = 1.0

# Manhattan distance threshold that counts as "Agent 1 is very close"
# for the purpose of awarding the yielding bonus.
# Raise to 1 if the bonus fires too broadly and causes reward instability.
YIELDING_BONUS_PROXIMITY_THRESHOLD = 2


# ─── AGENT 2 TRAINING ENVIRONMENT ────────────────────────────────────────────

class Agent2TrainingEnv(TrainingEnv):
    """
    Extends TrainingEnv for Agent 2. Adds 6 extra observation features
    describing Agent 1's CURRENT position AND its NEXT position so Agent 2
    can see where Agent 1 is heading and vacate proactively.

    Extra observation appended (in order):
        [19] agent1_relative_x      : (agent1.grid_x - agent2.grid_x) / GRID_WIDTH
        [20] agent1_relative_y      : (agent1.grid_y - agent2.grid_y) / GRID_HEIGHT
        [21] agent1_loaded          : float(agent1 is carrying a box)
        [22] agent1_returning_home  : float(agent1 is in home-return phase)
        [23] agent1_next_relative_x : (agent1_next_x - agent2.grid_x) / GRID_WIDTH
        [24] agent1_next_relative_y : (agent1_next_y - agent2.grid_y) / GRID_HEIGHT

    Features [23,24] are the critical ones: they tell Agent 2 WHERE Agent 1
    will be NEXT step — computed from Agent 1's action before either agent
    moves. This lets Agent 2 vacate a cell before Agent 1 arrives rather than
    being forced out after the fact.
    """

    # Base observation size from WarehouseEnv: 7 + 8 + 2 + 2 = 19
    AGENT1_EXTRA_FEATURES = 6
    EXTENDED_OBSERVATION_SIZE = 19 + AGENT1_EXTRA_FEATURES  # = 25

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        from gymnasium import spaces
        self.observation_size = self.EXTENDED_OBSERVATION_SIZE
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.EXTENDED_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        # Agent 1 current state — injected before every step.
        self.agent1_grid_x = ROBOT_HOME_GRID_X
        self.agent1_grid_y = ROBOT_HOME_GRID_Y
        self.agent1_loaded = False
        self.agent1_returning_home = True
        # Agent 1 NEXT position — where it will land this step.
        # Computed from Agent 1's action BEFORE either agent moves.
        self.agent1_next_grid_x = ROBOT_HOME_GRID_X
        self.agent1_next_grid_y = ROBOT_HOME_GRID_Y

    # ── observation ──────────────────────────────────────────────────────────

    def _get_observation(self):
        """Base 19-feature observation extended with 6 Agent 1 features."""
        base_observation = super()._get_observation()   # np.float32 array, len 19

        agent2_robot = self.robot
        agent1_relative_x = (self.agent1_grid_x - agent2_robot.grid_x) / GRID_WIDTH
        agent1_relative_y = (self.agent1_grid_y - agent2_robot.grid_y) / GRID_HEIGHT
        agent1_next_relative_x = (self.agent1_next_grid_x - agent2_robot.grid_x) / GRID_WIDTH
        agent1_next_relative_y = (self.agent1_next_grid_y - agent2_robot.grid_y) / GRID_HEIGHT

        extra_features = np.array([
            agent1_relative_x,
            agent1_relative_y,
            float(self.agent1_loaded),
            float(self.agent1_returning_home),
            agent1_next_relative_x,
            agent1_next_relative_y,
        ], dtype=np.float32)

        return np.concatenate([base_observation, extra_features])

    # ── internal helper ───────────────────────────────────────────────────────

    def update_agent1_state(self, agent1_grid_x, agent1_grid_y,
                            agent1_loaded, agent1_returning_home,
                            agent1_next_grid_x, agent1_next_grid_y):
        """
        Called by TwoAgentWarehouseEnv after Agent 1's action is selected
        but BEFORE either agent steps. agent1_next_grid_x/y is Agent 1's
        predicted landing cell — the look-ahead that enables proactive vacating.
        """
        self.agent1_grid_x = agent1_grid_x
        self.agent1_grid_y = agent1_grid_y
        self.agent1_loaded = agent1_loaded
        self.agent1_returning_home = agent1_returning_home
        self.agent1_next_grid_x = agent1_next_grid_x
        self.agent1_next_grid_y = agent1_next_grid_y


# ─── TWO-AGENT WAREHOUSE ENVIRONMENT ─────────────────────────────────────────

class TwoAgentWarehouseEnv:
    """
    Wraps two independent WarehouseEnv instances and steps them simultaneously.

    Agent 1: frozen policy (eval, no_grad). Acts as a predictable moving
             obstacle. Uses its own TrainingEnv so its BFS / home-return
             / reward logic is completely unchanged.

    Agent 2: actively trained. Uses Agent2TrainingEnv, which extends
             TrainingEnv with Agent 1's position in the observation and
             collision/proximity penalties in the reward.

    Collision resolution — Agent 1 has priority:
        After both agents select their actions, Agent 1 always moves freely.
        If Agent 2's intended next cell would equal Agent 1's resulting cell,
        Agent 2 is blocked (stays in place) and receives AGENT_COLLISION_PENALTY.
        This makes Agent 1 a fully predictable obstacle: Agent 2 must learn
        to route around it.

    Separate spawn zones (optional but recommended):
        Agent 1 uses columns 5–10; Agent 2 uses columns 11–16.
        This prevents them from competing for the same shelf pickup points.
        Configured via TrainingEnv.SPAWN_COLUMN_RANGE on each sub-env.
    """

    def __init__(self,
                 agent1_qtable_path: str,       # path to warehouse_data.pkl
                 agent1_qtable_folder: str,     # folder containing the pkl
                 compute_device: torch.device,
                 render_mode=None):

        # ── Agent 1 sub-environment ──────────────────────────────────────────
        self.agent1_env = TrainingEnv(render_mode=None)
        self.agent1_env.SPAWN_COLUMN_RANGE = {5, 6, 7, 8, 9, 10}
        self.agent1_env.SPAWN_ROW_RANGE = None

        # ── Agent 2 sub-environment ──────────────────────────────────────────
        self.agent2_env = Agent2TrainingEnv(render_mode=render_mode)
        self.agent2_env.SPAWN_COLUMN_RANGE = {11, 12, 13, 14, 15, 16}
        self.agent2_env.SPAWN_ROW_RANGE = None

        self.render_mode = render_mode
        self.compute_device = compute_device

        # ── Load Agent 1's Q-table and wrap it ───────────────────────────────
        # DualQAgent is epsilon-greedy during training, but here we always
        # exploit (epsilon effectively 0) — the wrapper's select_action()
        # goes straight to argmax on the Q-table.
        raw_qtable_agent = DualQAgent(action_dim=self.agent1_env.action_space.n)
        raw_qtable_agent.load_tables(agent1_qtable_folder, agent1_qtable_path)
        raw_qtable_agent.epsilon = 0.0   # pure exploitation — no random actions

        self.agent1_policy = QTablePolicy(raw_qtable_agent)

        print(f"  ✅ Agent 1 Q-table loaded from '{agent1_qtable_folder}/{agent1_qtable_path}' (frozen).")
        print(f"     Q-table size: {len(raw_qtable_agent.q_table):,} states")

        # ── Expose Agent 2 spaces for the training loop ───────────────────────
        self.observation_space = self.agent2_env.observation_space
        self.action_space = self.agent2_env.action_space

        # ── Internal state ────────────────────────────────────────────────────
        self._agent1_current_observation = None
        self.score = 0
        self.agent2_collision_count = 0

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Resets both sub-environments. Retries until the two robots start
        on different grid cells (very unlikely to collide but guaranteed safe).
        """
        agent1_obs, _ = self.agent1_env.reset(seed=seed, options=options)
        agent2_obs, agent2_info = self.agent2_env.reset(seed=seed, options=options)

        # Guarantee distinct starting positions.
        max_retries = 20
        for _ in range(max_retries):
            if (self.agent1_env.robot.grid_x != self.agent2_env.robot.grid_x or
                    self.agent1_env.robot.grid_y != self.agent2_env.robot.grid_y):
                break
            agent2_obs, agent2_info = self.agent2_env.reset(options=options)

        self._agent1_current_observation = agent1_obs
        self.score = 0
        self.agent2_collision_count = 0

        # Tell the Q-table policy which shelf Agent 1 is currently targeting
        # so bx,by in the state tuple stays correct throughout the episode.
        self.agent1_policy.current_shelf_target_x = self.agent1_env.target_grid_x
        self.agent1_policy.current_shelf_target_y = self.agent1_env.target_grid_y

        # Sync Agent 1's state into Agent 2's observation.
        # Compute Agent 1's first action preview so the initial observation
        # already has a valid look-ahead instead of the default home coords.
        agent1_first_action = self.agent1_policy.select_action(
            self._agent1_current_observation
        )
        agent1_first_next_x, agent1_first_next_y = self._predict_next_position(
            self.agent1_env.robot.grid_x,
            self.agent1_env.robot.grid_y,
            agent1_first_action,
            self.agent1_env.obstacle_positions,
        )
        self._sync_agent1_state_into_agent2_env(
            agent1_next_x=agent1_first_next_x,
            agent1_next_y=agent1_first_next_y,
        )
        agent2_obs = self.agent2_env._get_observation()

        return agent2_obs, agent2_info

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, agent2_action: int):
        """
        Steps both agents simultaneously.

        Order of operations:
            1. Query Agent 1's frozen policy → agent1_action.
            2. Compute where Agent 1 will land (without stepping yet).
            3. Check if Agent 2's intended move would land on Agent 1's
               new cell — if so, block Agent 2 and apply collision penalty.
            4. Step Agent 1's env (Agent 1 always moves freely).
            5. Step Agent 2's env (possibly with overridden collision result).
            6. Apply proximity / yielding reward shaping on top.
            7. Sync Agent 1 state into Agent 2's env for next observation.
        """

        agent1_robot = self.agent1_env.robot
        agent2_robot = self.agent2_env.robot

        # ── 1. Agent 1 selects its action from Q-table (pure exploitation) ─────
        # QTablePolicy.select_action() converts the 19-feature obs down to the
        # 6-field state tuple the Q-table was trained on, then returns argmax.
        # No gradients, no GPU — pure dictionary lookup.
        agent1_action = self.agent1_policy.select_action(
            self._agent1_current_observation
        )

        # ── 2. Predict where Agent 1 will land ───────────────────────────────
        agent1_predicted_next_x, agent1_predicted_next_y = (
            self._predict_next_position(
                agent1_robot.grid_x, agent1_robot.grid_y,
                agent1_action,
                self.agent1_env.obstacle_positions,
            )
        )

        # ── 3. Detect if Agent 2's move would collide with Agent 1 ───────────
        agent2_predicted_next_x, agent2_predicted_next_y = (
            self._predict_next_position(
                agent2_robot.grid_x, agent2_robot.grid_y,
                agent2_action,
                self.agent2_env.obstacle_positions,
            )
        )

        # Condition A: Agent 2 moves INTO Agent 1's next cell.
        agent2_moves_into_agent1_next_cell = (
            agent2_predicted_next_x == agent1_predicted_next_x and
            agent2_predicted_next_y == agent1_predicted_next_y
        )

        # Condition B: Agent 1 moves INTO Agent 2's CURRENT cell.
        # Agent 1 is deterministic and ignores Agent 2 — it will walk straight
        # through. Agent 2 must learn to vacate before this happens.
        # We detect this here and force Agent 2 to move away.
        agent1_moves_into_agent2_current_cell = (
            agent1_predicted_next_x == agent2_robot.grid_x and
            agent1_predicted_next_y == agent2_robot.grid_y
        )

        agent2_would_collide_with_agent1 = (
            agent2_moves_into_agent1_next_cell or
            agent1_moves_into_agent2_current_cell
        )

        # ── 4. Step Agent 1 freely ────────────────────────────────────────────
        (
            agent1_next_obs,
            _agent1_reward,
            agent1_terminated,
            agent1_truncated,
            _,
        ) = self.agent1_env.step(agent1_action)
        self._agent1_current_observation = agent1_next_obs

        if agent1_terminated or agent1_truncated:
            # Reset Agent 1 silently so it keeps acting as a moving obstacle.
            agent1_reset_obs, _ = self.agent1_env.reset()
            self._agent1_current_observation = agent1_reset_obs

        # Keep the Q-table policy's shelf target in sync with Agent 1's env.
        # target_grid_x/y changes whenever Agent 1 picks up a box (switches to
        # dropoff) or spawns a new target — we always want the SHELF coords.
        # TrainingEnv stores the shelf in target_grid_x/y until pickup, at which
        # point returning_home becomes True and target switches to home.
        # When NOT loaded and NOT returning home, target IS the shelf.
        if not self.agent1_env.robot.loaded and not self.agent1_env.returning_home:
            self.agent1_policy.current_shelf_target_x = self.agent1_env.target_grid_x
            self.agent1_policy.current_shelf_target_y = self.agent1_env.target_grid_y

        # ── 5. Step Agent 2 ───────────────────────────────────────────────────
        #
        # If Agent 2's intended move would land on Agent 1's new cell, we
        # override Agent 2's action to WAIT (5) so the base step logic keeps
        # Agent 2 in place. The collision penalty is added on top below.
        #
        effective_agent2_action = agent2_action
        if agent2_would_collide_with_agent1:
            if agent1_moves_into_agent2_current_cell and agent2_action < 4:
                # Agent 1 is walking into Agent 2's cell. Agent 2 must MOVE —
                # not wait. Waiting keeps it in place and the collision still
                # happens. We keep Agent 2's intended movement action so it
                # tries to step away; if the move itself is invalid (wall),
                # the base step will block it and apply a wall-collision penalty
                # on top of our evacuation penalty below.
                effective_agent2_action = agent2_action
            elif agent2_moves_into_agent1_next_cell and agent2_action < 4:
                # Agent 2 is walking into Agent 1's next cell — force wait.
                effective_agent2_action = 5
            self.agent2_collision_count += 1

        (
            agent2_next_obs,
            agent2_base_reward,
            agent2_terminated,
            agent2_truncated,
            agent2_info,
        ) = self.agent2_env.step(effective_agent2_action)

        # ── 6. Reward shaping on top of base reward ───────────────────────────
        collision_and_proximity_reward = self._calculate_agent2_extra_reward(
            agent2_robot=agent2_robot,
            agent1_next_grid_x=self.agent1_env.robot.grid_x,
            agent1_next_grid_y=self.agent1_env.robot.grid_y,
            agent2_action_was_blocked=agent2_would_collide_with_agent1,
            agent2_action_original=agent2_action,
            agent2_base_reward=agent2_base_reward,
        )

        total_agent2_reward = agent2_base_reward + collision_and_proximity_reward

        # ── 7. Sync Agent 1 state (with next-step look-ahead) for next obs ─────
        # After both agents have moved, Agent 1's NEW current position becomes
        # the current position for the next step, and we re-predict Agent 1's
        # next action now so Agent 2's observation has fresh look-ahead.
        agent1_next_action_preview = self.agent1_policy.select_action(
            self._agent1_current_observation
        )
        agent1_preview_next_x, agent1_preview_next_y = self._predict_next_position(
            self.agent1_env.robot.grid_x,
            self.agent1_env.robot.grid_y,
            agent1_next_action_preview,
            self.agent1_env.obstacle_positions,
        )
        self._sync_agent1_state_into_agent2_env(
            agent1_next_x=agent1_preview_next_x,
            agent1_next_y=agent1_preview_next_y,
        )
        agent2_next_obs = self.agent2_env._get_observation()

        self.score = self.agent2_env.score

        return agent2_next_obs, total_agent2_reward, agent2_terminated, agent2_truncated, agent2_info

    # ── reward shaping ────────────────────────────────────────────────────────

    def _calculate_agent2_extra_reward(
        self,
        agent2_robot,
        agent1_next_grid_x: int,
        agent1_next_grid_y: int,
        agent2_action_was_blocked: bool,
        agent2_action_original: int,
        agent2_base_reward: float,
    ) -> float:
        """
        Computes the extra reward adjustment for Agent 2 based on its
        spatial relationship to Agent 1 after both have stepped.

        Returns a float that is ADDED to the base step reward.
        """
        extra_reward = 0.0

        # ── Collision / evacuation penalty ───────────────────────────────────
        # Fires in two cases:
        #   A) Agent 2 tried to move into Agent 1's next cell (blocked → wait).
        #   B) Agent 1 is walking into Agent 2's current cell (evacuation needed).
        # Both are penalised equally — Agent 2 should have moved away earlier.
        if agent2_action_was_blocked:
            extra_reward += AGENT_COLLISION_PENALTY
            return extra_reward   # no need to evaluate proximity on top

        # ── Proximity penalties (post-step position) ──────────────────────────
        manhattan_distance_to_agent1 = (
            abs(agent2_robot.grid_x - agent1_next_grid_x) +
            abs(agent2_robot.grid_y - agent1_next_grid_y)
        )

        if manhattan_distance_to_agent1 == 1:
            extra_reward += PROXIMITY_DISTANCE_1_PENALTY
        elif manhattan_distance_to_agent1 == 2:
            extra_reward += PROXIMITY_DISTANCE_2_PENALTY

        # ── Yielding bonus ────────────────────────────────────────────────────
        # Agent 2 deliberately waited or took a non-direct action while
        # Agent 1 was very close AND Agent 2 is still making forward progress
        # on its own goal (positive base reward means net BFS improvement).
        agent2_chose_to_yield = agent2_action_original in (4, 5)  # interact or wait
        agent1_is_very_close = manhattan_distance_to_agent1 <= YIELDING_BONUS_PROXIMITY_THRESHOLD
        agent2_still_progressing = agent2_base_reward > 0.0

        if agent2_chose_to_yield and agent1_is_very_close and agent2_still_progressing:
            extra_reward += YIELDING_BONUS

        return extra_reward

    # ── helpers ───────────────────────────────────────────────────────────────

    def _predict_next_position(
        self,
        current_grid_x: int,
        current_grid_y: int,
        action: int,
        obstacle_positions: set,
    ):
        """
        Returns the grid cell an agent will occupy after taking `action`,
        without mutating any environment state. Mirrors the movement logic
        in WarehouseEnv.step exactly.

        For non-movement actions (interact=4, wait=5), the agent stays put.
        For movement into walls/obstacles, the agent stays put.
        """
        if action >= 4:
            return current_grid_x, current_grid_y

        direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        delta_x, delta_y = direction_deltas[action]
        next_x = current_grid_x + delta_x
        next_y = current_grid_y + delta_y

        is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
        is_passable = (next_x, next_y) not in obstacle_positions

        if is_in_bounds and is_passable:
            return next_x, next_y

        return current_grid_x, current_grid_y

    def _sync_agent1_state_into_agent2_env(self,
                                           agent1_next_x: int,
                                           agent1_next_y: int):
        """
        Pushes Agent 1's current position, status, AND predicted next position
        into Agent 2's environment so the observation includes the look-ahead.
        Call this AFTER Agent 1's action is known but BEFORE Agent 2 acts.
        """
        self.agent2_env.update_agent1_state(
            agent1_grid_x=self.agent1_env.robot.grid_x,
            agent1_grid_y=self.agent1_env.robot.grid_y,
            agent1_loaded=self.agent1_env.robot.loaded,
            agent1_returning_home=self.agent1_env.returning_home,
            agent1_next_grid_x=agent1_next_x,
            agent1_next_grid_y=agent1_next_y,
        )

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        """
        Renders Agent 2's environment (which owns the pygame screen).
        Agent 1 is drawn as a second robot on top using a distinct visual.
        """
        if self.render_mode is None:
            return

        # Let Agent 2's env do all the base rendering (shelves, dropoff, etc.)
        self.agent2_env.render()

        # Draw Agent 1 on Agent 2's screen as a distinct overlay.
        if self.agent2_env.screen is not None:
            import pygame

            agent1_pixel_x = (
                PADDING_BORDER + self.agent1_env.robot.grid_x * GRID_SPACING
            )
            agent1_pixel_y = (
                PADDING_BORDER + self.agent1_env.robot.grid_y * GRID_SPACING
            )

            # Use the side-facing robot image for Agent 1 to visually
            # distinguish it from Agent 2 (which uses the vertical image).
            agent1_image = (
                ROBOT_IMAGE_SIDE_BOX
                if self.agent1_env.robot.loaded
                else ROBOT_IMAGE_SIDE
            )
            center_offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
            center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
            self.agent2_env.screen.blit(
                agent1_image,
                (agent1_pixel_x + center_offset_x, agent1_pixel_y + center_offset_y),
            )

            # Draw a cyan border around Agent 1 to make it unmistakably distinct.
            pygame.draw.rect(
                self.agent2_env.screen,
                (0, 220, 220),
                (agent1_pixel_x, agent1_pixel_y, TILE_SIZE, TILE_SIZE),
                2,
            )

            pygame.display.flip()

    def heuristic_action(self):
        """Exposes Agent 2's heuristic for use in the training loop."""
        return self.agent2_env.heuristic_action() this is the new env. I will provide more next. Wait


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
    train() thisis train_agent2.py, """
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
    main() this is test_two_agents and import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
from world import create_map
from constants import *
from robot import Robot
import pygame
import sys

from pygame_manager import ensure_init


class RewardManager:
    """Centralized reward shaping logic to keep the environment clean."""

    def __init__(self):
        self.step_penalty = -0.05
        self.collision_penalty = -3
        self.wait_penalty = -5
        self.failed_interact_penalty = -4

        self.progress_reward_scale = 1.0
        self.regress_penalty_scale = 4.0
        self.pickup_bonus = 10.0
        self.delivery_bonus = 20.0
        self.proximity_bonus = 0.5



    def calculate(self, event, distance_delta=0, proximity_bonus=0.0):
        if event == "collision":
            return self.collision_penalty
        if event == "pickup":
            return self.pickup_bonus
        if event == "delivery":
            return self.delivery_bonus
        if event == "wait":
            return self.wait_penalty
        if event == "failed_interact":
            return self.failed_interact_penalty

        if distance_delta >= 0:
            progress_reward = distance_delta * self.progress_reward_scale
        else:
            progress_reward = distance_delta * self.regress_penalty_scale

        return progress_reward + self.step_penalty + proximity_bonus


ROBOT_HOME_GRID_X = 2
ROBOT_HOME_GRID_Y = 0


class WarehouseEnv(gym.Env):

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.reward_manager = RewardManager()

        if render_mode == "human":
            import pygame
            if pygame.get_init():
                self.screen = pygame.display.get_surface()  # grab existing window
                self.clock = pygame.time.Clock()
            else:
                pygame.init()
                window_width = GRID_WIDTH * GRID_SPACING + (2 * PADDING_BORDER)
                window_height = GRID_HEIGHT * GRID_SPACING + (2 * PADDING_BORDER)
                self.screen = pygame.display.set_mode((window_width, window_height))
                self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()

        self.obstacle_positions = {
            self._to_grid_coords(obj)
            for obj in self.shelves + self.dropoff_platforms
        }

        # Observation: [x, y, loaded, target_dx, target_dy, dropoff_dx, dropoff_dy]
        #              + 8 adjacent cells + last_move_x + last_move_y + can_pickup + can_deliver
        self.observation_size = 7 + 8 + 2 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.observation_size,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(6)

        central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_grid_x, self.dropoff_grid_y = self._to_grid_coords(central_platform)
        self.dropoff_distance_map = self._bfs_distance_map(
            self.dropoff_grid_x, self.dropoff_grid_y
        )
        self.home_distance_map = self._bfs_distance_map(
            ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y,
            target_is_walkable=True
        )

        self.total_episodes_elapsed = 0

        self.target_grid_x = 0
        self.target_grid_y = 0
        self.target_distance_map = {}

    def _to_grid_coords(self, obj):
        grid_x = round((obj.x - PADDING_BORDER) / GRID_SPACING)
        grid_y = round((obj.y - PADDING_BORDER) / GRID_SPACING)
        return grid_x, grid_y

    def _bfs_distance_map(self, start_grid_x, start_grid_y, target_is_walkable=False):
        """
        Computes BFS shortest distances to the target cell.
        target_is_walkable=False: seeds from neighbors (for shelves/dropoffs the robot cannot stand on)
        target_is_walkable=True:  seeds from the cell itself (for walkable targets like home)
        """
        distance_map = {}
        search_queue = deque()

        if target_is_walkable:
            distance_map[(start_grid_x, start_grid_y)] = 0
            search_queue.append((start_grid_x, start_grid_y, 0))
        else:
            for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_x = start_grid_x + delta_x
                neighbor_y = start_grid_y + delta_y
                is_in_bounds = 0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT
                is_walkable = (neighbor_x, neighbor_y) not in self.obstacle_positions
                if is_in_bounds and is_walkable and (neighbor_x, neighbor_y) not in distance_map:
                    distance_map[(neighbor_x, neighbor_y)] = 0
                    search_queue.append((neighbor_x, neighbor_y, 0))

        while search_queue:
            current_x, current_y, current_dist = search_queue.popleft()
            for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_x = current_x + delta_x
                neighbor_y = current_y + delta_y
                is_in_bounds = 0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT
                is_not_obstacle = (neighbor_x, neighbor_y) not in self.obstacle_positions
                if is_in_bounds and is_not_obstacle and (neighbor_x, neighbor_y) not in distance_map:
                    distance_map[(neighbor_x, neighbor_y)] = current_dist + 1
                    search_queue.append((neighbor_x, neighbor_y, current_dist + 1))

        return distance_map


    def _clear_target_shelf(self):
        """Remove the box from the shelf that is currently the target."""
        for shelf in self.shelves:
            shelf_grid = self._to_grid_coords(shelf)
            if shelf_grid == (self.target_grid_x, self.target_grid_y):
                shelf.has_box = False
                shelf.image = shelf.empty_image
                break


    def _spawn_new_target(self):
        """Clears all shelves and places a box on a new random shelf."""
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        target_shelf = random.choice(self.shelves)
        target_shelf.has_box = True
        target_shelf.image = target_shelf.loaded_image

        self.target_grid_x, self.target_grid_y = self._to_grid_coords(target_shelf)
        self.target_distance_map = self._bfs_distance_map(
            self.target_grid_x, self.target_grid_y
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.recent_positions = deque(maxlen=8)
        self.last_action = -1

        central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_grid_x, self.dropoff_grid_y = self._to_grid_coords(central_platform)

        self._spawn_new_target()

        blocked_positions = self.obstacle_positions | {
            self._to_grid_coords(platform) for platform in self.dropoff_platforms
        }

        all_valid_spawn_positions = [
            (grid_x, grid_y)
            for grid_x in range(GRID_WIDTH)
            for grid_y in range(GRID_HEIGHT)
            if (grid_x, grid_y) not in blocked_positions
        ]

        robot_start_x, robot_start_y = random.choice(all_valid_spawn_positions)
        self.robot = Robot(start_x=robot_start_x, start_y=robot_start_y)

        self.total_episodes_elapsed += 1

        return self._get_observation(), {}

    def _get_observation(self):
        robot = self.robot

        last_move_x = 0.0
        last_move_y = 0.0
        if self.last_action == 0:
            last_move_y = -1.0
        elif self.last_action == 1:
            last_move_y = 1.0
        elif self.last_action == 2:
            last_move_x = -1.0
        elif self.last_action == 3:
            last_move_x = 1.0

        if robot.loaded:
            nav_target_x, nav_target_y = self.dropoff_grid_x, self.dropoff_grid_y
        else:
            nav_target_x, nav_target_y = self.target_grid_x, self.target_grid_y

        observation = [
            robot.grid_x / GRID_WIDTH,
            robot.grid_y / GRID_HEIGHT,
            float(robot.loaded),
            (nav_target_x - robot.grid_x) / GRID_WIDTH,
            (nav_target_y - robot.grid_y) / GRID_HEIGHT,
            (self.dropoff_grid_x - robot.grid_x) / GRID_WIDTH,
            (self.dropoff_grid_y - robot.grid_y) / GRID_HEIGHT,
        ]

        for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                  (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            neighbor_x = robot.grid_x + delta_x
            neighbor_y = robot.grid_y + delta_y
            is_out_of_bounds = not (0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT)
            is_obstacle = (neighbor_x, neighbor_y) in self.obstacle_positions
            observation.append(1.0 if is_obstacle or is_out_of_bounds else 0.0)

        can_pickup = float(
            not robot.loaded
            and abs(robot.grid_x - self.target_grid_x) + abs(robot.grid_y - self.target_grid_y) == 1
        )
        can_deliver = float(
            robot.loaded
            and abs(robot.grid_x - self.dropoff_grid_x) + abs(robot.grid_y - self.dropoff_grid_y) == 1
        )

        observation.append(last_move_x)
        observation.append(last_move_y)
        observation.append(can_pickup)
        observation.append(can_deliver)

        return np.array(observation, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        robot = self.robot

        active_distance_map = (
            self.dropoff_distance_map if robot.loaded else self.target_distance_map
        )
        distance_before = active_distance_map.get((robot.grid_x, robot.grid_y), 50)

        current_event = "move"
        earned_proximity_bonus = 0.0

        if action < 4:
            direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            delta_x, delta_y = direction_deltas[action]
            next_x = robot.grid_x + delta_x
            next_y = robot.grid_y + delta_y

            is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
            is_passable = (next_x, next_y) not in self.obstacle_positions

            if is_in_bounds and is_passable:
                robot.grid_x, robot.grid_y = next_x, next_y
            else:
                current_event = "collision"

            if not robot.loaded:
                dist_to_target = self.target_distance_map.get((robot.grid_x, robot.grid_y), 50)
                if dist_to_target <= 2:
                    earned_proximity_bonus = self.reward_manager.proximity_bonus

        elif action == 4:
            dist_to_dropoff = (
                abs(robot.grid_x - self.dropoff_grid_x)
                + abs(robot.grid_y - self.dropoff_grid_y)
            )
            is_directly_adjacent_to_target = (
                (robot.grid_x == self.target_grid_x and abs(robot.grid_y - self.target_grid_y) == 1) or
                (robot.grid_y == self.target_grid_y and abs(robot.grid_x - self.target_grid_x) == 1)
            )

            if not robot.loaded and is_directly_adjacent_to_target:
                robot.loaded = True
                current_event = "pickup"
                self._clear_target_shelf()

            elif robot.loaded and dist_to_dropoff == 1:
                robot.loaded = False
                self.score += 1
                current_event = "delivery"
                self._on_delivery()

            else:
                current_event = "failed_interact"
        else:
            current_event = "wait"

        distance_after = active_distance_map.get((robot.grid_x, robot.grid_y), 50)
        reward = self.reward_manager.calculate(
            current_event,
            distance_delta=(distance_before - distance_after),
            proximity_bonus=earned_proximity_bonus,
        )

        if action < 4:
            current_position = (robot.grid_x, robot.grid_y)

            pos_list = list(self.recent_positions)
            visit_count = pos_list.count(current_position)
            is_pingpong = len(pos_list) >= 2 and current_position == pos_list[-2]
            
            if is_pingpong and visit_count >= 2:
                reward -= visit_count * 2.0  # scales: 4.0, 6.0, 8.0 ...
            elif visit_count >= 2:
                reward -= visit_count * 0.5  # mild: 1.0, 1.5, 2.0 ...

            self.recent_positions.append(current_position)

        is_done = self.steps >= 500
        self.last_action = action

        return self._get_observation(), reward, is_done, False, {}

    def _on_delivery(self):
        """
        Called after every successful delivery.
        Base behavior: immediately spawn next target.
        Subclasses override this to inject home-return or queue logic.
        """
        self._spawn_new_target()

    def heuristic_action(self):
        robot = self.robot
        active_distance_map = (
            self.dropoff_distance_map if robot.loaded else self.target_distance_map
        )
        current_distance = active_distance_map.get((robot.grid_x, robot.grid_y), 50)
        direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        best_action = None
        best_distance = current_distance

        for action_index, (delta_x, delta_y) in enumerate(direction_deltas):
            next_x = robot.grid_x + delta_x
            next_y = robot.grid_y + delta_y
            is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
            is_passable = (next_x, next_y) not in self.obstacle_positions
            if is_in_bounds and is_passable:
                neighbor_distance = active_distance_map.get((next_x, next_y), 50)
                if neighbor_distance < best_distance:
                    best_distance = neighbor_distance
                    best_action = action_index

        if not robot.loaded:
            dist_to_target_shelf = (
                abs(robot.grid_x - self.target_grid_x)
                + abs(robot.grid_y - self.target_grid_y)
            )
            if dist_to_target_shelf == 1 and best_action is None:
                return 4
        elif robot.loaded:
            dist_to_dropoff = (
                abs(robot.grid_x - self.dropoff_grid_x)
                + abs(robot.grid_y - self.dropoff_grid_y)
            )
            if dist_to_dropoff == 1:
                return 4

        if best_action is None:
            return random.randint(0, 3)

        return best_action
    

    def _handle_pygame_events(self):
        """Process pygame events to keep the window responsive."""
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            window_width = GRID_WIDTH * GRID_SPACING + (2 * PADDING_BORDER)
            window_height = GRID_HEIGHT * GRID_SPACING + (2 * PADDING_BORDER)
            self.screen = ensure_init(window_width, window_height, "Warehouse")
            self.clock = pygame.time.Clock()


        self._handle_pygame_events()
        self.screen.fill((30, 30, 30))

        for charge_station in self.charge_stations:
            self.screen.blit(charge_station.image, (charge_station.x, charge_station.y))

        for dropoff_platform in self.dropoff_platforms:
            self.screen.blit(dropoff_platform.image, (dropoff_platform.x, dropoff_platform.y))

        for shelf in self.shelves:
            shelf_grid_position = self._to_grid_coords(shelf)
            is_current_target = (
                not self.robot.loaded
                and shelf_grid_position == (self.target_grid_x, self.target_grid_y)
            )
            self.screen.blit(shelf.shadow_image, (shelf.x - 1, shelf.y + 4))
            if is_current_target:
                pygame.draw.rect(
                    self.screen, (255, 255, 0),
                    (shelf.x - 2, shelf.y - 2, TILE_SIZE + 4, TILE_SIZE + 4), 2
                )
            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        robot_pixel_x = PADDING_BORDER + self.robot.grid_x * GRID_SPACING
        robot_pixel_y = PADDING_BORDER + self.robot.grid_y * GRID_SPACING

        robot_image = ROBOT_IMAGE_VERTICAL_BOX if self.robot.loaded else ROBOT_IMAGE_VERTICAL
        center_offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
        center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
        self.screen.blit(robot_image, (robot_pixel_x + center_offset_x, robot_pixel_y + center_offset_y))

        pygame.display.flip()
        self.clock.tick(20) thsi is warehouse_env for the Newer NN version. This approach has been really messy and has fragmented code and reward system. import sys
import pygame
from constants import PADDING_BORDER, GRID_SPACING, GRID_WIDTH, GRID_HEIGHT


class ClickController:
    """
    Listens for mouse clicks on the pygame window and converts them
    to shelf grid coordinates, then enqueues them in the environment.
    """

    def __init__(self, env):
        self.env = env

    def handle_pygame_events(self):
        """
        Call this every step. Processes all pending pygame events
        and enqueues any shelf clicks into the environment's target queue.
        """
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if pygame_event.type == pygame.MOUSEBUTTONDOWN and pygame_event.button == 1:
                mouse_pixel_x, mouse_pixel_y = pygame_event.pos
                self._handle_shelf_click(mouse_pixel_x, mouse_pixel_y)

    def _handle_shelf_click(self, mouse_pixel_x, mouse_pixel_y):
        """Convert pixel click position to grid coords and enqueue if it's a shelf."""
        clicked_grid_x = round((mouse_pixel_x - PADDING_BORDER) / GRID_SPACING)
        clicked_grid_y = round((mouse_pixel_y - PADDING_BORDER) / GRID_SPACING)

        is_in_bounds = 0 <= clicked_grid_x < GRID_WIDTH and 0 <= clicked_grid_y < GRID_HEIGHT
        if not is_in_bounds:
            print(f"  ✗ Click out of bounds, ignored.")
            return

        for shelf in self.env.shelves:
            shelf_grid_x, shelf_grid_y = self.env._to_grid_coords(shelf)
            if shelf_grid_x == clicked_grid_x and shelf_grid_y == clicked_grid_y:
                # Visually mark the shelf as having a box immediately
                shelf.has_box = True
                shelf.image = shelf.loaded_image
                self.env.enqueue_target(clicked_grid_x, clicked_grid_y)
                return

        print(f"  ✗ No shelf at grid ({clicked_grid_x}, {clicked_grid_y}), click ignored.") I want to continue the training of robot2 while robot 1 does its thing and work. A common queue should be responsible for assigning tasks to the robots. They should go to the shelf and drop the item and then go back to charge station for 2 ticks and then then are assigned other targets. See extremely carefully and tell me what problems do you see when implementing training with this approach. when i run the train_agent2.py, it shows the following log  [ep4 s1980] R1 ret=True pos=( 2,13) ld=0 sc=12 | R2 fetching   pos=( 7, 6) ld=0 tgt=(10,7) act=3 rew=0.95 sc=27
  [ep4 s2000] R1 ret=False pos=( 3, 3) ld=0 sc=0 | R2 delivering pos=( 2,12) ld=1 tgt=(10,7) act=2 rew=0.95 sc=27
Ep    4 | R2:27 R1: 0 | Coll: 29 | Rew: 206.20 eps:0.693 buf:10000
Ep    5 | R2:29 R1: 0 | Coll: 15 | Rew: 810.45 eps:0.692 buf:12000
. First of all observe what is causing this problem and then recommend proper solution with clear instructions