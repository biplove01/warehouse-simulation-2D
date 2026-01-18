import gymnasium as gym
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

    # Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4:Interact (E)
    self.action_space = spaces.Discrete(5)

    # Observation: [robot_x, robot_y, robot_loaded, nearest_box_x, nearest_box_y, dropofff_x dropoff_y]
    self.observation_space = spaces.Box(
                                        low=np.array([0, 0, 0, 0, 0, 0, 0, 8], dtype=np.float32),
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
    self.max_steps = 800


  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.score = 0
    self.current_step = 0
    self.reward = 0
    self.state_history = deque(maxlen=8)

    self.shelves, self.charge_stations, self.dropoff_platforms = create_map()

    # collects blocked grid cells
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

    box_pos = [0, 0]
    for s in self.shelves:
      if s.has_box:
        box_pos = [round((s.x - PADDING_BORDER)/GRID_SPACING),
                   round((s.y - PADDING_BORDER)/GRID_SPACING)]
        break

    drop_pos = [(round(self.dropoff_platforms[0].x - PADDING_BORDER)/GRID_SPACING),
                (round(self.dropoff_platforms[0].y-PADDING_BORDER)/GRID_SPACING)]

    return np.array([
      self.robot.grid_x,
      self.robot.grid_y,
      float(self.robot.loaded),
      box_pos[0], box_pos[1],
      drop_pos[0], drop_pos[1],
      {'up':0, 'down':1, 'left':2, 'right':3}[self.robot.direction]
    ], dtype=np.float32)


  def step(self, action):
    self.reward -= 0.1
    terminated = False
    terncated = False

    obstracles = self.shelves + self.dropoff_platforms
    moved = False
    _, _, _, box_x, box_y, drop_x, drop_y, _  = self._get_obs()
    target_x = None
    target_y = None

    # target calculation before action
    if self.robot.loaded:
      target_x = drop_x
      target_y = drop_y
    else:
      target_x = box_x
      target_y = box_y
    curr_distance = self.shortest_path_dist(self.robot.grid_x, self.robot.grid_y, target_x, target_y)

    if action < 4:
      dirs = ['up', 'down', 'left', 'right']
      moved = self.robot.handle_inputs_single(dirs[action], obstracles)
      if not moved:
        self.reward = -2


    if action == 4:
      if self.robot.loaded:
        if not self.robot.drop_box(self.dropoff_platforms):
          self.reward -= 8          # false drop penalty
        else:
          self.reward += 200        # drop reward
          terminated = True
          self.score += 1

      else:
        if not self.robot.pickup_box(self.shelves):
          self.reward -= 8          # false pickup penalty
        else:
          self.reward += 50         # pickup reward


    # target recalculation after action
    next_obs = self._get_obs()
    _, _, _, n_box_x, n_box_y, n_drop_x, n_drop_y, _  = next_obs
    next_target_x = None
    next_target_y = None
    if self.robot.loaded:
        next_target_x, next_target_y = n_drop_x, n_drop_y
    else:
        next_target_x, next_target_y = n_box_x, n_box_y


    next_distance = self.shortest_path_dist(self.robot.grid_x, self.robot.grid_y, next_target_x, next_target_y)


    # reward for moving close to the target
    self.reward += 0.99 * (-next_distance) - (-curr_distance)

    # negative reward for looping actions
    state_tuple = tuple(next_obs.astype(int).tolist())
    self.state_history.append(state_tuple)
    if len(self.state_history) == self.state_history.maxlen:
      if list(self.state_history).count(state_tuple) >= 3:
        self.reward -= 60


    # negative reward on timeout
    self.current_step += 1
    if self.current_step >= self.max_steps:
      terncated = True
      if not terminated:
        self.reward -= 300


    if self.render_mode == 'human':
      self._render_frame()


    return next_obs, terminated, terncated, {}


  def shortest_path_dist(self, start_x, start_y, goal_x, goal_y):
    if goal_x is None or goal_y is None:
      return 9999

    obstracles = set()
    for obj in self.shelves + self.dropoff_platforms:
      gx = round(obj.x - PADDING_BORDER / GRID_SPACING)
      gy = round(obj.y - PADDING_BORDER / GRID_SPACING)
      obstracles.add((gx, gy))

    queue = deque([(start_x, start_y, 0)])
    visited = set([(start_x, start_y)])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
      x, y, dist = queue.popleft()
      if x == goal_x and y == goal_y:
        return dist

      for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in visited and (nx, ny) not in obstracles:
          visited.add((nx, ny))
          queue.append((nx, ny, dist + 1))

    return 9999


  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      self.window = pygame.display.set_mode((GRID_WIDTH * GRID_SPACING, GRID_HEIGHT * GRID_SPACING))
      self.clock = pygame.time.Clock()

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

    if self.render_mode == "human":
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()
      self.clock.tick(self.metadata["render_fps"])

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()
