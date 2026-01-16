import gymnasium as gym
from gymnasium import spaces
import numpy as np

import pygame
from world import create_map
from constants import *
from robot import Robot

class WarehouseEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

  def __init__(self, render_mode=None):
    super().__init__()
    self.render_mode = render_mode

    # Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4:Interact (E)
    self.action_space = spaces.Discrete(5)

    # Observation: [robot_x, robot_y, robot_loaded, box_x, box_y]
    # Normalized between 0 and 1
    self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
