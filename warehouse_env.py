import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
from world import create_map
from constants import *
from robot import Robot

class RewardManager:
    """Centralized reward shaping logic to keep the environment clean."""
    def __init__(self):
        self.step_penalty = -0.01
        self.collision_penalty = -0.5
        self.wait_penalty = -0.1
        self.move_reward_scale = 1.0  # Multiplier for progress toward goal
        self.pickup_bonus = 10.0
        self.delivery_bonus = 20.0

    def calculate(self, event, dist_delta=0):
        """
        dist_delta: (previous_distance - current_distance)
        Positive delta means we got closer.
        """
        if event == "collision": return self.collision_penalty
        if event == "pickup": return self.pickup_bonus
        if event == "delivery": return self.delivery_bonus
        if event == "wait": return self.wait_penalty
        
        # Default: Reward for progress + step penalty
        return (dist_delta * self.move_reward_scale) + self.step_penalty

class WarehouseEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.reward_manager = RewardManager()
        
        # Observation: [x, y, loaded, target_dx, target_dy, dropoff_dx, dropoff_dy] + 8 adj cells
        self.obs_size = 7 + 8
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(6) # Up, Down, Left, Right, Interact, Wait

        # Static Map Setup
        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()
        self.obstacle_set = {self._obj_grid(s) for s in self.shelves + self.dropoff_platforms}
        
        # Dropoff location
        central = self.dropoff_platforms[len(self.dropoff_platforms)//2]
        self.drop_gx, self.drop_gy = self._obj_grid(central)
        self.dropoff_dist_map = self._bfs_dist_map(self.drop_gx, self.drop_gy)

    def _obj_grid(self, obj):
        return round((obj.x - PADDING_BORDER) / GRID_SPACING), round((obj.y - PADDING_BORDER) / GRID_SPACING)

    def _bfs_dist_map(self, gx, gy):
        """Precomputes distances from every cell to a target (gx, gy)."""
        dist_map = {(gx, gy): 0}
        queue = deque([(gx, gy, 0)])
        while queue:
            x, y, d = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in self.obstacle_set:
                    if (nx, ny) not in dist_map:
                        dist_map[(nx, ny)] = d + 1
                        queue.append((nx, ny, d+1))
        return dist_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot = Robot(start_x=1, start_y=1) # Fixed spawn for stability
        self.steps = 0
        self.score = 0
        
        # Spawn one package
        shelf = random.choice(self.shelves)
        self.target_gx, self.target_gy = self._obj_grid(shelf)
        self.target_dist_map = self._bfs_dist_map(self.target_gx, self.target_gy)
        
        return self._get_obs(), {}

    def _get_obs(self):
        r = self.robot
        # Progress Targets
        tx, ty = (self.drop_gx, self.drop_gy) if r.loaded else (self.target_gx, self.target_gy)
        
        obs = [
            r.grid_x / GRID_WIDTH, r.grid_y / GRID_HEIGHT,
            float(r.loaded),
            (tx - r.grid_x) / GRID_WIDTH, (ty - r.grid_y) / GRID_HEIGHT,
            (self.drop_gx - r.grid_x) / GRID_WIDTH, (self.drop_gy - r.grid_y) / GRID_HEIGHT
        ]
        # 8-neighbor occupancy
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = r.grid_x+dx, r.grid_y+dy
            obs.append(1.0 if (nx, ny) in self.obstacle_set or not (0<=nx<GRID_WIDTH) else 0.0)
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        r = self.robot
        
        # 1. Track distance before move
        d_map = self.dropoff_dist_map if r.loaded else self.target_dist_map
        pre_dist = d_map.get((r.grid_x, r.grid_y), 50)
        
        event = "move"
        
        # 2. Execute Action
        if action < 4: # Move
            dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
            nx, ny = r.grid_x + dx, r.grid_y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in self.obstacle_set:
                r.grid_x, r.grid_y = nx, ny
            else:
                event = "collision"
        elif action == 4: # Interact
            dist_to_target = abs(r.grid_x - self.target_gx) + abs(r.grid_y - self.target_gy)
            dist_to_drop = abs(r.grid_x - self.drop_gx) + abs(r.grid_y - self.drop_gy)
            
            if not r.loaded and dist_to_target == 1:
                r.loaded = True
                event = "pickup"
            elif r.loaded and dist_to_drop <= 1:
                r.loaded = False
                self.score += 1
                event = "delivery"
            else:
                event = "collision" # Failed interaction
        else: # Wait
            event = "wait"

        # 3. Calculate Reward
        post_dist = d_map.get((r.grid_x, r.grid_y), 50)
        reward = self.reward_manager.calculate(event, dist_delta=(pre_dist - post_dist))
        
        # 4. Check Done
        done = (event == "delivery") or (self.steps >= 1000)
        
        return self._get_obs(), reward, done, False, {}