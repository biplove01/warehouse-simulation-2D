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
            self.window = None