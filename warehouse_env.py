import gymnasium as gym
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
        self.clock.tick(20)