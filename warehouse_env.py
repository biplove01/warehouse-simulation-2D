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


class RewardManager:
    """Centralized reward shaping logic to keep the environment clean."""

    def __init__(self):
        self.step_penalty = -0.05
        self.collision_penalty = -0.5
        self.wait_penalty = -1.5 # increased for behavour shaping
        self.failed_interact_penalty = -2.5   # Much steeper than a plain wait

        self.progress_reward_scale = 2.0
        self.regress_penalty_scale = 4.0
        self.pickup_bonus = 10.0
        self.delivery_bonus = 20.0
        self.proximity_bonus = 0.5

    def calculate(self, event, distance_delta=0, proximity_bonus=0.0):
        """
        distance_delta: (previous_distance - current_distance)
        Positive delta = robot moved closer. Negative delta = robot moved away.

        Asymmetric scaling: moving away from the target is penalized
        more harshly than moving toward it is rewarded. This strongly
        discourages the robot from wandering or backtracking.
        """
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


class WarehouseEnv(gym.Env):

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.window_size = (
            GRID_WIDTH * GRID_SPACING + 2 * PADDING_BORDER,
            GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER,
        )
        self.reward_manager = RewardManager()

        # Load map objects
        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()

        # Build obstacle set from shelves and dropoff platforms
        self.obstacle_positions = {
            self._to_grid_coords(obj)
            for obj in self.shelves + self.dropoff_platforms
        }

        # Observation: [x, y, loaded, target_dx, target_dy, dropoff_dx, dropoff_dy] + 8 adjacent cells + can_pickup + can_deliver
        self.observation_size = 7 + 8 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.observation_size,), dtype=np.float32
        )

        # Actions: Up, Down, Left, Right, Interact, Wait
        self.action_space = spaces.Discrete(6)

        # Precompute BFS distance map from the central dropoff platform
        central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_grid_x, self.dropoff_grid_y = self._to_grid_coords(central_platform)
        self.dropoff_distance_map = self._bfs_distance_map(
            self.dropoff_grid_x, self.dropoff_grid_y
        )

        # Track total episodes elapsed for curriculum learning
        self.total_episodes_elapsed = 0

        # Placeholders so these attributes always exist before reset() is called
        self.target_grid_x = 0
        self.target_grid_y = 0
        self.target_distance_map = {}

    def _to_grid_coords(self, obj):
        """Convert a world object's pixel position to grid coordinates."""
        grid_x = round((obj.x - PADDING_BORDER) / GRID_SPACING)
        grid_y = round((obj.y - PADDING_BORDER) / GRID_SPACING)
        return grid_x, grid_y


    def _bfs_distance_map(self, start_grid_x, start_grid_y):
        """
        Precomputes BFS shortest distances from every reachable cell
        to the nearest walkable cell adjacent to (start_grid_x, start_grid_y).

        We seed the BFS from all walkable neighbors of the target cell,
        not the target cell itself — because the target (shelf or dropoff)
        is always an obstacle the robot cannot stand on.
        """
        distance_map = {}
        search_queue = deque()

        # Seed the BFS from all walkable neighbors of the target cell.
        # These are the cells the robot can actually stand on to interact.
        for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_x = start_grid_x + delta_x
            neighbor_y = start_grid_y + delta_y

            is_in_bounds = 0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT
            is_walkable = (neighbor_x, neighbor_y) not in self.obstacle_positions

            if is_in_bounds and is_walkable and (neighbor_x, neighbor_y) not in distance_map:
                distance_map[(neighbor_x, neighbor_y)] = 0
                search_queue.append((neighbor_x, neighbor_y, 0))

        # The rest of the BFS loop stays exactly the same
        while search_queue:
            current_x, current_y, current_dist = search_queue.popleft()

            for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_x = current_x + delta_x
                neighbor_y = current_y + delta_y

                is_in_bounds = (
                    0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT
                )
                is_not_obstacle = (neighbor_x, neighbor_y) not in self.obstacle_positions

                if is_in_bounds and is_not_obstacle and (neighbor_x, neighbor_y) not in distance_map:
                    distance_map[(neighbor_x, neighbor_y)] = current_dist + 1
                    search_queue.append((neighbor_x, neighbor_y, current_dist + 1))

        return distance_map

    # NORMAL SPAWNING METHOD

    # def _spawn_new_target(self):
    #     """Clear all shelves and place a box on a new random shelf."""
    #     for shelf in self.shelves:
    #         shelf.has_box = False
    #         shelf.image = shelf.empty_image

    #     new_target_shelf = random.choice(self.shelves)
    #     new_target_shelf.has_box = True
    #     new_target_shelf.image = new_target_shelf.loaded_image

    #     self.target_grid_x, self.target_grid_y = self._to_grid_coords(new_target_shelf)
    #     self.target_distance_map = self._bfs_distance_map(
    #         self.target_grid_x, self.target_grid_y
    #     )


    # TARGETED SPAWNING METHOD 
    def _spawn_new_target(self):
        """Clear all shelves and place a box on a shelf in the extreme left/right zones."""
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        # Restrict spawning to the extreme left (col 1–2) and right (col 19–20) shelves
        # These are the zones where the robot currently underperforms.
        extreme_zone_shelves = [
            shelf for shelf in self.shelves
            if self._to_grid_coords(shelf)[0] in {1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 29}
        ]

        # Fall back to all shelves if the filtered list is somehow empty
        spawn_pool = extreme_zone_shelves if extreme_zone_shelves else self.shelves

        new_target_shelf = random.choice(spawn_pool)
        new_target_shelf.has_box = True
        new_target_shelf.image = new_target_shelf.loaded_image

        self.target_grid_x, self.target_grid_y = self._to_grid_coords(new_target_shelf)
        self.target_distance_map = self._bfs_distance_map(
            self.target_grid_x, self.target_grid_y
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.recent_positions = deque(maxlen=10)  

        # Set dropoff grid position using the central platform
        central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_grid_x, self.dropoff_grid_y = self._to_grid_coords(central_platform)

        # Spawn the first target shelf
        self._spawn_new_target()

        # Build the set of all blocked positions (shelves + dropoff platforms)
        blocked_positions = self.obstacle_positions | {
            self._to_grid_coords(platform) for platform in self.dropoff_platforms
        }

        # Build a list of all valid spawn positions on the grid
        all_valid_spawn_positions = [
            (grid_x, grid_y)
            for grid_x in range(GRID_WIDTH)
            for grid_y in range(GRID_HEIGHT)
            if (grid_x, grid_y) not in blocked_positions
        ]

        # Curriculum learning: for early episodes, only consider positions
        # close to the target shelf so the robot can learn pickup first
        if self.total_episodes_elapsed < 300:
            nearby_spawn_positions = [
                (grid_x, grid_y)
                for grid_x, grid_y in all_valid_spawn_positions
                if abs(grid_x - self.target_grid_x) <= 3
                and abs(grid_y - self.target_grid_y) <= 3
            ]
            # Fall back to any valid position if none are nearby (edge case)
            spawn_pool = nearby_spawn_positions if nearby_spawn_positions else all_valid_spawn_positions
        else:
            spawn_pool = all_valid_spawn_positions

        robot_start_x, robot_start_y = random.choice(spawn_pool)
        self.robot = Robot(start_x=robot_start_x, start_y=robot_start_y)

        self.total_episodes_elapsed += 1

        return self._get_observation(), {}



    def _get_observation(self):
        robot = self.robot

        # Determine the current navigation target based on whether robot is loaded
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

        # 8-neighbor occupancy (1.0 = blocked, 0.0 = free)
        for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            neighbor_x = robot.grid_x + delta_x
            neighbor_y = robot.grid_y + delta_y

            is_out_of_bounds = not (0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT)
            is_obstacle = (neighbor_x, neighbor_y) in self.obstacle_positions

            observation.append(1.0 if is_obstacle or is_out_of_bounds else 0.0)


        # Is the robot currently in a valid position to pick up?
        can_pickup = float(
            not robot.loaded
            and abs(robot.grid_x - self.target_grid_x) + abs(robot.grid_y - self.target_grid_y) == 1
        )
        # Is the robot currently in a valid position to deliver?
        can_deliver = float(
            robot.loaded
            and abs(robot.grid_x - self.dropoff_grid_x) + abs(robot.grid_y - self.dropoff_grid_y) <= 2
        )
        observation.append(can_pickup)
        observation.append(can_deliver)

        return np.array(observation, dtype=np.float32)
    
    
    def step(self, action):
        self.steps += 1
        robot = self.robot

        # 1. Record distance before taking the action
        active_distance_map = (
            self.dropoff_distance_map if robot.loaded else self.target_distance_map
        )
        distance_before = active_distance_map.get((robot.grid_x, robot.grid_y), 50)

        current_event = "move"
        earned_proximity_bonus = 0.0

        # 2. Execute the action
        if action < 4:  # Movement actions: Up, Down, Left, Right
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

            # Proximity bonus: reward the robot for getting close to the target shelf
            if not robot.loaded:
                dist_to_target = self.target_distance_map.get((robot.grid_x, robot.grid_y), 50)
                if dist_to_target <= 2:
                    earned_proximity_bonus = self.reward_manager.proximity_bonus

        elif action == 4:  # Interact action
            dist_to_target_shelf = (
                abs(robot.grid_x - self.target_grid_x)
                + abs(robot.grid_y - self.target_grid_y)
            )
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

            elif robot.loaded and dist_to_dropoff <= 2:
                robot.loaded = False
                self.score += 1
                current_event = "delivery"
                self._spawn_new_target()

            else:
                current_event = "failed_interact"

        else:
            current_event = "wait"

        # 3. Calculate reward based on event, distance progress, and proximity
        distance_after = active_distance_map.get((robot.grid_x, robot.grid_y), 50)
        reward = self.reward_manager.calculate(
            current_event,
            distance_delta=(distance_before - distance_after),
            proximity_bonus=earned_proximity_bonus,
        )

        # 4. Apply revisit penalty to discourage oscillation (movement actions only)
        if action < 4:
            current_position = (robot.grid_x, robot.grid_y)
            if current_position in self.recent_positions:
                reward -= 0.3
            self.recent_positions.append(current_position)

        # 5. Episode ends only when step limit is reached
        is_done = self.steps >= 500

        return self._get_observation(), reward, is_done, False, {}


    def heuristic_action(self):
        """
        Returns the best action toward the current target using BFS distance.
        Used during exploration instead of random actions to speed up training.
        """
        robot = self.robot

        active_distance_map = (
            self.dropoff_distance_map if robot.loaded else self.target_distance_map
        )

        current_distance = active_distance_map.get((robot.grid_x, robot.grid_y), 50)

        # Check all 4 movement directions and pick the one that reduces distance most
        direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
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

        # If adjacent to the target, interact — but only when no better move exists
        if not robot.loaded:
            dist_to_target_shelf = (
                abs(robot.grid_x - self.target_grid_x)
                + abs(robot.grid_y - self.target_grid_y)
            )
            if dist_to_target_shelf == 1 and best_action is None:
                return 4  # Interact

        elif robot.loaded:
            dist_to_dropoff = (
                abs(robot.grid_x - self.dropoff_grid_x)
                + abs(robot.grid_y - self.dropoff_grid_y)
            )
            if dist_to_dropoff <= 2:
                return 4  # Interact

        # Fall back to a random movement action if no better move found (e.g. trapped)
        if best_action is None:
            return random.randint(0, 3)

        return best_action

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            window_width = GRID_WIDTH * GRID_SPACING + (2 * PADDING_BORDER)
            window_height = GRID_HEIGHT * GRID_SPACING + (2 * PADDING_BORDER)
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Warehouse Simulation")
            self.clock = pygame.time.Clock()

        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((30, 30, 30))

        # Draw charge stations
        for charge_station in self.charge_stations:
            self.screen.blit(charge_station.image, (charge_station.x, charge_station.y))

        # Draw dropoff platforms
        for dropoff_platform in self.dropoff_platforms:
            self.screen.blit(dropoff_platform.image, (dropoff_platform.x, dropoff_platform.y))

        # Draw shelves with shadow and target highlight
        for shelf in self.shelves:
            shelf_grid_position = self._to_grid_coords(shelf)
            is_current_target = (
                not self.robot.loaded
                and shelf_grid_position == (self.target_grid_x, self.target_grid_y)
            )

            # Draw shadow behind the shelf
            self.screen.blit(shelf.shadow_image, (shelf.x - 1, shelf.y + 4))

            # Highlight the target shelf in yellow
            if is_current_target:
                pygame.draw.rect(
                    self.screen, (255, 255, 0), (shelf.x - 2, shelf.y - 2, TILE_SIZE + 4, TILE_SIZE + 4), 2
                )

            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        # Draw the robot using its actual sprite
        robot_pixel_x = PADDING_BORDER + self.robot.grid_x * GRID_SPACING
        robot_pixel_y = PADDING_BORDER + self.robot.grid_y * GRID_SPACING

        # Pick the correct image based on loaded state
        # Since the RL robot doesn't track direction, we default to vertical facing
        if self.robot.loaded:
            robot_image = ROBOT_IMAGE_VERTICAL_BOX
        else:
            robot_image = ROBOT_IMAGE_VERTICAL

        # Center the robot sprite on its grid cell
        center_offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
        center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
        self.screen.blit(robot_image, (robot_pixel_x + center_offset_x, robot_pixel_y + center_offset_y))

        pygame.display.flip()
        self.clock.tick(30)