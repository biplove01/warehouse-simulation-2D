"""
env_multi.py — Multi-agent warehouse environment (PettingZoo ParallelEnv)
Supports NUMBER_OF_ROBOTS agents (default 5 from constants.py).

Key design decisions vs single-agent env:
  - Grid-based pickup/drop (no pixel/direction checks) for reliable headless training
  - BFS distance maps precomputed at reset → O(1) reward shaping per step
  - Priority-ordered collision resolution (lower robot_id wins contested cells)
  - Greedy nearest-package assignment (keeps RL focused on navigation, not planning)
  - Shared occupancy map rebuilt once per step, shared across all obs calls
"""

from pettingzoo import ParallelEnv
from gymnasium import spaces
import functools
import numpy as np
from collections import deque
import pygame
from world import create_map
from constants import *
from robot import Robot
import random

NUM_AGENTS = NUMBER_OF_ROBOTS          # 5 — from your constants.py
# Obs layout: [sx,sy,loaded, tdx,tdy, ddx,ddy] + (N-1)*[rdx,rdy,r_loaded] + 8 adj cells
OBS_SIZE   = 7 + (NUM_AGENTS - 1) * 3 + 8   # 27 for 5 robots
N_ACTIONS  = 6                          # up/down/left/right/interact/wait


class WarehouseMultiEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "warehouse_multi_v1", "render_fps": 30}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = [f"robot_{i}" for i in range(NUM_AGENTS)]
        self.window = None
        self.clock = None
        self._font = None

    # ── spaces (cached — PettingZoo requirement) ──────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(N_ACTIONS)

    # ── private helpers ───────────────────────────────────────────────────────

    def _obj_grid(self, obj):
        """Convert a sprite's pixel position to grid coordinates."""
        gx = round((obj.x - PADDING_BORDER) / GRID_SPACING)
        gy = round((obj.y - PADDING_BORDER) / GRID_SPACING)
        return gx, gy

    def _get_shelf_at(self, gx, gy):
        for s in self.shelves:
            if self._obj_grid(s) == (gx, gy):
                return s
        return None

    def _build_obstacle_set(self):
        """Shelves + dropoff platforms — static for the whole episode."""
        return {self._obj_grid(obj) for obj in self.shelves + self.dropoff_platforms}

    def _bfs_dist_map(self, goal_x, goal_y):
        """
        Reverse BFS from goal_x,goal_y outward.
        dist_map[(x,y)] = shortest navigable distance from (x,y) to goal.
        Stops as adjacent (distance 1) since robots interact from adjacent cell.
        """
        dist_map = {}
        queue = deque()

        # goal itself is distance 0 (robot standing on goal — only for dropoff)
        dist_map[(goal_x, goal_y)] = 0

        # Seed adjacents at distance 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = goal_x+dx, goal_y+dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in self._obstacle_set
                    and (nx, ny) not in dist_map):
                dist_map[(nx, ny)] = 1
                queue.append((nx, ny, 1))

        while queue:
            x, y, d = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                        and (nx, ny) not in self._obstacle_set
                        and (nx, ny) not in dist_map):
                    dist_map[(nx, ny)] = d + 1
                    queue.append((nx, ny, d+1))

        return dist_map

    def _get_dist(self, dist_map, x, y):
        return dist_map.get((x, y), 9999)

    def _build_occupancy(self):
        """
        Cell-type map rebuilt once per step, shared across all robot obs.
          0.33 = shelf (static obstacle)
          0.66 = robot
          1.00 = charging station
        """
        occ = {}
        for s in self.shelves:
            occ[self._obj_grid(s)] = 0.33
        for c in self.charge_stations:
            occ[self._obj_grid(c)] = 1.00
        for r in self.robots:
            occ[(r.grid_x, r.grid_y)] = 0.66
        return occ

    # ── package assignment ────────────────────────────────────────────────────

    def _assign_packages(self):
        """
        Greedy nearest-available assignment.
        Called at reset and after every delivery or robot death.
        Robots with an existing assignment are skipped.
        """
        claimed = {v for v in self.assignments.values() if v is not None}
        available = [p for p in self.target_queue if p not in claimed]

        for i, robot in enumerate(self.robots):
            if self.assignments[i] is not None:
                continue
            if not available:
                break
            best = min(
                available,
                key=lambda p: self._get_dist(
                    self._pkg_dist_maps.get(p, {}), robot.grid_x, robot.grid_y)
            )
            self.assignments[i] = best
            available.remove(best)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.score        = 0
        self.current_step = 0
        self.max_steps    = 15000
        self.goal_deliveries = 10 if self.render_mode is None else 100

        # Build world
        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()
        for shelf in self.shelves:
            if hasattr(shelf, "has_box"):
                shelf.has_box = False

        self.target_queue = []  # list so we can .remove() by value

        # Dropoff anchor (central platform, index 2)
        central = self.dropoff_platforms[2]
        self.dropoff_gx, self.dropoff_gy = self._obj_grid(central)

        # Precompute static obstacle set + dropoff distance map
        self._obstacle_set      = self._build_obstacle_set()
        self._dropoff_dist_map  = self._bfs_dist_map(self.dropoff_gx, self.dropoff_gy)

        # Spawn 10 packages; precompute each package's dist map
        shelf_positions = [(s, *self._obj_grid(s)) for s in self.shelves]
        random.shuffle(shelf_positions)
        self._pkg_dist_maps = {}
        for shelf, gx, gy in shelf_positions[:10]:
            shelf.has_box = True
            self.target_queue.append((gx, gy))
            self._pkg_dist_maps[(gx, gy)] = self._bfs_dist_map(gx, gy)

        # Spawn robots on distinct free cells
        blocked = {self._obj_grid(o)
                   for o in self.shelves + self.dropoff_platforms + self.charge_stations}
        self.robots = []
        spawned = set()
        for _ in range(NUM_AGENTS):
            while True:
                rx = int(np.random.randint(0, GRID_WIDTH))
                ry = int(np.random.randint(0, GRID_HEIGHT))
                if (rx, ry) not in blocked and (rx, ry) not in spawned:
                    break
            spawned.add((rx, ry))
            self.robots.append(Robot(start_x=rx, start_y=ry))

        # Per-robot bookkeeping
        self.assignments     = {i: None for i in range(NUM_AGENTS)}
        self.state_histories = [deque(maxlen=20) for _ in range(NUM_AGENTS)]
        self.stuck_counters  = [0] * NUM_AGENTS
        self._last_positions = [(r.grid_x, r.grid_y) for r in self.robots]

        self._assign_packages()

        self._step_occ = self._build_occupancy()  # ← add this line

        observations = {f"robot_{i}": self._get_obs(i) for i in range(NUM_AGENTS)}
        infos        = {f"robot_{i}": {} for i in range(NUM_AGENTS)}

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self, robot_id):
        robot  = self.robots[robot_id]
        W, H   = GRID_WIDTH, GRID_HEIGHT
        occ    = self._step_occ   # set once per step in step(); also set after reset

        sx     = robot.grid_x / W
        sy     = robot.grid_y / H
        loaded = float(robot.loaded)

        # Current navigation target
        if robot.loaded:
            tx, ty = self.dropoff_gx, self.dropoff_gy
        elif self.assignments[robot_id] is not None:
            tx, ty = self.assignments[robot_id]
        else:
            tx, ty = robot.grid_x, robot.grid_y   # no target: delta = 0

        tdx = np.clip((tx - robot.grid_x) / W, -1.0, 1.0)
        tdy = np.clip((ty - robot.grid_y) / H, -1.0, 1.0)
        ddx = np.clip((self.dropoff_gx - robot.grid_x) / W, -1.0, 1.0)
        ddy = np.clip((self.dropoff_gy - robot.grid_y) / H, -1.0, 1.0)

        # Teammate relative positions + loaded flag
        tm_feats = []
        for j in range(NUM_AGENTS):
            if j == robot_id:
                continue
            tm = self.robots[j]
            tm_feats += [
                np.clip((tm.grid_x - robot.grid_x) / W, -1.0, 1.0),
                np.clip((tm.grid_y - robot.grid_y) / H, -1.0, 1.0),
                float(tm.loaded),
            ]

        # 8-directional adjacency — what is in each neighbouring cell?
        adj = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = robot.grid_x+dx, robot.grid_y+dy
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                adj.append(1.0)   # out-of-bounds treated as wall
            else:
                adj.append(occ.get((nx, ny), 0.0))

        return np.array([sx, sy, loaded, tdx, tdy, ddx, ddy] + tm_feats + adj,
                        dtype=np.float32)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        """
        actions: dict {"robot_0": int, ..., "robot_4": int}

        Movement resolution order (prevents two robots sharing a cell):
          1. Lower robot_id gets priority.
          2. If a cell is already claimed, the later robot stays put and gets a bump penalty.
          3. Interact and Wait actions are resolved after all moves.
        """
        # Rebuild occupancy once — shared by _get_obs calls this step
        self._step_occ = self._build_occupancy()

        rewards      = {f"robot_{i}": -0.01  for i in range(NUM_AGENTS)}
        terminations = {a: False for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos        = {a: {}    for a in self.agents}

        dir_map = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}   # up/down/left/right

        # ── Pre-step distances (for shaping) ──
        pre_dists = {}
        for i, robot in enumerate(self.robots):
            if robot.loaded:
                dm = self._dropoff_dist_map
            elif self.assignments[i] is not None:
                dm = self._pkg_dist_maps.get(self.assignments[i], {})
            else:
                dm = {}
            pre_dists[i] = self._get_dist(dm, robot.grid_x, robot.grid_y)

        # ── Movement: priority collision resolution ──
        claimed   = {}    # (gx,gy) → robot_id
        intended  = {}    # robot_id → (gx,gy)

        for i in range(NUM_AGENTS):                # lower id = higher priority
            robot  = self.robots[i]
            action = actions[f"robot_{i}"]

            if action < 4:
                dx, dy = dir_map[action]
                nx, ny = robot.grid_x+dx, robot.grid_y+dy
                in_bounds    = (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT)
                free_obstacle = (nx, ny) not in self._obstacle_set
                free_claimed  = (nx, ny) not in claimed
                rewards[f"robot_{i}"] -= 0.05 # Small "energy cost" for any move

                if in_bounds and free_obstacle and free_claimed:
                    intended[i]      = (nx, ny)
                    claimed[(nx, ny)] = i
                    robot.direction   = ["up","down","left","right"][action]
                else:
                    intended[i] = (robot.grid_x, robot.grid_y)
                    claimed[(robot.grid_x, robot.grid_y)] = i
                    rewards[f"robot_{i}"] -= 1.5  # was 0.5   # bumped into obstacle/robot
            else:
                intended[i] = (robot.grid_x, robot.grid_y)
                claimed.setdefault((robot.grid_x, robot.grid_y), i)

        # Apply resolved positions
        for i, robot in enumerate(self.robots):
            robot.grid_x, robot.grid_y = intended[i]
            robot.update_image()

        # Update occupancy with post-move positions
        self._step_occ = self._build_occupancy()

        # ── Interact (4) and Wait (5) ──
        for i, robot in enumerate(self.robots):
            action = actions[f"robot_{i}"]
            # Interact
            if action == 4:
                reward_bonus, pickup, dropoff = self._handle_interact(i, robot, claimed)
                rewards[f"robot_{i}"] += reward_bonus
                # return pickup/dropoff in infos so trainer can read it
                infos[f"robot_{i}"]["pickup"] = pickup
                infos[f"robot_{i}"]["dropoff"] = dropoff
            
            #  Wait
            elif action == 5: 
                # Only penalize waiting if THIS robot actually has a job to do
                if self.assignments[i] is not None:
                    rewards[f"robot_{i}"] -= 1.0
                else:
                    # Small bonus for staying still when unassigned to prevent jitter
                    rewards[f"robot_{i}"] += 0.25


        # ── Post-step reward shaping (progress toward target) ──
        for i, robot in enumerate(self.robots):
            if robot.loaded:
                dm = self._dropoff_dist_map
            elif self.assignments[i] is not None:
                dm = self._pkg_dist_maps.get(self.assignments[i], {})
            else:
                dm = {}
            post_dist = self._get_dist(dm, robot.grid_x, robot.grid_y)

            # Reward for moving Closer (3x stronger signal):
            dist_diff = pre_dists[i] - post_dist
            rewards[f"robot_{i}"] += 2.0 * dist_diff


        # ── Arrival Bonus: Reward reaching the destination doorstep ──
        for i, robot in enumerate(self.robots):
            if robot.loaded:
                dm = self._dropoff_dist_map
            elif self.assignments[i] is not None:
                dm = self._pkg_dist_maps.get(self.assignments[i], {})
            else:
                continue

            post_dist = self._get_dist(dm, robot.grid_x, robot.grid_y)
            
            # Check if this move brought us to the doorstep (dist 1) from further away
            if post_dist == 1 and pre_dists[i] > 1:
                rewards[f"robot_{i}"] += 5.0  # Arrival bonus
            
            # Keep a small "being adjacent" nudge if they haven't interacted yet
            elif post_dist == 1:
                rewards[f"robot_{i}"] += 0.1


        # ── Deadlock / loop detection ──
        for i, robot in enumerate(self.robots):
            key = (robot.grid_x, robot.grid_y, int(robot.loaded), self.assignments[i])
            self.state_histories[i].append(key)
            visit_count = self.state_histories[i].count(key)
            if visit_count >= 3:
                # Escalating penalty — the more times you revisit, the worse it gets
                rewards[f"robot_{i}"] -= 1 * (visit_count - 2)

            # Stuck counter → forced random nudge
            if (robot.grid_x, robot.grid_y) == self._last_positions[i]:
                self.stuck_counters[i] += 1
            else:
                self.stuck_counters[i] = 0

            if self.stuck_counters[i] > 12:
                self._force_random_move(i, robot, claimed)
                self.stuck_counters[i] = 0

        # ── Oscillation detection (position only, ignores state) ──
        for i, robot in enumerate(self.robots):
            recent_positions = [
                (s[0], s[1]) for s in self.state_histories[i]
            ]
            if len(recent_positions) >= 6:
                # Check if robot is alternating between just 2 positions
                unique_recent = set(recent_positions[-6:])
                if len(unique_recent) <= 2:
                    rewards[f"robot_{i}"] -= 3  #  oscillation penalty

        self._last_positions = [(r.grid_x, r.grid_y) for r in self.robots]

        # ── Termination / truncation ──
        self.current_step += 1
        episode_done = self.score >= self.goal_deliveries

        if episode_done:
            for a in self.agents:
                terminations[a] = True

        if self.current_step >= self.max_steps:
            for a in self.agents:
                truncations[a] = True
            for i in range(NUM_AGENTS):
                rewards[f"robot_{i}"] -= 20

        # Build final observations
        observations = {f"robot_{i}": self._get_obs(i) for i in range(NUM_AGENTS)}

        # PettingZoo: prune finished agents
        self.agents = [a for a in self.agents
                       if not terminations[a] and not truncations[a]]

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos

    # ── interact helper ───────────────────────────────────────────────────────

    def _handle_interact(self, robot_id, robot, claimed):
        bonus = 0
        pickup = 0
        dropoff = 0

        if robot.loaded:
            # Robot is carrying a box — try to drop it off
            for plat in self.dropoff_platforms:
                pgx, pgy = self._obj_grid(plat)
                if abs(robot.grid_x - pgx) + abs(robot.grid_y - pgy) <= 1:
                    robot.loaded = False
                    robot.update_image()
                    self.score += 1

                    pkg = self.assignments[robot_id]
                    if pkg in self.target_queue:
                        self.target_queue.remove(pkg)
                        if pkg in self._pkg_dist_maps:
                            del self._pkg_dist_maps[pkg]
                    self.assignments[robot_id] = None
                    self._assign_packages()

                    # ✅ Full delivery completed — robot picked up AND delivered a box
                    # This is the ultimate goal, highest reward in the system
                    bonus += 30
                    dropoff = 1  # ← flag
                    return bonus, pickup, dropoff

            # ❌ Robot tried to drop but wasn't adjacent to any platform
            # Penalize to discourage randomly spamming interact while loaded
            bonus -= 5

        else:
            # Robot is unloaded — try to pick up assigned package
            pkg = self.assignments[robot_id]
            if pkg is not None:
                tx, ty = pkg
                shelf = self._get_shelf_at(tx, ty)
                if (shelf and shelf.has_box
                        and abs(robot.grid_x - tx) + abs(robot.grid_y - ty) == 1):
                    robot.loaded = True
                    shelf.has_box = False
                    shelf.image = shelf.empty_image
                    robot.update_image()

                    # ✅ Successful pickup — robot reached its assigned shelf and correctly executed interact while adjacent
                    bonus += 20
                    pickup = 1
                else:
                    # ❌ Robot tried to interact but either:
                    #    - is not adjacent to shelf (dist > 1)
                    #    - shelf has no box (already picked up by someone else)
                    bonus -= 5
            else:
                # ❌ Robot has no assignment at all — tried to pick up
                # but there's nothing assigned to it yet
                # Small penalty — not the robot's fault, just discourage
                # wasting interact actions when unassigned
                bonus -= 2

        return bonus, pickup, dropoff

    # ── deadlock helper ───────────────────────────────────────────────────────

    def _force_random_move(self, robot_id, robot, claimed):
        """Apply a random valid move to a stuck robot."""
        dirs = [(0,-1),(0,1),(-1,0),(1,0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = robot.grid_x+dx, robot.grid_y+dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in self._obstacle_set
                    and (nx, ny) not in claimed):
                robot.grid_x, robot.grid_y = nx, ny
                claimed[(nx, ny)] = robot_id
                return

    # ── render ────────────────────────────────────────────────────────────────

    def _render_frame(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (GRID_WIDTH * GRID_SPACING, GRID_HEIGHT * GRID_SPACING))
            pygame.display.set_caption(f"Warehouse — {NUM_AGENTS} robots")
            self.clock  = pygame.time.Clock()
            self._font  = pygame.font.SysFont("monospace", 13)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                clicked_shelf = None
                for shelf in self.shelves:
                    sgx, sgy = self._obj_grid(shelf)
                    # Recompute pixel bounds directly from grid position
                    px = PADDING_BORDER + sgx * GRID_SPACING
                    py = PADDING_BORDER + sgy * GRID_SPACING
                    # Use full GRID_SPACING so there's no gap between tiles
                    if px <= mx < px + GRID_SPACING and py <= my < py + GRID_SPACING:
                        clicked_shelf = shelf
                        break

                if clicked_shelf and not clicked_shelf.has_box:
                    clicked_shelf.has_box = True
                    clicked_shelf.image   = clicked_shelf.loaded_image
                    pkg = self._obj_grid(clicked_shelf)
                    self.target_queue.append(pkg)
                    self._pkg_dist_maps[pkg] = self._bfs_dist_map(*pkg)
                    self._assign_packages()
                    

        canvas = pygame.Surface((GRID_WIDTH * GRID_SPACING, GRID_HEIGHT * GRID_SPACING))
        canvas.fill("#5FCB9B")

        for obj in self.charge_stations + self.dropoff_platforms:
            canvas.blit(obj.image, obj)

        for obj in self.shelves:
            canvas.blit(obj.shadow_image, (obj.x - 3, obj.y + 12))

        # Draw all robots
        for robot in self.robots:
            r_rect = robot.get_pixel_rect()
            canvas.blit(robot.image, (r_rect.x, r_rect.y))

        for obj in self.shelves:
            canvas.blit(obj.image, obj)
        
         # DEBUG: draw shelf bounding boxes
        for shelf in self.shelves:
            sgx, sgy = self._obj_grid(shelf)
            px = PADDING_BORDER + sgx * GRID_SPACING
            py = PADDING_BORDER + sgy * GRID_SPACING
            pygame.draw.rect(canvas, (255, 0, 0),
                            pygame.Rect(px, py, GRID_SPACING, GRID_SPACING), 1)

        # HUD
        hud_text = (f"Score: {self.score}/{self.goal_deliveries}  "
                    f"Step: {self.current_step}  "
                    f"Queue: {len(self.target_queue)}")
        hud = self._font.render(hud_text, True, (0, 0, 0))
        canvas.blit(hud, (6, 6))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

       

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None