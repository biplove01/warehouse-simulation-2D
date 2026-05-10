import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

from warehouse_env import WarehouseEnv, ROBOT_HOME_GRID_X, ROBOT_HOME_GRID_Y
from model import QNetwork
from train import TrainingEnv, HOME_WAIT_STEPS_REQUIRED
from constants import *


# ─── REWARD CONSTANTS ─────────────────────────────────────────────────────────

# Agent 2 was forced to yield (either moved toward Agent 1's destination, or
# failed to clear Agent 1's path when Agent 1 walked toward Agent 2's cell).
AGENT_COLLISION_PENALTY = -20.0

# Agent 2 is one cell away from Agent 1 after the step (Manhattan distance 1).
PROXIMITY_DISTANCE_1_PENALTY = -2.0

# Agent 2 is two cells away from Agent 1 after the step (Manhattan distance 2).
PROXIMITY_DISTANCE_2_PENALTY = -0.5

# Bonus for voluntarily waiting/yielding while Agent 1 is nearby AND Agent 2
# is still making forward progress on its own delivery goal.
YIELDING_BONUS = 1.0

# Manhattan distance at which Agent 1 is considered "very close" for the
# yielding bonus.
YIELDING_BONUS_PROXIMITY_THRESHOLD = 2

# Agent 2's home / charging station.
# Agent 1 uses charge station (2, 0)  ← defined as ROBOT_HOME_GRID_X/Y in warehouse_env.py
# Agent 2 uses charge station (4, 0)  ← last station in the row, guaranteed distinct.
AGENT2_HOME_GRID_X = 4
AGENT2_HOME_GRID_Y = 0

# Agent 2's drop-off platform.
# Agent 1 uses dropoff_platforms[2] → grid (2, 14)  ← central_platform logic in WarehouseEnv.reset()
# Agent 2 uses dropoff_platforms[0] → grid (0, 14)  ← first platform, guaranteed distinct.
AGENT2_DROPOFF_GRID_X = 0
AGENT2_DROPOFF_GRID_Y = 14


# ─── SHARED TARGET QUEUE ──────────────────────────────────────────────────────

class SharedTargetQueue:
    """
    Central authority for all shelf targets.

    Rules
    -----
    * A target (grid_x, grid_y) is popped from the pending pool the moment
      any robot claims it — so a second robot can never receive the same shelf.
    * When a robot completes delivery it calls release_assignment(), which
      immediately queues a fresh target for whoever becomes free next.
    * If the pending pool is empty when a robot asks, a new target is
      generated on the spot (skipping any already-assigned shelves).
    """

    def __init__(self, shelves, bfs_fn):
        self.shelves = shelves
        self._bfs_distance_map = bfs_fn          # callable: (gx, gy) → dist dict

        # robot_id (1 or 2) → (grid_x, grid_y) of the shelf it is heading to
        self._assigned_targets: dict[int, tuple[int, int]] = {}

        # Targets waiting to be claimed (FIFO)
        self._pending_targets: deque[tuple[int, int]] = deque()

    def reset(self):
        """Clear all state and seed the queue with one fresh target."""
        self._assigned_targets.clear()
        self._pending_targets.clear()
        self._enqueue_new_target()

    def request_target(self, robot_id: int):
        """
        A robot asks for its next target.
        Returns (target_position, distance_map) or None if the queue is
        temporarily exhausted (extremely rare).
        """
        if not self._pending_targets:
            self._enqueue_new_target()

        if not self._pending_targets:
            return None

        target_position = self._pending_targets.popleft()
        self._assigned_targets[robot_id] = target_position
        distance_map = self._bfs_distance_map(target_position[0], target_position[1])
        return target_position, distance_map

    def release_assignment(self, robot_id: int):
        """
        Robot signals delivery complete. Frees its slot and immediately
        queues a fresh target so the other robot can claim it if free.
        """
        self._assigned_targets.pop(robot_id, None)
        self._enqueue_new_target()

    def _currently_claimed_positions(self) -> set:
        claimed = set(self._assigned_targets.values())
        claimed.update(self._pending_targets)
        return claimed

    def _enqueue_new_target(self):
        already_claimed = self._currently_claimed_positions()
        available_shelves = [
            shelf for shelf in self.shelves
            if self._shelf_grid_pos(shelf) not in already_claimed
        ]
        if not available_shelves:
            return   # all shelves already claimed — nothing to add right now

        chosen_shelf = random.choice(available_shelves)
        chosen_shelf.has_box = True
        chosen_shelf.image = chosen_shelf.loaded_image
        self._pending_targets.append(self._shelf_grid_pos(chosen_shelf))

    @staticmethod
    def _shelf_grid_pos(shelf) -> tuple:
        grid_x = round((shelf.x - PADDING_BORDER) / GRID_SPACING)
        grid_y = round((shelf.y - PADDING_BORDER) / GRID_SPACING)
        return grid_x, grid_y


# ─── AGENT 1 TRAINING ENVIRONMENT (queue-aware, otherwise unchanged) ──────────

class Agent1TrainingEnv(TrainingEnv):
    """
    Thin wrapper around TrainingEnv that routes target acquisition through the
    SharedTargetQueue so Agent 1 and Agent 2 never compete for the same shelf.

    Agent 1 is fully pre-trained and frozen.  Only target selection is changed
    here — all movement, reward, and policy logic is identical to TrainingEnv.
    Agent 1 has ZERO knowledge of Agent 2's existence or position.
    """

    def __init__(self, shared_target_queue: SharedTargetQueue, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.shared_target_queue = shared_target_queue
        self.robot_id = 1

    def _spawn_new_target(self):
        result = self.shared_target_queue.request_target(self.robot_id)
        if result is None:
            return
        target_position, distance_map = result
        self.target_grid_x, self.target_grid_y = target_position
        self.target_distance_map = distance_map
        self.returning_home = False

    def _on_delivery(self):
        """Release assignment in the queue then head back to Agent 1's home."""
        self.shared_target_queue.release_assignment(self.robot_id)

        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        self.returning_home = True
        self.consecutive_wait_steps_at_home = 0
        self.target_grid_x = ROBOT_HOME_GRID_X
        self.target_grid_y = ROBOT_HOME_GRID_Y
        self.target_distance_map = self.home_distance_map


# ─── AGENT 2 TRAINING ENVIRONMENT ────────────────────────────────────────────

class Agent2TrainingEnv(TrainingEnv):
    """
    Extends TrainingEnv for Agent 2.

    Key differences from Agent 1's TrainingEnv
    -------------------------------------------
    * Observation extended by 4 features describing Agent 1's current state.
    * Home / charging station is AGENT2_HOME_GRID_X / AGENT2_HOME_GRID_Y —
      a distinct cell from Agent 1's station.
    * Target acquisition goes through the SharedTargetQueue.

    Extra observation features appended (in order):
        agent1_relative_x     : (agent1.grid_x - agent2.grid_x) / GRID_WIDTH
        agent1_relative_y     : (agent1.grid_y - agent2.grid_y) / GRID_HEIGHT
        agent1_loaded         : float(agent1 is carrying a box)
        agent1_returning_home : float(agent1 is in home-return phase)
    """

    AGENT1_EXTRA_FEATURES = 6
    EXTENDED_OBSERVATION_SIZE = 19 + AGENT1_EXTRA_FEATURES   # = 25

    def __init__(self, shared_target_queue: SharedTargetQueue, render_mode=None):
        super().__init__(render_mode=render_mode)

        self.shared_target_queue = shared_target_queue
        self.robot_id = 2

        import gymnasium as gym
        from gymnasium import spaces
        self.observation_size = self.EXTENDED_OBSERVATION_SIZE
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.EXTENDED_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        # Override the dropoff target so Agent 2 delivers to platform (0, 14)
        # instead of Agent 1's platform (2, 14).
        # This fixes the base class observation (dropoff_dx, dropoff_dy,
        # can_deliver) and the delivery adjacency check in WarehouseEnv.step().
        self.dropoff_grid_x = AGENT2_DROPOFF_GRID_X
        self.dropoff_grid_y = AGENT2_DROPOFF_GRID_Y
        self.dropoff_distance_map = self._bfs_distance_map(
            AGENT2_DROPOFF_GRID_X, AGENT2_DROPOFF_GRID_Y
        )

        # Agent 1 state — injected by TwoAgentWarehouseEnv before every step.
        self.agent1_grid_x = ROBOT_HOME_GRID_X
        self.agent1_grid_y = ROBOT_HOME_GRID_Y
        self.agent1_loaded = False
        self.agent1_returning_home = True

    # ── observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        """Base 19-feature observation extended with 4 Agent 1 features."""
        base_observation = super()._get_observation()

        agent2_robot = self.robot

        # Agent 1 relative position
        agent1_relative_x = (self.agent1_grid_x - agent2_robot.grid_x) / GRID_WIDTH
        agent1_relative_y = (self.agent1_grid_y - agent2_robot.grid_y) / GRID_HEIGHT

        # Agent 1's current navigation target (shelf, dropoff, or home)
        agent1_target_relative_x = (self.agent1_target_x - agent2_robot.grid_x) / GRID_WIDTH
        agent1_target_relative_y = (self.agent1_target_y - agent2_robot.grid_y) / GRID_HEIGHT

        extra_features = np.array([
            agent1_relative_x,
            agent1_relative_y,
            float(self.agent1_loaded),
            float(self.agent1_returning_home),
            agent1_target_relative_x,
            agent1_target_relative_y,
        ], dtype=np.float32)

        return np.concatenate([base_observation, extra_features])

    # ── Agent 1 state injection ───────────────────────────────────────────────

    def update_agent1_state(self, agent1_grid_x, agent1_grid_y, agent1_loaded,
                            agent1_returning_home, agent1_target_x, agent1_target_y):
        self.agent1_grid_x = agent1_grid_x
        self.agent1_grid_y = agent1_grid_y
        self.agent1_loaded = agent1_loaded
        self.agent1_returning_home = agent1_returning_home
        self.agent1_target_x = agent1_target_x
        self.agent1_target_y = agent1_target_y


    # ── reset override ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Calls the parent reset, then restores Agent 2's dropoff coordinates.
        WarehouseEnv.reset() reassigns dropoff_grid_x/y from central_platform
        every episode — we override that here so Agent 2 always delivers to its
        own distinct platform (AGENT2_DROPOFF_GRID_X, AGENT2_DROPOFF_GRID_Y).
        """
        observation, info = super().reset(seed=seed, options=options)
        self.dropoff_grid_x = AGENT2_DROPOFF_GRID_X
        self.dropoff_grid_y = AGENT2_DROPOFF_GRID_Y
        self.dropoff_distance_map = self._bfs_distance_map(
            AGENT2_DROPOFF_GRID_X, AGENT2_DROPOFF_GRID_Y
        )
        return observation, info

    # ── target acquisition via shared queue ───────────────────────────────────

    def _spawn_new_target(self):
        """Routes through SharedTargetQueue — no two robots chase the same shelf."""
        result = self.shared_target_queue.request_target(self.robot_id)
        if result is None:
            return
        target_position, distance_map = result
        self.target_grid_x, self.target_grid_y = target_position
        self.target_distance_map = distance_map
        self.returning_home = False

    def _on_delivery(self):
        """Release assignment then send Agent 2 to its own home station."""
        self.shared_target_queue.release_assignment(self.robot_id)

        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image = shelf.empty_image

        self.returning_home = True
        self.consecutive_wait_steps_at_home = 0
        self.target_grid_x = AGENT2_HOME_GRID_X
        self.target_grid_y = AGENT2_HOME_GRID_Y
        self.target_distance_map = self._bfs_distance_map(
            AGENT2_HOME_GRID_X, AGENT2_HOME_GRID_Y
        )

    # ── heuristic ─────────────────────────────────────────────────────────────

    def heuristic_action(self):
        """BFS-guided heuristic that navigates toward Agent 2's own home."""
        if self.returning_home:
            robot = self.robot
            robot_at_home = (
                robot.grid_x == AGENT2_HOME_GRID_X
                and robot.grid_y == AGENT2_HOME_GRID_Y
            )
            if robot_at_home:
                return 5   # wait

            agent2_home_distance_map = self._bfs_distance_map(
                AGENT2_HOME_GRID_X, AGENT2_HOME_GRID_Y
            )
            direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            best_action = None
            best_distance = agent2_home_distance_map.get(
                (robot.grid_x, robot.grid_y), 50
            )

            for action_index, (delta_x, delta_y) in enumerate(direction_deltas):
                next_x = robot.grid_x + delta_x
                next_y = robot.grid_y + delta_y
                if (
                    0 <= next_x < GRID_WIDTH
                    and 0 <= next_y < GRID_HEIGHT
                    and (next_x, next_y) not in self.obstacle_positions
                ):
                    neighbor_distance = agent2_home_distance_map.get(
                        (next_x, next_y), 50
                    )
                    if neighbor_distance < best_distance:
                        best_distance = neighbor_distance
                        best_action = action_index

            return best_action if best_action is not None else random.randint(0, 3)

        return WarehouseEnv.heuristic_action(self)

    # ── step override for Agent 2's distinct home station ────────────────────

    def step(self, action):
        """
        Mirrors TrainingEnv.step exactly, but uses AGENT2_HOME_GRID_X/Y
        instead of the global ROBOT_HOME constants for the home-return phase.
        """
        robot = self.robot

        if self.returning_home:
            self.steps += 1
            robot_at_home = (
                robot.grid_x == AGENT2_HOME_GRID_X
                and robot.grid_y == AGENT2_HOME_GRID_Y
            )

            if robot_at_home:
                if action == 5:
                    self.consecutive_wait_steps_at_home += 1
                    reward = 1.0
                    if self.consecutive_wait_steps_at_home >= HOME_WAIT_STEPS_REQUIRED:
                        self.consecutive_wait_steps_at_home = 0
                        self._spawn_new_target()
                else:
                    self.consecutive_wait_steps_at_home = 0
                    reward = -1.0
            else:
                self.consecutive_wait_steps_at_home = 0
                agent2_home_distance_map = self._bfs_distance_map(
                    AGENT2_HOME_GRID_X, AGENT2_HOME_GRID_Y
                )
                distance_before = agent2_home_distance_map.get(
                    (robot.grid_x, robot.grid_y), 50
                )

                if action < 4:
                    direction_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                    delta_x, delta_y = direction_deltas[action]
                    next_x = robot.grid_x + delta_x
                    next_y = robot.grid_y + delta_y

                    is_in_bounds = (
                        0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
                    )
                    is_passable = (next_x, next_y) not in self.obstacle_positions

                    if is_in_bounds and is_passable:
                        robot.grid_x, robot.grid_y = next_x, next_y
                        distance_after = agent2_home_distance_map.get(
                            (robot.grid_x, robot.grid_y), 50
                        )
                        distance_delta = distance_before - distance_after
                        if distance_delta >= 0:
                            reward = (
                                distance_delta
                                * self.reward_manager.progress_reward_scale
                            )
                        else:
                            reward = (
                                distance_delta
                                * self.reward_manager.regress_penalty_scale
                            )
                        reward += self.reward_manager.step_penalty
                    else:
                        reward = self.reward_manager.collision_penalty
                else:
                    reward = self.reward_manager.step_penalty

            is_done = self.steps >= 500
            self.last_action = action
            return self._get_observation(), reward, is_done, False, {}

        return super().step(action)


# ─── TWO-AGENT WAREHOUSE ENVIRONMENT ─────────────────────────────────────────

class TwoAgentWarehouseEnv:
    """
    Wraps two independent training environments and steps them simultaneously.

    Agent 1 — frozen, pre-trained, acts as a moving obstacle.
               Has ZERO knowledge of Agent 2. Always moves freely.
               We only observe its intended action so Agent 2 can react.

    Agent 2 — actively trained. Must learn to navigate around Agent 1.

    Collision resolution — Agent 1 always has right of way
    -------------------------------------------------------
    Each step proceeds in this exact order:

      1. Observe Agent 1's frozen policy → agent1_intended_next_cell.
      2. Agent 2 selects its action     → agent2_intended_next_cell.
      3. Two conditions force Agent 2 to yield (action overridden to WAIT):

           Condition A — Agent 2 moves into the same cell Agent 1 is moving into:
               agent2_intended_next_cell == agent1_intended_next_cell

           Condition B — Agent 1 is moving into Agent 2's CURRENT cell
               (Agent 1 is walking straight toward Agent 2's tile):
               agent1_intended_next_cell == agent2_current_cell

         In both cases Agent 2 is forced to WAIT and receives
         AGENT_COLLISION_PENALTY. Agent 1 is NEVER altered.

      4. Agent 1 steps freely.
      5. Agent 2 steps with the effective (possibly overridden) action.
      6. Proximity / yielding reward shaping applied on top.

    Shared target queue
    -------------------
    A SharedTargetQueue ensures no two robots are ever assigned the same shelf.

    Separate home stations
    ----------------------
    Agent 1 docks at (ROBOT_HOME_GRID_X,  ROBOT_HOME_GRID_Y).
    Agent 2 docks at (AGENT2_HOME_GRID_X, AGENT2_HOME_GRID_Y).
    """

    def __init__(
        self,
        agent1_model_path: str,
        compute_device: torch.device,
        render_mode=None,
    ):

        self.agent1_target_x = ROBOT_HOME_GRID_X
        self.agent1_target_y = ROBOT_HOME_GRID_Y

        # Build Agent 2's env first — it owns the pygame screen.
        # Pass shared_target_queue=None temporarily; it is set right after.
        self.agent2_env = Agent2TrainingEnv(
            shared_target_queue=None,
            render_mode=render_mode,
        )

        # Build the shared queue using Agent 2's shelf list and BFS function.
        self.shared_target_queue = SharedTargetQueue(
            shelves=self.agent2_env.shelves,
            bfs_fn=self.agent2_env._bfs_distance_map,
        )
        self.agent2_env.shared_target_queue = self.shared_target_queue

        # Agent 1's env — never rendered, no knowledge of Agent 2.
        self.agent1_env = Agent1TrainingEnv(
            shared_target_queue=self.shared_target_queue,
            render_mode=None,
        )

        self.render_mode = render_mode
        self.compute_device = compute_device

        # ── Load and freeze Agent 1's policy ─────────────────────────────────
        agent1_state_dim = self.agent1_env.observation_space.shape[0]
        agent1_action_dim = self.agent1_env.action_space.n
        self.agent1_policy_network = QNetwork(
            agent1_state_dim, agent1_action_dim
        ).to(compute_device)

        checkpoint = torch.load(agent1_model_path, map_location=compute_device)
        if isinstance(checkpoint, dict):
            agent1_weights = checkpoint.get("policy", checkpoint)
        else:
            agent1_weights = checkpoint
        self.agent1_policy_network.load_state_dict(agent1_weights)
        self.agent1_policy_network.eval()

        print(f"  ✅ Agent 1 policy loaded from '{agent1_model_path}' (frozen).")
        print(f"  🏠 Agent 1 home: ({ROBOT_HOME_GRID_X}, {ROBOT_HOME_GRID_Y})")
        print(f"  🏠 Agent 2 home: ({AGENT2_HOME_GRID_X}, {AGENT2_HOME_GRID_Y})")

        # Expose Agent 2's spaces to the training loop.
        self.observation_space = self.agent2_env.observation_space
        self.action_space = self.agent2_env.action_space

        # Internal bookkeeping.
        self._agent1_current_observation = None
        self.score = 0                    # Agent 2's delivery count this episode
        self.agent2_collision_count = 0   # times Agent 2 was forced to yield

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Resets both sub-environments and the shared target queue.
        Retries until the two robots start on different grid cells.
        """

        agent1_obs, _ = self.agent1_env.reset(seed=seed, options=options)
        agent2_obs, agent2_info = self.agent2_env.reset(seed=seed, options=options)

        self.shared_target_queue.reset()

        for _ in range(20):
            if (
                self.agent1_env.robot.grid_x != self.agent2_env.robot.grid_x
                or self.agent1_env.robot.grid_y != self.agent2_env.robot.grid_y
            ):
                break
            agent2_obs, agent2_info = self.agent2_env.reset(options=options)

        self._agent1_current_observation = agent1_obs
        self.score = 0
        self.agent2_collision_count = 0

        self._sync_agent1_state_into_agent2_env()
        agent2_obs = self.agent2_env._get_observation()

        return agent2_obs, agent2_info

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, agent2_action: int):
        """
        Steps both agents. Agent 1 always moves freely. Agent 2 is forced
        to yield whenever it would end up on Agent 1's cell.

        Step order (critical — do not reorder):
          1. Observe Agent 1's intended action from its frozen policy.
          2. Predict where Agent 1 will land.
          3. Predict where Agent 2 wants to go.
          4. Detect collision: does Agent 2 need to yield?
          5. Step Agent 1 freely (no changes ever).
          6. Step Agent 2 with effective action (wait if yielding required).
          7. Apply extra reward shaping.
          8. Sync Agent 1 state into Agent 2's env for next observation.
        """

        agent1_robot = self.agent1_env.robot
        agent2_robot = self.agent2_env.robot

        # ── 1. Observe Agent 1's intended action (frozen policy, no grad) ─────
        with torch.no_grad():
            agent1_state_tensor = torch.as_tensor(
                self._agent1_current_observation,
                dtype=torch.float32,
                device=self.compute_device,
            ).unsqueeze(0)
            agent1_action = self.agent1_policy_network(
                agent1_state_tensor
            ).argmax().item()

        # ── 2. Predict where Agent 1 will land ───────────────────────────────
        agent1_intended_next_x, agent1_intended_next_y = (
            self._predict_next_position(
                agent1_robot.grid_x,
                agent1_robot.grid_y,
                agent1_action,
                self.agent1_env.obstacle_positions,
            )
        )

        # ── 3. Predict where Agent 2 wants to go ─────────────────────────────
        agent2_intended_next_x, agent2_intended_next_y = (
            self._predict_next_position(
                agent2_robot.grid_x,
                agent2_robot.grid_y,
                agent2_action,
                self.agent2_env.obstacle_positions,
            )
        )

        # ── 4. Collision detection — two conditions where Agent 2 must yield ──
        #
        # Condition A: Agent 2 is moving into the same cell Agent 1 is moving into.
        agent2_moves_into_agent1_destination = (
            agent2_intended_next_x == agent1_intended_next_x
            and agent2_intended_next_y == agent1_intended_next_y
        )

        # Condition B: Agent 1 is moving INTO Agent 2's current cell.
        # Agent 2 must vacate (wait) so Agent 1 can enter freely.
        agent1_moves_into_agent2_current_cell = (
            agent1_intended_next_x == agent2_robot.grid_x
            and agent1_intended_next_y == agent2_robot.grid_y
        )

        agent2_must_yield = (
            agent2_moves_into_agent1_destination
            or agent1_moves_into_agent2_current_cell
        )

        # ── 5. Step Agent 1 freely — NEVER altered ────────────────────────────
        (
            agent1_next_obs,
            _agent1_reward,
            agent1_terminated,
            agent1_truncated,
            _,
        ) = self.agent1_env.step(agent1_action)
        self._agent1_current_observation = agent1_next_obs

        if agent1_terminated or agent1_truncated:
            # Silently reset Agent 1 so it continues as a moving obstacle.
            agent1_reset_obs, _ = self.agent1_env.reset()
            self._agent1_current_observation = agent1_reset_obs

        # ── 6. Step Agent 2 — override to WAIT if yielding is required ────────
        effective_agent2_action = agent2_action
        if agent2_must_yield and agent2_action < 4:
            # Agent 2 stays in place. The collision penalty is applied below
            # regardless — Agent 2 is penalised for intending the collision,
            # not just for the resulting wait.
            effective_agent2_action = 5
            self.agent2_collision_count += 1

        (
            agent2_next_obs,
            agent2_base_reward,
            agent2_terminated,
            agent2_truncated,
            agent2_info,
        ) = self.agent2_env.step(effective_agent2_action)

        # ── 7. Extra reward shaping ───────────────────────────────────────────
        extra_reward = self._calculate_agent2_extra_reward(
            agent2_robot=agent2_robot,
            agent1_next_grid_x=self.agent1_env.robot.grid_x,
            agent1_next_grid_y=self.agent1_env.robot.grid_y,
            agent2_was_forced_to_yield=agent2_must_yield,
            agent2_original_action=agent2_action,
            agent2_base_reward=agent2_base_reward,
        )

        total_agent2_reward = agent2_base_reward + extra_reward

        # ── 8. Sync Agent 1 state for Agent 2's next observation ──────────────
        self._sync_agent1_state_into_agent2_env()
        agent2_next_obs = self.agent2_env._get_observation()

        self.score = self.agent2_env.score

        return (
            agent2_next_obs,
            total_agent2_reward,
            agent2_terminated,
            agent2_truncated,
            agent2_info,
        )

    # ── reward shaping ────────────────────────────────────────────────────────

    def _calculate_agent2_extra_reward(
        self,
        agent2_robot,
        agent1_next_grid_x: int,
        agent1_next_grid_y: int,
        agent2_was_forced_to_yield: bool,
        agent2_original_action: int,
        agent2_base_reward: float,
    ) -> float:
        """
        Extra reward adjustment added on top of Agent 2's base step reward.

        The collision penalty fires whenever Agent 2 was forced to yield —
        whether that was because it moved toward Agent 1's destination (Condition A)
        or failed to clear Agent 1's path (Condition B). This teaches Agent 2
        to anticipate both situations proactively.

        Proximity penalties apply when there was no hard collision, giving
        Agent 2 a softer gradient to learn to stay at a safe distance.

        The yielding bonus rewards voluntary patience near Agent 1 while
        Agent 2 is still making progress — encouraging learned cooperation
        rather than purely reactive collision avoidance.
        """
        extra_reward = 0.0

        # ── Hard collision / yield penalty ────────────────────────────────────
        if agent2_was_forced_to_yield:
            extra_reward += AGENT_COLLISION_PENALTY
            # Return early — no proximity penalty on top of a hard collision.
            return extra_reward

        # ── Soft proximity penalties ──────────────────────────────────────────
        manhattan_distance_to_agent1 = abs(
            agent2_robot.grid_x - agent1_next_grid_x
        ) + abs(agent2_robot.grid_y - agent1_next_grid_y)

        if manhattan_distance_to_agent1 == 1:
            extra_reward += PROXIMITY_DISTANCE_1_PENALTY
        elif manhattan_distance_to_agent1 == 2:
            extra_reward += PROXIMITY_DISTANCE_2_PENALTY

        # ── Voluntary yielding bonus ──────────────────────────────────────────
        agent2_voluntarily_yielded = agent2_original_action in (4, 5)
        agent1_is_very_close = (
            manhattan_distance_to_agent1 <= YIELDING_BONUS_PROXIMITY_THRESHOLD
        )
        agent2_still_progressing = agent2_base_reward > 0.0

        if agent2_voluntarily_yielded and agent1_is_very_close and agent2_still_progressing:
            extra_reward += YIELDING_BONUS

        return extra_reward

    # ── helpers ───────────────────────────────────────────────────────────────

    def _predict_next_position(
        self,
        current_grid_x: int,
        current_grid_y: int,
        action: int,
        obstacle_positions: set,
    ) -> tuple:
        """
        Returns the grid cell an agent will occupy after `action` without
        mutating any environment state. Mirrors WarehouseEnv.step exactly.

        Non-movement actions (interact=4, wait=5) → stays put.
        Movement into wall / out-of-bounds         → stays put.
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

    def _sync_agent1_state_into_agent2_env(self):
        """
        Pushes Agent 1's current position and status into Agent 2's env
        so the next observation Agent 2 receives is accurate.
        """
        self.agent2_env.update_agent1_state(
            agent1_grid_x=self.agent1_env.robot.grid_x,
            agent1_grid_y=self.agent1_env.robot.grid_y,
            agent1_loaded=self.agent1_env.robot.loaded,
            agent1_returning_home=self.agent1_env.returning_home,
        )

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        """
        Renders Agent 2's environment (which owns the pygame screen).
        Agent 1 is drawn on top as a visually distinct overlay.
        """
        if self.render_mode is None:
            return

        self.agent2_env.render()

        if self.agent2_env.screen is not None:
            import pygame

            agent1_pixel_x = (
                PADDING_BORDER + self.agent1_env.robot.grid_x * GRID_SPACING
            )
            agent1_pixel_y = (
                PADDING_BORDER + self.agent1_env.robot.grid_y * GRID_SPACING
            )

            agent1_image = (
                ROBOT_IMAGE_SIDE_BOX
                if self.agent1_env.robot.loaded
                else ROBOT_IMAGE_SIDE
            )
            center_offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
            center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
            self.agent2_env.screen.blit(
                agent1_image,
                (
                    agent1_pixel_x + center_offset_x,
                    agent1_pixel_y + center_offset_y,
                ),
            )

            # Cyan border makes Agent 1 unmistakably distinct from Agent 2.
            pygame.draw.rect(
                self.agent2_env.screen,
                (0, 220, 220),
                (agent1_pixel_x, agent1_pixel_y, TILE_SIZE, TILE_SIZE),
                2,
            )

            pygame.display.flip()

    def heuristic_action(self):
        """Exposes Agent 2's BFS heuristic for use in the training loop."""
        return self.agent2_env.heuristic_action()