import torch
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
        return self.agent2_env.heuristic_action()