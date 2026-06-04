"""
integrated_dual_agent_final.py - Clean Kafka Integration with Queue-Driven Agents

Final working version that:
1. Waits for Kafka messages to arrive
2. Enqueues items into task queue
3. Assigns tasks to agents round-robin (first-finished-gets-next)
4. Agents navigate to the correct shelf (mapped by itemCode + size)
5. Each item is picked up and delivered quantity times
6. Sends HTTP callback on item completion
7. Returns to rest when queue is empty
"""

import json
import logging
import os
import sys
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum

import numpy as np
import requests
import torch
import torch.nn as nn
import pygame

# Import existing components
from unified_env import (
    TwoAgentWarehouseEnv,
    bfs_distance_map,
    bfs_best_action,
    predict_next,
    PHASE_FETCHING,
    PHASE_DELIVERING,
    PHASE_HOMING,
    AGENT1_HOME_X, AGENT1_HOME_Y,
    AGENT2_HOME_X, AGENT2_HOME_Y,
)
from reader import (
    WarehouseData,
    WarehouseItemData,
    OrderKafkaReader,
    KafkaConfig,
)
from constants import PADDING_BORDER, GRID_SPACING

# =========================================================================
# Logging
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("warehouse-final")


# =========================================================================
# Configuration
# =========================================================================

COMPLETION_API_BASE = "http://localhost:8080/api/v1/admin/pack/item"
COMPLETION_API_HEADER_KEY = "SystemKey"
COMPLETION_API_HEADER_VALUE = "SystemKey illsontpygamesystem"

# Number of items in the e-commerce catalog (codes 1-10)
NUM_ITEM_CODES = 10
# Number of sizes
NUM_SIZES = 5
# Total shelf slots needed
TOTAL_SHELF_SLOTS = NUM_ITEM_CODES * NUM_SIZES  # 50


# =========================================================================
# Size Mapping
# =========================================================================

class SizeEnum(Enum):
    """Maps size strings to indices (0-4)"""
    SMALL = ("small", 0)
    MEDIUM = ("medium", 1)
    LARGE = ("large", 2)
    XL = ("xl", 3)
    XXL = ("xxl", 4)

    @classmethod
    def from_string(cls, size_str: str) -> int:
        """Convert size string (case-insensitive) to index"""
        size_lower = size_str.lower().strip()
        for size in cls:
            if size.value[0] == size_lower:
                return size.value[1]
        raise ValueError(f"Unknown size: {size_str}")


def compute_shelf_index(item_code: int, size_str: str) -> int:
    """
    Compute deterministic shelf index from item code and size.

    Formula: (item_code - 1) * 5 + size_index
    Gives indices 0-49 for items 1-10 and 5 sizes.
    """
    size_index = SizeEnum.from_string(size_str)
    return (item_code - 1) * NUM_SIZES + size_index


# =========================================================================
# Task Management
# =========================================================================

@dataclass
class WarehouseTask:
    """Represents a single item to be picked and delivered"""
    order_id: int
    tracer_code: int
    item_name: str
    item_code: int
    size: str
    quantity: int
    shelf_index: int  # Computed (item_code - 1) * 5 + size_index

    @classmethod
    def from_warehouse_item(cls, order_id: int, item: WarehouseItemData) -> "WarehouseTask":
        """Convert WarehouseItemData to WarehouseTask"""
        shelf_index = compute_shelf_index(item.item_code, item.size)
        return cls(
            order_id=order_id,
            tracer_code=item.order_tracer_code,
            item_name=item.item_name,
            item_code=item.item_code,
            size=item.size,
            quantity=item.quantity,
            shelf_index=shelf_index,
        )

    def __str__(self) -> str:
        return (
            f"Task(oId={self.order_id}, code={self.item_code}, "
            f"name={self.item_name!r}, size={self.size}, qty={self.quantity}, "
            f"shelf={self.shelf_index}, tracer={self.tracer_code})"
        )


class TaskQueue:
    """Thread-safe queue for warehouse tasks"""

    def __init__(self):
        self._queue: deque = deque()
        self._lock = threading.Lock()

    def enqueue(self, task: WarehouseTask) -> None:
        """Add a task to the queue"""
        with self._lock:
            self._queue.append(task)
            log.info(f"Enqueued: {task}")

    def dequeue(self) -> Optional[WarehouseTask]:
        """Remove and return the next task"""
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def peek(self) -> Optional[WarehouseTask]:
        """View the next task without removing it"""
        with self._lock:
            return self._queue[0] if self._queue else None

    def size(self) -> int:
        """Get queue size"""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if the queue is empty"""
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> None:
        """Clear the queue"""
        with self._lock:
            self._queue.clear()


# =========================================================================
# Kafka Integration
# =========================================================================

class KafkaOrderListener:
    """Listens to Kafka and feeds tasks into the queue"""

    def __init__(self, task_queue: TaskQueue, config: Optional[KafkaConfig] = None):
        self.task_queue = task_queue
        self.config = config or KafkaConfig()
        self.reader = None
        self._stop_evt = threading.Event()
        self._processed_orders: Set[int] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start listening for Kafka messages"""
        try:
            self.reader = OrderKafkaReader(self.config)
            # Hook the message handler
            self.reader._handle_message = self._handle_warehouse_message
            self.reader.start()
            log.info("✅ Kafka order listener started")
        except Exception as e:
            log.error(f"❌ Failed to start Kafka listener: {e}")
            raise

    def stop(self) -> None:
        """Stop listening"""
        self._stop_evt.set()
        if self.reader:
            self.reader.stop()
        log.info("Kafka order listener stopped")

    def _handle_warehouse_message(self, message) -> None:
        """Process incoming Kafka message and enqueue tasks"""
        try:
            warehouse_data = WarehouseData.from_json(message.value)
            order_id = warehouse_data.o_id

            with self._lock:
                # Avoid duplicate processing
                if order_id in self._processed_orders:
                    log.debug(f"Order {order_id} already processed")
                    return
                self._processed_orders.add(order_id)

            log.info(f"📨 Order {order_id}: {len(warehouse_data.items)} items")

            # Convert each item to a task and enqueue
            for item in warehouse_data.items:
                task = WarehouseTask.from_warehouse_item(order_id, item)
                self.task_queue.enqueue(task)

            log.info(f"📦 Queue size: {self.task_queue.size()}")

        except Exception as exc:
            log.error(f"Failed to process message: {exc}", exc_info=True)


# =========================================================================
# Deep Q-Network
# =========================================================================

class DeepQNetwork(nn.Module):
    def __init__(self, observation_dimension: int, total_actions: int):
        super(DeepQNetwork, self).__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(observation_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_actions),
        )

    def forward(self, state_tensor):
        return self.network_layers(state_tensor)


# =========================================================================
# Agent State Machine
# =========================================================================

class AgentPhase(Enum):
    """Phases of an agent's task lifecycle"""
    IDLE = "idle"           # At home, waiting for a task
    FETCHING = "fetching"   # Navigating to shelf to pick up
    DELIVERING = "delivering"  # Navigating to dropoff to deliver
    HOMING = "homing"       # Returning home (after task complete or between quantities)


@dataclass
class AgentState:
    """Tracks the current state of a single agent's task processing"""
    name: str                          # "Robot1" or "Robot2"
    home_x: int
    home_y: int
    dropoff_gx: int = 0
    dropoff_gy: int = 0
    phase: AgentPhase = AgentPhase.IDLE
    current_task: Optional[WarehouseTask] = None
    remaining_quantity: int = 0
    target_gx: int = 0                # Current navigation target grid x
    target_gy: int = 0                # Current navigation target grid y
    deliveries_completed: int = 0     # How many deliveries done for current task

    def is_busy(self) -> bool:
        return self.phase != AgentPhase.IDLE

    def assign_task(self, task: WarehouseTask, shelf_gx: int, shelf_gy: int):
        """Assign a new task to this agent"""
        self.current_task = task
        self.remaining_quantity = task.quantity
        self.deliveries_completed = 0
        self.target_gx = shelf_gx
        self.target_gy = shelf_gy
        self.phase = AgentPhase.FETCHING
        log.info(
            f"🤖 {self.name} assigned: {task.item_name} (code={task.item_code}, "
            f"size={task.size}, qty={task.quantity}) → shelf ({shelf_gx},{shelf_gy})"
        )

    def complete_delivery(self) -> bool:
        """
        Called after a delivery. Decrements remaining quantity.
        Returns True if the entire task is now complete.
        """
        self.remaining_quantity -= 1
        self.deliveries_completed += 1
        log.info(
            f"📦 {self.name} delivered {self.current_task.item_name} "
            f"({self.deliveries_completed}/{self.current_task.quantity})"
        )
        if self.remaining_quantity <= 0:
            return True
        return False

    def finish_task(self):
        """Reset after task is fully complete"""
        task = self.current_task
        self.current_task = None
        self.remaining_quantity = 0
        self.deliveries_completed = 0
        self.phase = AgentPhase.HOMING
        self.target_gx = self.home_x
        self.target_gy = self.home_y
        return task


# =========================================================================
# HTTP Completion Callback
# =========================================================================

def send_completion_callback(tracer_code: int) -> bool:
    """
    Send HTTP POST to mark an item as complete.

    POST http://localhost:8080/api/v1/admin/pack/item/{orderTracerCode}
    Header: SystemKey = "SystemKey illsontpygamesystem"

    Returns True on success, False on failure.
    """
    url = f"{COMPLETION_API_BASE}/{tracer_code}"
    headers = {COMPLETION_API_HEADER_KEY: COMPLETION_API_HEADER_VALUE}
    try:
        response = requests.post(url, headers=headers, timeout=5)
        if response.status_code == 200:
            log.info(f"✅ HTTP callback success: {url} (status={response.status_code})")
            return True
        else:
            log.warning(f"⚠️ HTTP callback returned status {response.status_code}: {url}")
            return False
    except requests.exceptions.RequestException as exc:
        log.error(f"❌ HTTP callback failed for tracer {tracer_code}: {exc}")
        return False


# =========================================================================
# Shelf Coordinate Mapper
# =========================================================================

class ShelfMapper:
    """
    Maps (item_code, size) → grid coordinates using the environment's
    shelf list from world.py.

    The first TOTAL_SHELF_SLOTS (50) shelves in the list are used.
    shelf_index = (item_code - 1) * 5 + size_index
    """

    def __init__(self, shelves: list):
        self._coords: Dict[int, Tuple[int, int]] = {}
        for idx, shelf in enumerate(shelves):
            if idx >= TOTAL_SHELF_SLOTS:
                break
            gx = round((shelf.x - PADDING_BORDER) / GRID_SPACING)
            gy = round((shelf.y - PADDING_BORDER) / GRID_SPACING)
            self._coords[idx] = (gx, gy)

        log.info(f"ShelfMapper: mapped {len(self._coords)} shelves to grid coordinates")

    def get_coords(self, shelf_index: int) -> Tuple[int, int]:
        """Get grid (x, y) for a shelf index"""
        if shelf_index not in self._coords:
            raise ValueError(
                f"Shelf index {shelf_index} not mapped. "
                f"Valid range: 0–{len(self._coords) - 1}"
            )
        return self._coords[shelf_index]

    def get_shelf_index(self, item_code: int, size_str: str) -> int:
        """Compute shelf index from item_code and size"""
        return compute_shelf_index(item_code, size_str)

    def get_coords_for_item(self, item_code: int, size_str: str) -> Tuple[int, int]:
        """Get grid coords for an item+size combination"""
        idx = self.get_shelf_index(item_code, size_str)
        return self.get_coords(idx)


# =========================================================================
# Main Simulation
# =========================================================================

class FinalWarehouseSimulation:
    """
    Queue-driven dual-agent warehouse simulation.

    Flow:
    1. Kafka messages arrive → tasks enqueued
    2. Tasks assigned to agents round-robin (first-finished-gets-next)
    3. Agent navigates to shelf → picks up → navigates to dropoff → delivers
    4. Repeats quantity times per task
    5. HTTP callback on task completion
    6. Agents rest at home when idle
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        render: bool = True,
        model_path: str = "checkpoints/model_episode.pth",
    ):
        self.task_queue = task_queue
        self.render = render
        self.model_path = model_path

        # Initialize environment
        render_mode = "human" if render else None
        self.env = TwoAgentWarehouseEnv(render_mode=render_mode)

        # Build shelf mapper from environment shelves
        self.shelf_mapper = ShelfMapper(self.env.shelves)

        # Build obstacle set for BFS
        self.obstacle_positions = self.env.obstacle_positions

        # Agent states
        self.agent1_state = AgentState(
            name="Robot1",
            home_x=AGENT1_HOME_X,
            home_y=AGENT1_HOME_Y,
            dropoff_gx=self.env.r1_dropoff_gx,
            dropoff_gy=self.env.r1_dropoff_gy,
        )
        self.agent2_state = AgentState(
            name="Robot2",
            home_x=AGENT2_HOME_X,
            home_y=AGENT2_HOME_Y,
            dropoff_gx=self.env.r2_dropoff_gx,
            dropoff_gy=self.env.r2_dropoff_gy,
        )

        # Completion tracking
        self.completed_tasks: List[WarehouseTask] = []
        self.total_deliveries = 0

        # Load DQN model for R2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DeepQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
        ).to(self.device)
        self._load_model()

    def _load_model(self) -> None:
        """Load pre-trained model"""
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
            log.info(f"✅ Loaded model: {self.model_path}")
        except FileNotFoundError:
            log.warning(f"⚠️  Model not found: {self.model_path}. Using BFS fallback.")

    # -----------------------------------------------------------------
    # Task Dispatching
    # -----------------------------------------------------------------

    def _try_assign_tasks(self) -> None:
        """
        Try to assign tasks from the queue to idle agents.
        Round-robin: Agent1 first, then Agent2, then whichever is free.
        """
        for agent_state in [self.agent1_state, self.agent2_state]:
            if agent_state.is_busy():
                continue
            if self.task_queue.is_empty():
                break

            task = self.task_queue.dequeue()
            if task is None:
                break

            # Get shelf coordinates
            try:
                shelf_gx, shelf_gy = self.shelf_mapper.get_coords(task.shelf_index)
            except ValueError as e:
                log.error(f"Cannot map task to shelf: {e}. Skipping task: {task}")
                continue

            # Mark the shelf as having a box (so the environment renders it)
            self._set_shelf_box(task.shelf_index, has_box=True)

            # Assign to agent
            agent_state.assign_task(task, shelf_gx, shelf_gy)

            # Configure the environment's internal targets for this agent
            self._configure_env_agent(agent_state)

    def _set_shelf_box(self, shelf_index: int, has_box: bool) -> None:
        """Set the has_box flag on a shelf in the environment"""
        if shelf_index < len(self.env.shelves):
            shelf = self.env.shelves[shelf_index]
            shelf.has_box = has_box
            if has_box and hasattr(shelf, 'loaded_image'):
                shelf.image = shelf.loaded_image
            elif not has_box and hasattr(shelf, 'empty_image'):
                shelf.image = shelf.empty_image

    def _configure_env_agent(self, agent_state: AgentState) -> None:
        """
        Configure the environment's internal R1/R2 phase and target
        to match our agent state machine.
        """
        if agent_state.name == "Robot1":
            if agent_state.phase == AgentPhase.FETCHING:
                self.env._r1_phase = PHASE_FETCHING
                self.env._r1_target_gx = agent_state.target_gx
                self.env._r1_target_gy = agent_state.target_gy
                self.env._r1_target_dist = bfs_distance_map(
                    agent_state.target_gx, agent_state.target_gy,
                    self.obstacle_positions
                )
                self.env.a1_policy.current_shelf_target_x = agent_state.target_gx
                self.env.a1_policy.current_shelf_target_y = agent_state.target_gy
            elif agent_state.phase == AgentPhase.DELIVERING:
                self.env._r1_phase = PHASE_DELIVERING
            elif agent_state.phase in (AgentPhase.HOMING, AgentPhase.IDLE):
                self.env._r1_phase = PHASE_HOMING
                self.env._r1_target_gx = AGENT1_HOME_X
                self.env._r1_target_gy = AGENT1_HOME_Y
                self.env._r1_target_dist = self.env.r1_home_dist

        elif agent_state.name == "Robot2":
            if agent_state.phase == AgentPhase.FETCHING:
                self.env._r2_phase = PHASE_FETCHING
                self.env._r2_target_gx = agent_state.target_gx
                self.env._r2_target_gy = agent_state.target_gy
                self.env._r2_target_dist = bfs_distance_map(
                    agent_state.target_gx, agent_state.target_gy,
                    self.obstacle_positions
                )
            elif agent_state.phase == AgentPhase.DELIVERING:
                self.env._r2_phase = PHASE_DELIVERING
            elif agent_state.phase in (AgentPhase.HOMING, AgentPhase.IDLE):
                self.env._r2_phase = PHASE_HOMING
                self.env._r2_target_gx = AGENT2_HOME_X
                self.env._r2_target_gy = AGENT2_HOME_Y
                self.env._r2_target_dist = self.env.r2_home_dist

    # -----------------------------------------------------------------
    # Phase Monitoring (check if agents completed pickup/delivery)
    # -----------------------------------------------------------------

    def _monitor_agent_phases(self) -> None:
        """
        After each env step, check if agents have completed a pickup or
        delivery and update our state machine accordingly.
        """
        self._monitor_single_agent(
            self.agent1_state,
            self.env.robot1,
            is_robot1=True,
        )
        self._monitor_single_agent(
            self.agent2_state,
            self.env.robot2,
            is_robot1=False,
        )

    def _monitor_single_agent(self, agent_state: AgentState, robot, is_robot1: bool) -> None:
        """Monitor a single agent's phase transitions"""
        if not agent_state.is_busy():
            # If agent is idle and at home, keep env in homing
            if is_robot1:
                if self.env._r1_phase != PHASE_HOMING:
                    self.env._r1_phase = PHASE_HOMING
            else:
                if self.env._r2_phase != PHASE_HOMING:
                    self.env._r2_phase = PHASE_HOMING
            return

        task = agent_state.current_task
        if task is None:
            return

        if agent_state.phase == AgentPhase.FETCHING:
            # Check if the env agent just picked up (loaded became True)
            if robot.loaded:
                log.info(f"📥 {agent_state.name} picked up {task.item_name} from shelf ({agent_state.target_gx},{agent_state.target_gy})")
                agent_state.phase = AgentPhase.DELIVERING
                agent_state.target_gx = agent_state.dropoff_gx
                agent_state.target_gy = agent_state.dropoff_gy
                self._configure_env_agent(agent_state)

        elif agent_state.phase == AgentPhase.DELIVERING:
            # Check if the env agent just delivered (loaded became False after being True)
            if not robot.loaded:
                self.total_deliveries += 1
                task_complete = agent_state.complete_delivery()

                if task_complete:
                    # All quantities delivered for this task
                    finished_task = agent_state.finish_task()
                    self._on_task_complete(finished_task, agent_state.name)
                    self._configure_env_agent(agent_state)
                else:
                    # More quantities to deliver — go back to the shelf
                    shelf_gx, shelf_gy = self.shelf_mapper.get_coords(task.shelf_index)
                    agent_state.target_gx = shelf_gx
                    agent_state.target_gy = shelf_gy
                    agent_state.phase = AgentPhase.FETCHING
                    # Re-mark the shelf as having a box for next pickup
                    self._set_shelf_box(task.shelf_index, has_box=True)
                    self._configure_env_agent(agent_state)
                    log.info(
                        f"🔄 {agent_state.name} returning to shelf for next quantity "
                        f"({agent_state.remaining_quantity} remaining)"
                    )

        elif agent_state.phase == AgentPhase.HOMING:
            # Check if agent reached home
            if robot.grid_x == agent_state.home_x and robot.grid_y == agent_state.home_y:
                agent_state.phase = AgentPhase.IDLE
                log.info(f"🏠 {agent_state.name} is now resting at home")

    def _on_task_complete(self, task: WarehouseTask, agent_name: str) -> None:
        """Handle a fully completed task (all quantities delivered)"""
        self.completed_tasks.append(task)

        # Print completion message
        msg = f"✅ Item : {task.item_name} has been completed by the system"
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"  Order: {task.order_id} | Tracer: {task.tracer_code} | Agent: {agent_name}")
        print(f"  Delivered {task.quantity}x {task.item_name} ({task.size})")
        print(f"{'='*60}\n")
        log.info(msg)

        # Send HTTP callback in a background thread to avoid blocking
        threading.Thread(
            target=send_completion_callback,
            args=(task.tracer_code,),
            name=f"http-callback-{task.tracer_code}",
            daemon=True,
        ).start()

    # -----------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------

    def run(self, max_steps: int = 100000, timeout_sec: int = 600) -> None:
        """
        Main simulation loop:
        1. Wait for Kafka messages
        2. Assign tasks to idle agents
        3. Step the environment
        4. Monitor pickup/delivery events
        5. Return to rest when done
        """
        log.info("🚀 Starting warehouse simulation (queue-driven, Kafka-input)")
        print(f"\n{'='*60}")
        print("  🏭 WAREHOUSE SIMULATION STARTED")
        print("  Waiting for orders from Kafka...")
        print(f"{'='*60}\n")

        state, info = self.env.reset()
        action_mask = info["action_mask"]

        # Clear the environment's internal random queue so we control targets
        self.env.queue.clear()

        step_count = 0
        start_time = time.time()
        was_active = False

        try:
            while step_count < max_steps and (time.time() - start_time) < timeout_sec:
                # Handle pygame events
                if self.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            log.info("Window closed by user")
                            return

                # Try to assign tasks from queue to idle agents
                self._try_assign_tasks()

                # Check if any agent is active
                any_active = self.agent1_state.is_busy() or self.agent2_state.is_busy()

                # Print status transitions
                if any_active and not was_active:
                    print(f"\n{'='*60}")
                    print("  🟢 ORDERS RECEIVED FROM KAFKA — STARTING DELIVERY")
                    print(f"  Queue: {self.task_queue.size()} items pending")
                    print(f"{'='*60}\n")
                elif not any_active and was_active:
                    print(f"\n{'='*60}")
                    print("  🔴 ALL DELIVERIES COMPLETE — AGENTS RETURNING TO REST")
                    print(f"  Total items completed: {len(self.completed_tasks)}")
                    print(f"{'='*60}\n")

                was_active = any_active

                if any_active:
                    # Prevent environment from spawning random targets
                    self.env.queue.clear()

                    # Get R2 action from DQN (or BFS fallback)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.policy_net(state_tensor)[0]
                        mask_tensor = torch.tensor(
                            action_mask, dtype=torch.bool, device=self.device
                        )
                        q_values[~mask_tensor] = -float("inf")
                        best_action = int(q_values.argmax().item())

                    # Step environment
                    state, reward, done, truncated, info = self.env.step(best_action)
                    action_mask = info["action_mask"]

                    # Monitor phase transitions (did agent pick up or deliver?)
                    self._monitor_agent_phases()

                    step_count += 1

                    if self.render and step_count % 3 == 0:
                        pygame.time.delay(30)

                    if done or truncated:
                        # Reset environment but preserve our agent states
                        state, info = self.env.reset()
                        action_mask = info["action_mask"]
                        self.env.queue.clear()
                        # Re-configure agents
                        self._configure_env_agent(self.agent1_state)
                        self._configure_env_agent(self.agent2_state)

                    # Log progress periodically
                    if step_count % 200 == 0:
                        log.info(
                            f"Step {step_count}: R1={self.agent1_state.phase.value}, "
                            f"R2={self.agent2_state.phase.value}, "
                            f"Queue={self.task_queue.size()}, "
                            f"Completed={len(self.completed_tasks)}"
                        )
                else:
                    # Idle — wait for Kafka messages
                    time.sleep(0.5)
                    if self.render:
                        self.env.render()

        except KeyboardInterrupt:
            log.info("⏸️  Interrupted by user")
        finally:
            self.env.close()
            self._print_summary()

    def _print_summary(self) -> None:
        """Print final summary"""
        print(f"\n{'='*60}")
        print("  📊 FINAL SUMMARY")
        print(f"{'='*60}")

        if self.completed_tasks:
            print(f"\n  ✅ Total items completed: {len(self.completed_tasks)}")
            print(f"  📦 Total deliveries made: {self.total_deliveries}\n")
            for task in self.completed_tasks:
                print(
                    f"   ✓ {task.item_name} (code={task.item_code}, size={task.size}, "
                    f"qty={task.quantity}, tracer={task.tracer_code})"
                )
        else:
            print("\n  ⚠️  No items completed in this run\n")

        print(f"\n  🟣 AGENTS NOW AT REST")
        print(f"{'='*60}\n")

        log.info("Simulation complete")


# =========================================================================
# Main
# =========================================================================

def main():
    """Main entry point"""
    # Create task queue
    task_queue = TaskQueue()

    # Create Kafka listener
    kafka_listener = KafkaOrderListener(task_queue)

    # Create simulation
    simulation = FinalWarehouseSimulation(
        task_queue,
        render=True,
        model_path="checkpoints/model_episode.pth",
    )

    # Signal handling
    def shutdown_handler(signum, frame):
        log.info(f"Signal {signum} — shutting down")
        kafka_listener.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Start Kafka listener
    try:
        kafka_listener.start()
        time.sleep(2)
    except Exception as e:
        log.error(f"Failed to start Kafka: {e}")
        log.warning("Continuing without Kafka (use test mode instead)")

    # Run simulation
    try:
        simulation.run(max_steps=100000, timeout_sec=3600)
    except Exception as exc:
        log.error(f"Simulation failed: {exc}", exc_info=True)
    finally:
        kafka_listener.stop()


if __name__ == "__main__":
    main()
