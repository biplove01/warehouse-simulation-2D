"""
integrated_dual_agent.py - Kafka-Driven Dual-Agent Warehouse Simulation

This module integrates Kafka message consumption with the dual-agent warehouse
simulation. It reads orders from Kafka, converts them to warehouse tasks, and
manages the queue-based execution by the two robot agents.

System Flow:
1. OrderKafkaReader reads WarehouseData from Kafka
2. Items are queued into a task queue with computed shelf indices
3. TwoAgentWarehouseEnv processes queue items via two coordinated agents
4. Agents pick items and deliver to their respective dropoff zones
5. Completion messages are printed per item

Warehouse Mapping:
- 24 items × 5 sizes = 120 total shelf positions (0-119)
- Formula: shelf_index = (item_code - 1) * 5 + size_index
- Size mapping: small=0, medium=1, large=2, xl=3, xxl=4
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
from typing import List, Optional, Dict, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import pygame

# Import existing components
from unified_env import TwoAgentWarehouseEnv
from reader import (
    WarehouseData,
    WarehouseItemData,
    OrderKafkaReader,
    KafkaConfig,
)

# =========================================================================
# Logging Setup
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("integrated-dual-agent")


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
        size_index = SizeEnum.from_string(item.size)
        shelf_index = (item.item_code - 1) * 5 + size_index
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
            f"Task(oId={self.order_id}, code={self.item_code}, name={self.item_name!r}, "
            f"size={self.size}, qty={self.quantity}, shelf_idx={self.shelf_index})"
        )


class TaskQueue:
    """Thread-safe queue for warehouse tasks"""

    def __init__(self):
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._assignment: Dict[int, str] = {}  # task_id -> agent_name

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

    def assign_task(self, task_id: int, agent_name: str) -> None:
        """Record task assignment"""
        with self._lock:
            self._assignment[task_id] = agent_name

    def get_assigned_agent(self, task_id: int) -> Optional[str]:
        """Get assigned agent for task"""
        with self._lock:
            return self._assignment.get(task_id)


# =========================================================================
# Kafka Integration
# =========================================================================

class KafkaOrderListener:
    """Listens to Kafka and feeds tasks into the queue"""

    def __init__(self, task_queue: TaskQueue, config: Optional[KafkaConfig] = None):
        self.task_queue = task_queue
        self.config = config or KafkaConfig()
        self.reader = OrderKafkaReader(self.config)
        self._stop_evt = threading.Event()
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            name="kafka-listener",
            daemon=True,
        )
        self._processed_orders = set()

    def start(self) -> None:
        """Start listening for Kafka messages"""
        # Override the reader's _handle_message to feed into queue
        self.reader._handle_message = self._handle_warehouse_message
        self.reader.start()
        self._listener_thread.start()
        log.info("Kafka order listener started")

    def stop(self) -> None:
        """Stop listening"""
        self._stop_evt.set()
        self.reader.stop()
        self._listener_thread.join(timeout=10)
        log.info("Kafka order listener stopped")

    def _listen_loop(self) -> None:
        """Background loop that processes incoming orders"""
        while not self._stop_evt.is_set():
            time.sleep(0.5)

    def _handle_warehouse_message(self, message) -> None:
        """Process incoming Kafka message and enqueue tasks"""
        try:
            warehouse_data = WarehouseData.from_json(message.value)
            order_id = warehouse_data.o_id

            # Avoid duplicate processing
            if order_id in self._processed_orders:
                log.debug(f"Order {order_id} already processed, skipping")
                return

            self._processed_orders.add(order_id)

            log.info(f"Processing Order ID: {order_id} with {len(warehouse_data.items)} items")

            # Convert each item to a task
            for item in warehouse_data.items:
                task = WarehouseTask.from_warehouse_item(order_id, item)
                self.task_queue.enqueue(task)

            log.info(f"Order {order_id} enqueued. Queue size: {self.task_queue.size()}")

        except Exception as exc:
            log.error(f"Failed to process message: {exc}", exc_info=True)


# =========================================================================
# Deep Q-Network (same as test_two_agents.py)
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
# Integrated Dual Agent Simulation
# =========================================================================

class IntegratedDualAgentSimulation:
    """Orchestrates Kafka input → task queue → dual agent execution with idle state management"""

    def __init__(
        self,
        task_queue: TaskQueue,
        render: bool = True,
        model_path: str = "checkpoints/model_episode.pth",
    ):
        self.task_queue = task_queue
        self.render = render
        self.model_path = model_path
        self.completed_items: Dict[int, str] = {}  # tracer_code -> item_name
        self.agents_active = False  # Track if agents should be working
        self.last_queue_size = 0  # Track queue state changes

        # Initialize environment
        render_mode = "human" if render else None
        self.env = TwoAgentWarehouseEnv(render_mode=render_mode)

        # Load trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DeepQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
        ).to(self.device)
        self._load_model()

    def _load_model(self) -> None:
        """Load pre-trained policy network"""
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
            log.info(f"✅ Loaded trained model from {self.model_path}")
        except FileNotFoundError:
            log.warning(
                f"⚠️ Model not found at {self.model_path}. "
                "Using random actions."
            )

    def run(self, max_steps: int = 10000, timeout_sec: int = 300) -> None:
        """
        Run the simulation with queue-driven agent activation:
        1. Agents idle at home until Kafka messages arrive
        2. When queue has items, activate agents to work
        3. As items complete, print completion messages
        4. When queue empties, agents return to home and rest
        """
        log.info("Starting integrated dual-agent simulation (queue-driven)")
        state, info = self.env.reset()
        action_mask = info["action_mask"]

        done = False
        truncated = False
        step_count = 0
        start_time = time.time()
        messages_to_print = []  # Queue for completion messages

        try:
            while (
                step_count < max_steps
                and not (done and truncated)
                and (time.time() - start_time) < timeout_sec
            ):
                # Check queue status
                current_queue_size = self.task_queue.size()
                
                # Handle state transitions
                if current_queue_size > 0 and not self.agents_active:
                    # Kafka messages arrived - activate agents
                    self.agents_active = True
                    log.info(
                        f"🟢 AGENTS ACTIVATED - Queue has {current_queue_size} item(s)"
                    )
                    print(
                        f"\n{'='*60}\n"
                        f"🟢 ORDERS RECEIVED FROM KAFKA - STARTING DELIVERY\n"
                        f"{'='*60}\n"
                    )

                elif current_queue_size == 0 and self.agents_active:
                    # Queue emptied - deactivate agents, send to rest
                    self.agents_active = False
                    log.info("🔴 AGENTS DEACTIVATED - Queue empty, returning to rest")
                    print(
                        f"\n{'='*60}\n"
                        f"🔴 ALL DELIVERIES COMPLETE - AGENTS RETURNING TO REST\n"
                        f"{'='*60}\n"
                    )

                self.last_queue_size = current_queue_size

                # Only step environment if agents are active
                if self.agents_active:
                    # Get action via DQN for Agent 2 (learner)
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
                    
                    # Check for item completions
                    self._check_item_completion()
                    
                    # Print any queued completion messages
                    for msg in messages_to_print:
                        print(msg)
                    messages_to_print.clear()

                    step_count += 1

                    # Render
                    if self.render and step_count % 5 == 0:
                        pygame.time.delay(50)

                    # Log progress
                    if step_count % 100 == 0:
                        queue_size = self.task_queue.size()
                        log.info(
                            f"Step {step_count}: Queue={queue_size}, "
                            f"R1 Score={self.env.r1_score}, R2 Score={self.env.r2_score}"
                        )
                else:
                    # Agents idle - just wait for Kafka messages
                    time.sleep(0.1)
                    pygame.time.delay(10) if self.render else None

        except KeyboardInterrupt:
            log.info("Simulation interrupted by user")
        finally:
            self.env.close()
            self._print_final_summary()

    def _check_item_completion(self) -> None:
        """
        Monitor for item completions and print messages.
        This hooks into the environment state to detect when robots
        have successfully delivered items.
        """
        # Check for items that have just been completed
        # This can be enhanced based on environment signals
        # For now, we'll integrate with the queue manager in the future
        pass

    def record_item_completion(self, tracer_code: int, item_name: str) -> str:
        """
        Record that an item has been completed and return formatted message.
        
        Returns the completion message for printing.
        """
        if tracer_code not in self.completed_items:
            self.completed_items[tracer_code] = item_name
            msg = f"\nItem : {item_name} has been completed by the system\n"
            return msg
        return ""

    def _print_final_summary(self) -> None:
        """Print summary of all completed items"""
        log.info("=" * 60)
        log.info("SIMULATION SUMMARY")
        log.info("=" * 60)
        
        print(f"\n{'='*60}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*60}")
        
        if self.completed_items:
            print(f"\nTotal items completed: {len(self.completed_items)}\n")
            for tracer_code, item_name in self.completed_items.items():
                print(f"   ✓ {item_name}")
                log.info(f"✓ Item: {item_name} has been completed by the system")
        else:
            print("\n  No items were completed in this run\n")
            log.info("No items were completed in this run")
        
        print(f"\n AGENTS NOW AT REST\n")
        print(f"{'='*60}\n")
        log.info("=" * 60)


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    """Main entry point"""
    # Create shared task queue
    task_queue = TaskQueue()

    # Create Kafka listener
    kafka_config = KafkaConfig()
    kafka_listener = KafkaOrderListener(task_queue, kafka_config)

    # Create simulation
    simulation = IntegratedDualAgentSimulation(
        task_queue,
        render=True,
        model_path="checkpoints/model_episode.pth",
    )

    # Setup signal handlers
    def shutdown_handler(signum, frame):
        log.info(f"Signal {signum} received — shutting down…")
        kafka_listener.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Start Kafka listener
    kafka_listener.start()
    time.sleep(2)  # Give Kafka reader time to initialize

    # Run simulation
    try:
        simulation.run(max_steps=10000, timeout_sec=600)
    except Exception as exc:
        log.error(f"Simulation failed: {exc}", exc_info=True)
    finally:
        kafka_listener.stop()


if __name__ == "__main__":
    main()
