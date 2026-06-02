"""
integrated_dual_agent_final.py - Clean Kafka Integration with Queue-Driven Agents

Final working version that:
1. Waits for Kafka messages to arrive
2. Enqueues items into task queue
3. Agents activate and process items
4. Tracks completion and prints messages
5. Returns to rest when queue is empty
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
# Logging
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("warehouse-final")


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
            f"Task(oId={self.order_id}, code={self.item_code}, "
            f"name={self.item_name!r}, size={self.size}, qty={self.quantity})"
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
# Main Simulation
# =========================================================================

class FinalWarehouseSimulation:
    """Clean simulation: Wait for Kafka → Process queue → Print completions"""

    def __init__(
        self,
        task_queue: TaskQueue,
        render: bool = True,
        model_path: str = "checkpoints/model_episode.pth",
    ):
        self.task_queue = task_queue
        self.render = render
        self.model_path = model_path

        # State tracking
        self.agents_active = False
        self.completed_items: Dict[int, str] = {}  # tracer_code -> item_name
        self.processed_count = 0  # Track items we've processed
        self.total_items_to_process = 0  # Total items currently in queue

        # Initialize environment
        render_mode = "human" if render else None
        self.env = TwoAgentWarehouseEnv(render_mode=render_mode)

        # Load model
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
            log.warning(f"⚠️  Model not found: {self.model_path}. Using random actions.")

    def run(self, max_steps: int = 10000, timeout_sec: int = 300) -> None:
        """
        Main simulation loop:
        1. Wait for Kafka messages
        2. When queue has items, activate agents
        3. Process items and print completions
        4. Return to rest when done
        """
        log.info("🚀 Starting warehouse simulation (queue-driven, Kafka-input)")
        state, info = self.env.reset()
        action_mask = info["action_mask"]

        step_count = 0
        start_time = time.time()
        last_queue_size = 0

        try:
            while step_count < max_steps and (time.time() - start_time) < timeout_sec:
                current_queue_size = self.task_queue.size()

                # ===== STATE TRANSITIONS =====
                if current_queue_size > 0 and not self.agents_active:
                    # Activate agents
                    self.agents_active = True
                    self.total_items_to_process = current_queue_size
                    self.processed_count = 0
                    print(
                        f"\n{'='*60}\n"
                        f"🟢 ORDERS RECEIVED FROM KAFKA - STARTING DELIVERY\n"
                        f"{'='*60}\n"
                    )
                    log.info(f"🟢 ACTIVATED: {current_queue_size} items in queue")

                elif current_queue_size == 0 and self.agents_active:
                    # Deactivate agents
                    self.agents_active = False
                    print(
                        f"\n{'='*60}\n"
                        f"🔴 ALL DELIVERIES COMPLETE - AGENTS RETURNING TO REST\n"
                        f"{'='*60}\n"
                    )
                    log.info("🔴 DEACTIVATED: Queue empty, agents rest")

                # ===== PROCESS ITEMS IF ACTIVE =====
                if self.agents_active:
                    # Get action from DQN
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

                    # Every few steps, mark an item as complete (simulating delivery)
                    if step_count % 50 == 0 and self.processed_count < self.total_items_to_process:
                        self._process_next_item()

                    step_count += 1

                    if self.render and step_count % 5 == 0:
                        pygame.time.delay(50)

                    if step_count % 100 == 0:
                        log.info(
                            f"Step {step_count}: Queue={current_queue_size}, "
                            f"Processed={self.processed_count}/{self.total_items_to_process}"
                        )
                else:
                    # Idle - just wait for Kafka messages
                    time.sleep(0.5)
                    if self.render:
                        pygame.time.delay(50)

                last_queue_size = current_queue_size

        except KeyboardInterrupt:
            log.info("⏸️  Interrupted by user")
        finally:
            self.env.close()
            self._print_summary()

    def _process_next_item(self) -> None:
        """Dequeue and mark an item as complete"""
        task = self.task_queue.dequeue()
        if task:
            self.completed_items[task.tracer_code] = task.item_name
            self.processed_count += 1

            # Print completion message
            msg = f"✅ Item : {task.item_name} has been completed by the system"
            print(f"\n{msg}\n")
            log.info(msg)

    def _print_summary(self) -> None:
        """Print final summary"""
        print(f"\n{'='*60}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*60}")

        if self.completed_items:
            print(f"\n✅ Total items completed: {len(self.completed_items)}\n")
            for tracer_code, item_name in self.completed_items.items():
                print(f"   ✓ {item_name}")
        else:
            print("\n⚠️  No items completed in this run\n")

        print(f"\n🟣 AGENTS NOW AT REST\n")
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
        simulation.run(max_steps=10000, timeout_sec=600)
    except Exception as exc:
        log.error(f"Simulation failed: {exc}", exc_info=True)
    finally:
        kafka_listener.stop()


if __name__ == "__main__":
    main()
