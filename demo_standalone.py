"""
demo_standalone.py - Standalone Demo Without Kafka

This demo simulates warehouse orders without requiring Kafka connectivity.
It demonstrates:
1. Task creation from order data
2. Queue management
3. Task distribution to agents
4. Shelf index calculation
5. Item completion tracking

Run this to validate the integration before connecting to Kafka.
"""

import sys
import logging
import time
from collections import deque
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("demo-standalone")


# =========================================================================
# Simulated Data (without Kafka)
# =========================================================================

class SimulatedWarehouseOrder:
    """Represents a simulated order"""
    def __init__(self, order_id: int, items: List[dict]):
        self.o_id = order_id
        self.items = items


# Demo orders
DEMO_ORDERS = [
    SimulatedWarehouseOrder(
        order_id=38,
        items=[
            {
                "orderTracerCode": 52,
                "itemName": "Harden",
                "itemCode": 3,
                "size": "XXL",
                "quantity": 4,
            },
            {
                "orderTracerCode": 53,
                "itemName": "Random Item",
                "itemCode": 4,
                "size": "Medium",
                "quantity": 3,
            },
            {
                "orderTracerCode": 54,
                "itemName": "New Balance 530s",
                "itemCode": 1,
                "size": "Large",
                "quantity": 3,
            },
        ],
    ),
    SimulatedWarehouseOrder(
        order_id=39,
        items=[
            {
                "orderTracerCode": 55,
                "itemName": "Nike Air Max",
                "itemCode": 5,
                "size": "Medium",
                "quantity": 2,
            },
            {
                "orderTracerCode": 56,
                "itemName": "Adidas Ultraboost",
                "itemCode": 10,
                "size": "Large",
                "quantity": 1,
            },
        ],
    ),
]


# =========================================================================
# Import integration modules
# =========================================================================

try:
    from integrated_dual_agent import (
        WarehouseTask,
        WarehouseItemData,
        TaskQueue,
        SizeEnum,
    )
    from queue_manager import (
        QueueToEnvironmentBridge,
        compute_shelf_index,
        decompose_shelf_index,
    )
    log.info("✓ Successfully imported integration modules")
except ImportError as exc:
    log.error(f"✗ Failed to import integration modules: {exc}")
    sys.exit(1)


# =========================================================================
# Demo: Shelf Index Calculation
# =========================================================================

def demo_shelf_index_calculation():
    """Demonstrate shelf index formula"""
    log.info("=" * 60)
    log.info("DEMO 1: Shelf Index Calculation")
    log.info("=" * 60)

    test_cases = [
        (1, "small", 0),
        (1, "medium", 1),
        (1, "large", 2),
        (1, "xl", 3),
        (1, "xxl", 4),
        (2, "small", 5),
        (3, "xxl", 14),
        (24, "xxl", 119),
    ]

    for item_code, size, expected_idx in test_cases:
        computed_idx = compute_shelf_index(item_code, size)
        status = "✓" if computed_idx == expected_idx else "✗"
        log.info(
            f"{status} Item {item_code:2d}, {size:6s} → "
            f"index {computed_idx:3d} (expected {expected_idx:3d})"
        )

    log.info("")


# =========================================================================
# Demo: Shelf Index Decomposition
# =========================================================================

def demo_shelf_index_decomposition():
    """Demonstrate reverse shelf index calculation"""
    log.info("=" * 60)
    log.info("DEMO 2: Shelf Index Decomposition")
    log.info("=" * 60)

    test_indices = [0, 1, 4, 5, 14, 119]

    for idx in test_indices:
        item_code, size = decompose_shelf_index(idx)
        log.info(f"  Index {idx:3d} → Item {item_code:2d}, {size:6s}")

    log.info("")


# =========================================================================
# Demo: Task Creation
# =========================================================================

def demo_task_creation():
    """Demonstrate creating tasks from order items"""
    log.info("=" * 60)
    log.info("DEMO 3: Task Creation from Orders")
    log.info("=" * 60)

    queue = TaskQueue()

    for order in DEMO_ORDERS:
        log.info(f"\nProcessing Order ID: {order.o_id}")
        for item_dict in order.items:
            # Create WarehouseItemData from dict
            item = WarehouseItemData.from_dict(item_dict)
            # Create WarehouseTask
            task = WarehouseTask.from_warehouse_item(order.o_id, item)
            # Enqueue
            queue.enqueue(task)
            log.info(f"  ✓ {task}")

    log.info(f"\n✓ Total tasks in queue: {queue.size()}")
    log.info("")


# =========================================================================
# Demo: Queue Management
# =========================================================================

def demo_queue_management():
    """Demonstrate queue operations"""
    log.info("=" * 60)
    log.info("DEMO 4: Queue Management")
    log.info("=" * 60)

    queue = TaskQueue()

    # Populate queue
    all_tasks = []
    for order in DEMO_ORDERS:
        for item_dict in order.items:
            item = WarehouseItemData.from_dict(item_dict)
            task = WarehouseTask.from_warehouse_item(order.o_id, item)
            queue.enqueue(task)
            all_tasks.append(task)

    log.info(f"Queue size: {queue.size()}")

    # Dequeue one by one
    dequeued = []
    while queue.size() > 0:
        task = queue.dequeue()
        dequeued.append(task)
        log.info(f"  Dequeued: {task.item_name} (shelf_idx={task.shelf_index})")

    log.info(f"Dequeued {len(dequeued)} tasks. Queue now empty: {queue.size() == 0}")
    log.info("")


# =========================================================================
# Demo: Priority Rules
# =========================================================================

def demo_priority_rules():
    """Demonstrate single-item priority rule"""
    log.info("=" * 60)
    log.info("DEMO 5: Priority Rules (Single Item Priority to Robot1)")
    log.info("=" * 60)

    queue = TaskQueue()

    # Add single item
    item = WarehouseItemData.from_dict(DEMO_ORDERS[0].items[0])
    task = WarehouseTask.from_warehouse_item(DEMO_ORDERS[0].o_id, item)
    queue.enqueue(task)

    log.info(f"Queue size: {queue.size()} (single item)")

    # Simulate robot2 requesting
    log.info("\nRobot2 requests task from queue (single item in queue)")
    log.info("  → According to priority rule: DEFERRED")
    log.info("  → Priority given to Robot1")

    log.info("\nRobot1 requests task from queue")
    task = queue.dequeue()
    if task:
        log.info(f"  ✓ Robot1 receives: {task.item_name}")
    else:
        log.info("  ✗ No task available")

    log.info("")


# =========================================================================
# Demo: Task Distribution
# =========================================================================

def demo_task_distribution():
    """Simulate task distribution (without actual environment)"""
    log.info("=" * 60)
    log.info("DEMO 6: Task Distribution Simulation")
    log.info("=" * 60)

    queue = TaskQueue()

    # Populate queue
    for order in DEMO_ORDERS:
        for item_dict in order.items:
            item = WarehouseItemData.from_dict(item_dict)
            task = WarehouseTask.from_warehouse_item(order.o_id, item)
            queue.enqueue(task)

    log.info(f"Total tasks in queue: {queue.size()}\n")

    # Simulate distribution
    agents = ["Robot1", "Robot2", "Robot1"]
    for agent_name in agents:
        task = queue.peek()
        if task:
            log.info(f"{agent_name} gets task:")
            log.info(f"  Item: {task.item_name}")
            log.info(f"  Item Code: {task.item_code}")
            log.info(f"  Size: {task.size}")
            log.info(f"  Quantity: {task.quantity}")
            log.info(f"  Shelf Index: {task.shelf_index}")
            log.info(f"  Order Tracer Code: {task.tracer_code}\n")
            queue.dequeue()
        else:
            log.info(f"{agent_name} → No tasks available\n")

    log.info("")


# =========================================================================
# Demo: Item Completion Tracking
# =========================================================================

def demo_item_completion():
    """Simulate item completion tracking"""
    log.info("=" * 60)
    log.info("DEMO 7: Item Completion Tracking")
    log.info("=" * 60)

    queue = TaskQueue()

    # Populate with one order
    order = DEMO_ORDERS[0]
    for item_dict in order.items:
        item = WarehouseItemData.from_dict(item_dict)
        task = WarehouseTask.from_warehouse_item(order.o_id, item)
        queue.enqueue(task)

    completed_items = []

    # Simulate processing
    while queue.size() > 0:
        task = queue.dequeue()
        log.info(f"\nProcessing: {task.item_name}")
        log.info(f"  Quantity: {task.quantity}")

        # Simulate pickups and deliveries
        for pickup_num in range(1, task.quantity + 1):
            log.info(f"  → Pickup {pickup_num}/{task.quantity}")
            time.sleep(0.1)

        # Item complete
        completed_items.append(task.item_name)
        log.info(f"  ✓ Item : {task.item_name} has been completed by the system")

    log.info(f"\n✓ Total completed items: {len(completed_items)}")
    for item_name in completed_items:
        log.info(f"  ✓ {item_name}")

    log.info("")


# =========================================================================
# Master Demo
# =========================================================================

def run_all_demos():
    """Run all demos in sequence"""
    log.info("\n")
    log.info("╔" + "=" * 58 + "╗")
    log.info("║" + " " * 58 + "║")
    log.info("║" + "  WAREHOUSE INTEGRATION DEMO (STANDALONE)".center(58) + "║")
    log.info("║" + " " * 58 + "║")
    log.info("╚" + "=" * 58 + "╝")
    log.info("")

    try:
        demo_shelf_index_calculation()
        demo_shelf_index_decomposition()
        demo_task_creation()
        demo_queue_management()
        demo_priority_rules()
        demo_task_distribution()
        demo_item_completion()

        log.info("╔" + "=" * 58 + "╗")
        log.info("║" + " " * 58 + "║")
        log.info("║" + "  ALL DEMOS COMPLETED SUCCESSFULLY".center(58) + "║")
        log.info("║" + " " * 58 + "║")
        log.info("║" + "  Next: Run 'python integrated_dual_agent.py'".center(58) + "║")
        log.info("║" + "  to connect to Kafka and run full integration".center(58) + "║")
        log.info("║" + " " * 58 + "║")
        log.info("╚" + "=" * 58 + "╝")
        log.info("")

    except Exception as exc:
        log.error(f"✗ Demo failed: {exc}", exc_info=True)
        sys.exit(1)


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":
    run_all_demos()
