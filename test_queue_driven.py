"""
test_queue_driven.py - Test queue-driven agent activation

This test demonstrates the queue-driven agent lifecycle:
1. Agents start at rest (idle)
2. Orders arrive from Kafka
3. Agents activate and process orders
4. After completion, agents return to rest

Run this to see the agent state transitions.
"""

import sys
import time
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("queue-driven-test")

try:
    from integrated_dual_agent import (
        WarehouseTask,
        WarehouseItemData,
        TaskQueue,
        IntegratedDualAgentSimulation,
    )
except ImportError as exc:
    log.error(f"Failed to import: {exc}")
    sys.exit(1)


def simulate_kafka_messages(task_queue: TaskQueue):
    """Simulate Kafka messages arriving with delays"""
    time.sleep(3)  # Let agents start in resting state
    
    # First order arrives
    log.info("📨 SIMULATED KAFKA MESSAGE 1 ARRIVING...")
    order_items = [
        {
            "orderTracerCode": 52,
            "itemName": "Harden",
            "itemCode": 3,
            "size": "XXL",
            "quantity": 1,
        },
        {
            "orderTracerCode": 53,
            "itemName": "Nike Air Max",
            "itemCode": 5,
            "size": "Medium",
            "quantity": 1,
        },
    ]
    
    for item_dict in order_items:
        item = WarehouseItemData.from_dict(item_dict)
        task = WarehouseTask.from_warehouse_item(order_id=38, item=item)
        task_queue.enqueue(task)
    
    time.sleep(15)  # Let agents work for a while
    
    # Second order arrives
    log.info("📨 SIMULATED KAFKA MESSAGE 2 ARRIVING...")
    order_items_2 = [
        {
            "orderTracerCode": 54,
            "itemName": "New Balance 530s",
            "itemCode": 1,
            "size": "Large",
            "quantity": 1,
        },
    ]
    
    for item_dict in order_items_2:
        item = WarehouseItemData.from_dict(item_dict)
        task = WarehouseTask.from_warehouse_item(order_id=39, item=item)
        task_queue.enqueue(task)


def main():
    """Run the queue-driven test"""
    log.info("\n")
    log.info("╔" + "=" * 58 + "╗")
    log.info("║" + " " * 58 + "║")
    log.info("║" + "  QUEUE-DRIVEN AGENT LIFECYCLE TEST".center(58) + "║")
    log.info("║" + " " * 58 + "║")
    log.info("╚" + "=" * 58 + "╝\n")

    # Create shared queue
    task_queue = TaskQueue()

    # Start Kafka simulator thread
    kafka_thread = threading.Thread(
        target=simulate_kafka_messages,
        args=(task_queue,),
        daemon=True,
    )
    kafka_thread.start()

    # Create and run simulation
    try:
        simulation = IntegratedDualAgentSimulation(
            task_queue,
            render=False,  # Disable rendering for terminal test
            model_path="checkpoints/model_episode.pth",
        )
        
        log.info("Starting simulation with queue-driven agent activation...")
        log.info("Expected sequence:")
        log.info("  1. Agents at REST (queue empty)")
        log.info("  2. 🟢 ORDERS RECEIVED - Agents ACTIVATE")
        log.info("  3. ✅ Items completed (one by one)")
        log.info("  4. 🔴 ALL DELIVERIES COMPLETE - Agents RETURN TO REST\n")
        
        simulation.run(max_steps=2000, timeout_sec=120)

    except KeyboardInterrupt:
        log.info("Test interrupted by user")
    except Exception as exc:
        log.error(f"Test failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
