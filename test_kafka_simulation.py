"""
test_kafka_simulation.py - Test Kafka integration without real Kafka broker

This test:
1. Creates mock Kafka messages with warehouse items
2. Simulates them arriving in the task queue
3. Shows agents processing and completing items
4. Prints all messages to terminal

Run this to see the system working WITHOUT needing a real Kafka broker!
"""

import sys
import time
import threading
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("test-kafka-sim")

try:
    from integrated_dual_agent_final import (
        WarehouseTask,
        WarehouseItemData,
        TaskQueue,
        FinalWarehouseSimulation,
    )
except ImportError as exc:
    log.error(f"Import failed: {exc}")
    sys.exit(1)


def simulate_kafka_orders(task_queue: TaskQueue):
    """Simulate Kafka messages arriving with delays"""
    time.sleep(3)

    # First batch of orders
    log.info("📨 SIMULATING FIRST KAFKA BATCH...")
    orders_batch1 = [
        {
            "orderTracerCode": 52,
            "itemName": "Harden Basketball Shoes",
            "itemCode": 3,
            "size": "XXL",
            "quantity": 1,
        },
        {
            "orderTracerCode": 53,
            "itemName": "Nike Air Max 90",
            "itemCode": 5,
            "size": "Medium",
            "quantity": 1,
        },
        {
            "orderTracerCode": 54,
            "itemName": "Adidas Ultra Boost",
            "itemCode": 1,
            "size": "Large",
            "quantity": 1,
        },
    ]

    for item_dict in orders_batch1:
        item = WarehouseItemData(
            order_tracer_code=item_dict["orderTracerCode"],
            item_name=item_dict["itemName"],
            item_code=item_dict["itemCode"],
            size=item_dict["size"],
            quantity=item_dict["quantity"],
        )
        task = WarehouseTask.from_warehouse_item(order_id=100, item=item)
        task_queue.enqueue(task)

    log.info(f"✅ Batch 1 enqueued: {len(orders_batch1)} items")

    # Wait a bit
    time.sleep(20)

    # Second batch
    log.info("📨 SIMULATING SECOND KAFKA BATCH...")
    orders_batch2 = [
        {
            "orderTracerCode": 55,
            "itemName": "Puma Running Shoes",
            "itemCode": 7,
            "size": "Small",
            "quantity": 1,
        },
        {
            "orderTracerCode": 56,
            "itemName": "New Balance 530s",
            "itemCode": 2,
            "size": "XL",
            "quantity": 1,
        },
    ]

    for item_dict in orders_batch2:
        item = WarehouseItemData(
            order_tracer_code=item_dict["orderTracerCode"],
            item_name=item_dict["itemName"],
            item_code=item_dict["itemCode"],
            size=item_dict["size"],
            quantity=item_dict["quantity"],
        )
        task = WarehouseTask.from_warehouse_item(order_id=101, item=item)
        task_queue.enqueue(task)

    log.info(f"✅ Batch 2 enqueued: {len(orders_batch2)} items")


def main():
    """Run the test"""
    log.info("\n" + "="*70)
    log.info("KAFKA SIMULATION TEST - Queue-Driven Agents")
    log.info("="*70)
    log.info("This test simulates Kafka messages WITHOUT needing a broker")
    log.info("="*70 + "\n")

    # Create queue
    task_queue = TaskQueue()

    # Start simulated Kafka in background
    kafka_thread = threading.Thread(
        target=simulate_kafka_orders,
        args=(task_queue,),
        daemon=True,
    )
    kafka_thread.start()

    # Create and run simulation
    try:
        simulation = FinalWarehouseSimulation(
            task_queue,
            render=False,  # No rendering for terminal test
            model_path="checkpoints/model_episode.pth",
        )

        log.info("🚀 Starting simulation (agents will be idle until orders arrive)...\n")
        simulation.run(max_steps=1000, timeout_sec=120)

    except KeyboardInterrupt:
        log.info("Interrupted")
        sys.exit(0)
    except Exception as exc:
        log.error(f"Test failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
