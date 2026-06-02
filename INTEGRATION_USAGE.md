"""
INTEGRATION_USAGE.md - Complete Integration Guide

This guide explains how to use the Kafka-integrated dual-agent warehouse simulation.

## File Overview

### Core Files
- **integrated_dual_agent.py**: Main integration module
  - OrderKafkaListener: Listens to Kafka and feeds tasks to queue
  - WarehouseTask: Represents a single warehouse item
  - TaskQueue: Thread-safe queue for tasks
  - IntegratedDualAgentSimulation: Main simulation orchestrator

- **queue_manager.py**: Queue-to-environment bridge
  - QueueToEnvironmentBridge: Manages task distribution to agents
  - Utility functions for shelf index ↔ (item_code, size) conversion

- **reader_enhanced.py**: Enhanced Kafka consumer
  - OrderKafkaReaderEnhanced: Can work standalone or with callbacks
  - Mirrors original reader.py DTOs

- **test_two_agents.py**: PRESERVED - Original dual-agent test (untouched)

## System Architecture

```
Java Backend (Kafka Publisher)
    ↓
    ├→ Kafka Topic: "publish-event"
    │
    └→ OrderKafkaListener (reader_enhanced.py)
           ↓
        WarehouseData JSON
           ↓
        WarehouseTask objects
           ↓
        TaskQueue (thread-safe deque)
           ↓
    Queue-Driven Agent Activation:
    ├─ If Queue Empty → Agents IDLE at home
    ├─ If Queue has items → Agents ACTIVATE and work
    └─ After queue empty & tasks done → Agents return to REST
           ↓
        QueueToEnvironmentBridge (queue_manager.py)
           ├→ shelf_index → (grid_x, grid_y)
           ├→ task distribution logic
           └→ completion tracking with terminal output
           ↓
        TwoAgentWarehouseEnv (unified_env.py)
           ├→ Robot1 (Q-table based)
           ├→ Robot2 (DQN-based learner)
           └→ Pygame visualization
```

## Agent Activation Model

**Agents follow a queue-driven lifecycle:**

1. **RESTING STATE** 
   - Agents idle at home station
   - No Kafka messages in queue
   - Minimal CPU usage

2. **ACTIVATION TRIGGER**
   - Kafka message arrives
   - Task added to queue
   - Agents wake up and start working
   - Terminal: "🟢 ORDERS RECEIVED FROM KAFKA - STARTING DELIVERY"

3. **WORKING STATE**
   - Agents process tasks from queue
   - Robot1 & Robot2 coordinate independently
   - Each item completion prints to terminal:
     - "✅ Item : [name] has been completed by the system"

4. **RETURN TO REST**
   - Queue becomes empty
   - All tasks finished
   - Agents navigate back to home station
   - Terminal: "🔴 ALL DELIVERIES COMPLETE - AGENTS RETURNING TO REST"
   - Return to RESTING STATE

## Shelf Index Formula

The warehouse has 120 positions (24 items × 5 sizes):

    shelf_index = (item_code - 1) × 5 + size_index

Size indices:
- 0: small
- 1: medium
- 2: large
- 3: xl
- 4: xxl

Examples:
- Item 1, small    → shelf_index = 0
- Item 1, xxl      → shelf_index = 4
- Item 2, small    → shelf_index = 5
- Item 24, xxl     → shelf_index = 119

## Usage Scenarios

### Scenario 1: Run Full Integration with Kafka

```bash
# Ensure Kafka is running on localhost:9092
# Topic "publish-event" exists
# Ecommerce Java backend is publishing messages

python integrated_dual_agent.py
```

What happens:
1. **Initial State**: Agents idle at home (no console output)
2. **Order Arrives**: Kafka message received, agents activate
   - Terminal: "🟢 ORDERS RECEIVED FROM KAFKA - STARTING DELIVERY"
3. **Processing**: For each item completed:
   - Terminal: "✅ Item : [item_name] has been completed by the system"
4. **Queue Empty**: When all items delivered
   - Terminal: "🔴 ALL DELIVERIES COMPLETE - AGENTS RETURNING TO REST"
5. **Final Summary**: Shows all completed items
   - Agents return to home and rest

### Scenario 2: Test Without Kafka (Demo Mode)

See `demo_standalone.py` for an example that simulates warehouse orders
without requiring Kafka connectivity.

### Scenario 3: Use Original Test Suite

The original test_two_agents.py is PRESERVED and unchanged. Run it directly:

```bash
python test_two_agents.py
```

## API Usage Examples

### Example 1: Create and Enqueue a Task

```python
from integrated_dual_agent import WarehouseTask, TaskQueue, WarehouseItemData

# Create a task from warehouse item
item = WarehouseItemData(
    order_tracer_code=52,
    item_name="Harden",
    item_code=3,
    size="XXL",
    quantity=4
)

task = WarehouseTask.from_warehouse_item(order_id=38, item=item)
print(task)
# Output: Task(oId=38, code=3, name='Harden', size=XXL, qty=4, shelf_idx=14)

# Add to queue
queue = TaskQueue()
queue.enqueue(task)
```

### Example 2: Compute Shelf Index

```python
from queue_manager import compute_shelf_index, decompose_shelf_index

# Forward: (item_code, size) → shelf_index
idx = compute_shelf_index(item_code=3, size="XXL")
print(idx)  # Output: 14

# Reverse: shelf_index → (item_code, size)
code, sz = decompose_shelf_index(14)
print(f"Item {code}, {sz}")  # Output: Item 3, xxl
```

### Example 3: Use QueueToEnvironmentBridge

```python
from queue_manager import QueueToEnvironmentBridge
from integrated_dual_agent import TaskQueue

queue = TaskQueue()
# ... add tasks to queue ...

bridge = QueueToEnvironmentBridge(queue, shelves_from_environment)

# Dispatch to robot1
queued_task = bridge.dispatch_next_task("robot1")
if queued_task:
    print(f"Robot1 assigned: {queued_task.task.item_name}")
    
# After robot picks up
bridge.mark_item_pickup(queued_task)

# After robot delivers
is_complete = bridge.mark_item_delivery(queued_task)
if is_complete:
    print(bridge.get_completion_message(queued_task))

# Check progress
print(bridge.get_progress_summary())
```

### Example 4: Custom Kafka Callback

```python
from reader_enhanced import OrderKafkaReaderEnhanced, KafkaConfig
from integrated_dual_agent import TaskQueue

queue = TaskQueue()

def on_warehouse_data(warehouse_data):
    \"\"\"Custom callback when Kafka message arrives\"\"\"
    for item in warehouse_data.items:
        task = WarehouseTask.from_warehouse_item(warehouse_data.o_id, item)
        queue.enqueue(task)
    print(f"Enqueued {len(warehouse_data.items)} items from order {warehouse_data.o_id}")

config = KafkaConfig()
reader = OrderKafkaReaderEnhanced(config, on_message_callback=on_warehouse_data)
reader.start()

# Later...
print(f"Total messages processed: {reader.get_message_count()}")
reader.stop()
```

## Terminal Output & Agent Status

The simulation provides clear terminal feedback about agent activity:

### Resting State (No Orders)
```
[INFO] Starting integrated dual-agent simulation (queue-driven)
(No additional output - agents idle, waiting for Kafka messages)
```

### Order Received (Activation)
```
============================================================
🟢 ORDERS RECEIVED FROM KAFKA - STARTING DELIVERY
============================================================
```

### Item Completion (During Processing)
```
✅ Item : Harden has been completed by the system

✅ Item : New Balance 530s has been completed by the system
```

### All Deliveries Complete (Return to Rest)
```
============================================================
🔴 ALL DELIVERIES COMPLETE - AGENTS RETURNING TO REST
============================================================
```

### Final Summary
```
============================================================
📊 FINAL SUMMARY
============================================================

✅ Total items completed: 3

   ✓ Harden
   ✓ New Balance 530s
   ✓ Random Item

🟣 AGENTS NOW AT REST

============================================================
```

## Configuration

### Environment Variables

```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_TOPIC=publish-event
export KAFKA_GROUP_ID=order-reader-group
```

### Kafka Message Format

The integration expects messages in this exact JSON format:

```json
{
  "oId": 38,
  "items": [
    {
      "orderTracerCode": 52,
      "itemName": "Harden",
      "itemCode": 3,
      "size": "XXL",
      "quantity": 4
    },
    {
      "orderTracerCode": 53,
      "itemName": "Random Item",
      "itemCode": 4,
      "size": "Medium",
      "quantity": 3
    }
  ]
}
```

## Priority Rules

1. **Single Item in Queue**: Priority given to Robot1
   - If queue.size() == 1 and Robot2 requests, Robot2 is deferred
   - Robot1 gets first pick

2. **Multiple Items in Queue**:
   - Greedy assignment - first available robot gets next item
   - No preference between robots

3. **Task Completion**:
   - When all quantities for an item are delivered, task is marked complete
   - Completion message printed

## Thread Safety

- TaskQueue uses locks for thread-safe access
- Kafka listener runs in background thread
- Environment runs in main thread
- No race conditions expected

## Debugging

Enable debug logging:

```python
import logging
logging.getLogger("integrated-dual-agent").setLevel(logging.DEBUG)
logging.getLogger("queue-manager").setLevel(logging.DEBUG)
```

## Troubleshooting

### "Kafka connection failed"
- Ensure Kafka broker is running on localhost:9092
- Check environment variables (KAFKA_BOOTSTRAP_SERVERS)
- Verify topic "publish-event" exists

### "Model not found"
- Run training first: `python train_unified_agent2.py`
- Or specify custom model path: `model_path="path/to/model.pth"`

### "Shelves not properly mapped"
- Ensure shelves list has exactly 120 items
- Check shelf ordering matches expected grid layout

## Next Steps

1. Run `demo_standalone.py` to test without Kafka
2. Verify shelf index calculation with `queue_manager.py` test
3. Set up Kafka with ecommerce backend
4. Run `integrated_dual_agent.py` for full integration
5. Monitor console output for completion messages

## Integration Checklist

- [ ] Kafka broker running on localhost:9092
- [ ] Topic "publish-event" created
- [ ] Java ecommerce backend configured to publish to Kafka
- [ ] Python environment has required packages (see requirements.txt)
- [ ] PyTorch model trained (checkpoints/model_episode.pth exists)
- [ ] Run `demo_standalone.py` first for validation
- [ ] Run `integrated_dual_agent.py` for full integration
- [ ] Monitor logs for "Item: ... has been completed" messages
"""
