# Warehouse Automation Simulation - Kafka Integration

Complete integration of Kafka-driven dual-agent warehouse simulation using Java backend + Python RL simulation with DQN.

## 📋 Project Structure

### Core Integration Files (NEW)

| File | Purpose |
|------|---------|
| `integrated_dual_agent.py` | Main integration orchestrator - connects Kafka → task queue → dual agents |
| `queue_manager.py` | Queue-to-environment bridge with shelf index calculations |
| `reader_enhanced.py` | Enhanced Kafka consumer with optional queue integration |
| `demo_standalone.py` | Standalone demo (no Kafka required) - validates integration |

### Original Files (PRESERVED)

| File | Purpose |
|------|---------|
| `test_two_agents.py` | Original dual-agent test suite (UNCHANGED) |
| `unified_env.py` | Dual-agent warehouse environment |
| `reader.py` | Original Kafka reader (UNCHANGED) |
| Other files | Core robot, environment, and training modules |

### Documentation

| File | Purpose |
|------|---------|
| `INTEGRATION_USAGE.md` | Complete usage guide with API examples |
| `INTEGRATION_README.md` | This file |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ecommerce Website                        │
│              (Java Backend via Kafka)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Kafka Topic: "publish-event"                      │
│         Message: {"oId": 38, "items": [...]}               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          OrderKafkaListener (reader_enhanced.py)            │
│         • Deserializes WarehouseData from JSON             │
│         • Creates WarehouseTask objects                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            TaskQueue (thread-safe deque)                    │
│         • Stores all warehouse tasks                       │
│         • Thread-safe enqueue/dequeue                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      QueueToEnvironmentBridge (queue_manager.py)            │
│      • Shelf Index ↔ (item_code, size) conversion         │
│      • Task distribution to agents                        │
│      • Completion tracking & priority rules               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         TwoAgentWarehouseEnv (unified_env.py)              │
│    ┌──────────────────┬─────────────────────────────┐     │
│    │                  │                             │     │
│    ▼                  ▼                             ▼     │
│  Robot1         Robot2 (DQN Learner)          Pygame     │
│  (Q-Table)      • Reads tasks from queue      Rendering  │
│  • Follows BFS  • Executes DQN actions       (Visual)    │
│  • Picks items  • Delivers to dropoff                    │
│  • Delivers                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Completion Log  │
            │                 │
            │ "Item : [name]  │
            │ has been        │
            │ completed by    │
            │ the system"     │
            └─────────────────┘
```

## 🔢 Shelf Index Formula

The warehouse has **120 total shelf positions** (24 items × 5 sizes).

**Formula:** `shelf_index = (item_code - 1) × 5 + size_index`

### Size Index Mapping
- 0 = small
- 1 = medium
- 2 = large
- 3 = xl
- 4 = xxl

### Examples
| Item Code | Size | Shelf Index |
|-----------|------|-------------|
| 1 | small | 0 |
| 1 | xxl | 4 |
| 2 | small | 5 |
| 24 | xxl | 119 |

## 🚀 Quick Start

### Prerequisites

```bash
# Python packages
pip install -r requirements.txt

# Kafka (if using full integration)
# Ensure Kafka broker is running on localhost:9092
# Topic "publish-event" must exist
```

### Step 1: Validate Integration (No Kafka Required)

```bash
python demo_standalone.py
```

Expected output:
- ✓ Shelf index calculations validated
- ✓ Tasks created and queued
- ✓ Priority rules tested
- ✓ Item completion tracked

### Step 2: Run Full Integration with Kafka

```bash
# Terminal 1: Start Kafka consumer + simulation
python integrated_dual_agent.py

# Terminal 2: Send test orders (from Java backend)
# or manually publish messages to "publish-event" topic
```

Expected behavior:
1. OrderKafkaListener connects to Kafka
2. Awaits messages from ecommerce backend
3. For each message:
   - Deserializes WarehouseData
   - Creates tasks for each item
   - Enqueues to TaskQueue
4. Robots process queue items
5. Completion messages printed

### Step 3: Verify Original Tests Still Work

```bash
# Original test suite (UNCHANGED)
python test_two_agents.py
```

## 💾 WarehouseTask API

### Creating a Task

```python
from integrated_dual_agent import WarehouseTask, WarehouseItemData

item = WarehouseItemData(
    order_tracer_code=52,
    item_name="Harden",
    item_code=3,
    size="XXL",
    quantity=4
)

task = WarehouseTask.from_warehouse_item(order_id=38, item=item)
```

### Task Properties

```python
task.order_id            # Order ID from Kafka
task.tracer_code         # Item tracer code
task.item_name           # Item name
task.item_code           # 1-24
task.size                # "small", "medium", "large", "xl", "xxl"
task.quantity            # Number of items to pick
task.shelf_index         # Computed 0-119
```

## 🔄 Priority Rules

### Single Item in Queue
- If `queue.size() == 1` and Robot2 requests, **Robot2 is deferred**
- **Robot1 gets priority** for single items

### Multiple Items in Queue
- **Greedy assignment**: first available robot gets next item
- No preference between robots

## 🧵 Thread Safety

- `TaskQueue` uses locks for thread-safe access
- Kafka listener runs in background daemon thread
- Environment runs in main thread
- No race conditions

## 📊 Queue Manager Functions

### Compute Shelf Index
```python
from queue_manager import compute_shelf_index

idx = compute_shelf_index(item_code=3, size="XXL")
# Returns: 14
```

### Decompose Shelf Index
```python
from queue_manager import decompose_shelf_index

code, size = decompose_shelf_index(14)
# Returns: (3, "xxl")
```

## 🔧 Configuration

### Environment Variables

```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_TOPIC=publish-event
export KAFKA_GROUP_ID=order-reader-group
```

### Kafka Message Format

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

## 📋 Completion Message Format

When an item is fully completed:

```
✓ Item : [item_name] has been completed by the system
```

Example:
```
✓ Item : Harden has been completed by the system
✓ Item : New Balance 530s has been completed by the system
```

## 🐛 Debugging

Enable debug logging:

```python
import logging
logging.getLogger("integrated-dual-agent").setLevel(logging.DEBUG)
logging.getLogger("queue-manager").setLevel(logging.DEBUG)
logging.getLogger("order-reader-enhanced").setLevel(logging.DEBUG)
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Kafka connection failed | Ensure Kafka broker running on localhost:9092 |
| Topic not found | Create topic: `kafka-topics --create --topic publish-event` |
| Model not found | Run `python train_unified_agent2.py` first |
| Shelves not mapped | Ensure 120 shelf objects in environment |
| Demo shows errors | Check Python imports and torch installation |

## 📚 File Guide

### integrated_dual_agent.py
- `SizeEnum`: Maps size strings to indices
- `WarehouseTask`: Represents a single warehouse item
- `TaskQueue`: Thread-safe queue for tasks
- `KafkaOrderListener`: Kafka consumer integration
- `IntegratedDualAgentSimulation`: Main orchestrator

### queue_manager.py
- `QueueToEnvironmentBridge`: Maps tasks to shelf coordinates
- `QueuedShelfTask`: Task with agent assignment
- `compute_shelf_index()`: (item_code, size) → index
- `decompose_shelf_index()`: index → (item_code, size)

### reader_enhanced.py
- `OrderKafkaReaderEnhanced`: Kafka consumer with callbacks
- `WarehouseData`: Order data structure
- `WarehouseItemData`: Individual item structure
- `KafkaConfig`: Configuration class

### demo_standalone.py
- 7 demonstration scenarios
- No Kafka required
- Validates shelf calculations
- Tests task creation and distribution
- Simulates completion tracking

## ✅ Integration Checklist

- [x] Shelf index formula implemented (item_code × 5 + size_index)
- [x] WarehouseTask and TaskQueue created
- [x] Kafka integration with OrderKafkaListener
- [x] Queue-to-environment bridge (QueueToEnvironmentBridge)
- [x] Priority rules (single item → Robot1)
- [x] Completion message formatting
- [x] Thread safety with locks
- [x] Standalone demo (no Kafka required)
- [x] Original test_two_agents.py preserved
- [x] Comprehensive documentation

## 🎯 Next Steps

1. **Validate**: Run `python demo_standalone.py`
2. **Setup Kafka**: Ensure broker and topic exist
3. **Integrate**: Run `python integrated_dual_agent.py`
4. **Monitor**: Watch console for completion messages
5. **Extend**: Customize behavior in queue_manager.py

## 📝 Notes

- All 120 shelf positions are addressable via shelf_index (0-119)
- Each robot can independently pick and deliver items
- Kafka messages are deserialized into typed WarehouseTask objects
- Completion is tracked per item with tracer codes
- Environment visualization shows real-time robot movements

## 📞 Support

For issues or questions:
1. Check INTEGRATION_USAGE.md for API examples
2. Run demo_standalone.py to validate setup
3. Enable debug logging for detailed trace
4. Verify Kafka connectivity and message format

---

**Integration Complete!** Your warehouse simulation is now ready for Kafka-driven orders.
