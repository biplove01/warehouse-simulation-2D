"""
queue_manager.py - Bridge between Task Queue and Warehouse Environment

This module manages the conversion from WarehouseTask items to environment
shelf positions and coordinates. It handles:
- Task to shelf index mapping
- Queue prioritization (first robot gets priority if only one item)
- Task completion tracking
- Integration with the environment's grid-based shelf system
"""

import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import deque

from integrated_dual_agent_final import WarehouseTask, TaskQueue, SizeEnum, compute_shelf_index

log = logging.getLogger("queue-manager")


@dataclass
class QueuedShelfTask:
    """Represents a task assigned to a specific shelf location"""
    task: WarehouseTask
    shelf_grid_x: int
    shelf_grid_y: int
    agent_assigned: Optional[str] = None  # "robot1" or "robot2"
    remaining_quantity: int = 0

    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.task.quantity


class QueueToEnvironmentBridge:
    """
    Manages the integration between TaskQueue and TwoAgentWarehouseEnv.
    
    Responsibilities:
    1. Convert shelf indices to (grid_x, grid_y) coordinates
    2. Handle task distribution to agents
    3. Track task completion
    4. Manage queue priorities
    """

    def __init__(self, task_queue: TaskQueue, shelves: List):
        """
        Args:
            task_queue: The shared WarehouseTask queue
            shelves: List of shelf objects from environment (has .x, .y properties)
        """
        self.task_queue = task_queue
        self.shelves = shelves  # All shelves in environment
        self.current_task: Optional[QueuedShelfTask] = None
        self.completed_tasks: Dict[int, WarehouseTask] = {}  # tracer_code -> task
        self.in_progress: List[QueuedShelfTask] = []

        # Build shelf index to grid coordinate mapping
        self._shelf_to_coords = self._build_shelf_coords_map()

    def _build_shelf_coords_map(self) -> Dict[int, Tuple[int, int]]:
        """
        Create mapping from shelf indices (0-49) to grid coordinates.
        
        Formula: shelf_index = (item_code - 1) * 5 + size_index
        Items 1-10, sizes 0-4 → indices 0-49
        """
        from constants import PADDING_BORDER, GRID_SPACING

        coords_map = {}
        for idx, shelf in enumerate(self.shelves):
            if idx >= 50:  # Only map the first 50 shelves (10 items × 5 sizes)
                break
            gx = round((shelf.x - PADDING_BORDER) / GRID_SPACING)
            gy = round((shelf.y - PADDING_BORDER) / GRID_SPACING)
            coords_map[idx] = (gx, gy)
        return coords_map

    def get_shelf_coordinates(self, shelf_index: int) -> Tuple[int, int]:
        """
        Get grid (x, y) for a shelf index (0-49).
        
        Args:
            shelf_index: Index from 0 to 49
            
        Returns:
            (grid_x, grid_y) tuple
        """
        if shelf_index not in self._shelf_to_coords:
            raise ValueError(f"Invalid shelf index: {shelf_index}")
        return self._shelf_to_coords[shelf_index]

    def dispatch_next_task(self, agent_name: str, only_one_priority: bool = True) -> Optional[QueuedShelfTask]:
        """
        Dispatch the next task from queue to a specific agent.
        
        Rules:
        - If only one item in queue AND another agent is idle, give to first robot
        - Otherwise, assign greedily to requesting agent
        
        Args:
            agent_name: "robot1" or "robot2"
            only_one_priority: If True, apply priority rule for single items
            
        Returns:
            QueuedShelfTask or None if queue is empty
        """
        queue_size = self.task_queue.size()
        
        if queue_size == 0:
            return None
        
        # Apply priority rule
        if only_one_priority and queue_size == 1 and agent_name == "robot2":
            log.info(
                "Single item in queue and robot2 requesting. "
                "Priority given to robot1. Deferring robot2 request."
            )
            return None
        
        task = self.task_queue.dequeue()
        if not task:
            return None
        
        # Create queued shelf task
        shelf_coords = self.get_shelf_coordinates(task.shelf_index)
        queued_task = QueuedShelfTask(
            task=task,
            shelf_grid_x=shelf_coords[0],
            shelf_grid_y=shelf_coords[1],
            agent_assigned=agent_name,
            remaining_quantity=task.quantity,
        )
        
        self.in_progress.append(queued_task)
        
        log.info(
            f"Dispatched to {agent_name}: {task.item_name} "
            f"(code={task.item_code}, size={task.size}, qty={task.quantity}) "
            f"→ shelf_idx={task.shelf_index} (grid={shelf_coords})"
        )
        
        return queued_task

    def mark_item_pickup(self, queued_task: QueuedShelfTask) -> None:
        """Mark that an item has been picked up from the shelf"""
        log.debug(
            f"Picked up item from {queued_task.task.item_name} "
            f"({queued_task.remaining_quantity} remaining)"
        )

    def mark_item_delivery(self, queued_task: QueuedShelfTask) -> bool:
        """
        Mark item delivery. If all quantities delivered, task is complete.
        
        Returns:
            True if task is fully complete, False otherwise
        """
        if queued_task.remaining_quantity <= 0:
            return False
        
        queued_task.remaining_quantity -= 1
        
        if queued_task.remaining_quantity == 0:
            self._complete_task(queued_task)
            return True
        
        return False

    def _complete_task(self, queued_task: QueuedShelfTask) -> None:
        """Handle task completion with terminal output"""
        task = queued_task.task
        self.completed_tasks[task.tracer_code] = task
        self.in_progress.remove(queued_task)
        
        # Print completion message to terminal
        completion_msg = f"✅ Item : {task.item_name} has been completed by the system"
        print(f"\n{completion_msg}\n")
        
        log.info(
            f"✓ Item: {task.item_name} has been completed by the system "
            f"(order_id={task.order_id}, agent={queued_task.agent_assigned})"
        )

    def get_completion_message(self, queued_task: QueuedShelfTask) -> str:
        """Get formatted completion message for an item"""
        return f"Item : {queued_task.task.item_name} has been completed by the system"

    def has_pending_work(self) -> bool:
        """
        Check if there are any tasks in queue or currently in progress.
        
        Returns:
            True if work is pending, False if all done
        """
        return self.task_queue.size() > 0 or len(self.in_progress) > 0

    def get_queue_status(self) -> Dict:
        """Get current queue statistics"""
        return {
            "queue_size": self.task_queue.size(),
            "in_progress": len(self.in_progress),
            "completed": len(self.completed_tasks),
            "total_items_in_progress": sum(t.remaining_quantity for t in self.in_progress),
        }

    def get_progress_summary(self) -> str:
        """Get human-readable progress summary"""
        status = self.get_queue_status()
        return (
            f"Queue Status: "
            f"Queued={status['queue_size']}, "
            f"InProgress={status['in_progress']} "
            f"(items={status['total_items_in_progress']}), "
            f"Completed={status['completed']}"
        )


# =========================================================================
# Utility Functions
# =========================================================================

def decompose_shelf_index(shelf_index: int) -> Tuple[int, str]:
    """
    Reverse the shelf index back to (item_code, size).
    
    Args:
        shelf_index: 0-49
        
    Returns:
        (item_code, size_string) tuple
    """
    if not 0 <= shelf_index < 50:
        raise ValueError(f"Invalid shelf index: {shelf_index}")
    
    item_code = (shelf_index // 5) + 1
    size_index = shelf_index % 5
    
    # Reverse lookup in SizeEnum
    for size in SizeEnum:
        if size.value[1] == size_index:
            return item_code, size.value[0]
    
    raise ValueError(f"Could not decompose shelf index: {shelf_index}")


# =========================================================================
# Example Usage (for testing)
# =========================================================================

if __name__ == "__main__":
    # Test shelf index computation
    print("Testing shelf index computation (10 items × 5 sizes = 50 shelves):")
    print(f"Item 1, small → index {compute_shelf_index(1, 'small')}")    # Should be 0
    print(f"Item 1, xxl → index {compute_shelf_index(1, 'xxl')}")        # Should be 4
    print(f"Item 2, small → index {compute_shelf_index(2, 'small')}")    # Should be 5
    print(f"Item 10, xxl → index {compute_shelf_index(10, 'xxl')}")      # Should be 49
    
    print("\nTesting shelf index decomposition:")
    for idx in [0, 4, 5, 49]:
        item_code, size = decompose_shelf_index(idx)
        print(f"Index {idx} → Item {item_code}, {size}")
