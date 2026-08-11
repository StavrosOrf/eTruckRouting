"""Fleet-level customer tasks for joint assignment and route construction."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum


class TaskStatus(StrEnum):
    """Lifecycle of a customer task in the event-driven environment."""

    UNASSIGNED = "unassigned"
    CLAIMED = "claimed"
    IN_SERVICE = "in_service"
    SERVED = "served"


@dataclass
class CustomerTask:
    """One delivery task owned by the fleet rather than by a specific truck."""

    task_id: int
    node_id: int
    demand: float
    base_service_time: float
    earliest_service: float = 0.0
    latest_service: float = math.inf
    status: TaskStatus = TaskStatus.UNASSIGNED
    claimed_by: int | None = None
    claimed_at: float | None = None
    service_started_at: float | None = None
    served_at: float | None = None
    served_by: int | None = None

    def __post_init__(self) -> None:
        self.task_id = int(self.task_id)
        self.node_id = int(self.node_id)
        self.demand = float(self.demand)
        self.base_service_time = float(self.base_service_time)
        self.earliest_service = float(self.earliest_service)
        self.latest_service = float(self.latest_service)

        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.demand <= 0.0:
            raise ValueError("customer demand must be positive")
        if self.base_service_time < 0.0:
            raise ValueError("base service time cannot be negative")
        if self.earliest_service < 0.0:
            raise ValueError("earliest service time cannot be negative")
        if self.latest_service < self.earliest_service:
            raise ValueError("latest service time precedes earliest service time")

    @property
    def is_available(self) -> bool:
        return self.status is TaskStatus.UNASSIGNED

    @property
    def is_served(self) -> bool:
        return self.status is TaskStatus.SERVED

    def as_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        return result


class FleetTaskRegistry:
    """Atomic ownership and service state for all customers in an episode."""

    def __init__(self, tasks: Iterable[CustomerTask]):
        task_list = list(tasks)
        if not task_list:
            raise ValueError("a joint-routing instance requires at least one customer")

        self._tasks_by_id: dict[int, CustomerTask] = {}
        self._task_id_by_node: dict[int, int] = {}
        for task in task_list:
            if task.task_id in self._tasks_by_id:
                raise ValueError(f"duplicate task_id {task.task_id}")
            if task.node_id in self._task_id_by_node:
                raise ValueError(
                    f"duplicate customer node {task.node_id}; primary instances "
                    "require one task per node"
                )
            self._tasks_by_id[task.task_id] = task
            self._task_id_by_node[task.node_id] = task.task_id

    def __len__(self) -> int:
        return len(self._tasks_by_id)

    def __iter__(self):
        return iter(self.tasks())

    def tasks(self) -> list[CustomerTask]:
        return [self._tasks_by_id[key] for key in sorted(self._tasks_by_id)]

    def task_for_node(self, node_id: int) -> CustomerTask:
        try:
            task_id = self._task_id_by_node[int(node_id)]
        except KeyError as error:
            raise KeyError(f"node {node_id} is not a customer task") from error
        return self._tasks_by_id[task_id]

    def available_tasks(self, remaining_payload: float | None = None) -> list[CustomerTask]:
        tasks = [task for task in self.tasks() if task.is_available]
        if remaining_payload is None:
            return tasks
        capacity = float(remaining_payload)
        return [task for task in tasks if task.demand <= capacity]

    def pending_tasks(self) -> list[CustomerTask]:
        return [task for task in self.tasks() if not task.is_served]

    def claim(self, node_id: int, truck_id: int, timestamp: float) -> CustomerTask:
        task = self.task_for_node(node_id)
        if task.status is not TaskStatus.UNASSIGNED:
            raise ValueError(
                f"task {task.task_id} at node {node_id} is {task.status.value}, "
                "not available"
            )
        task.status = TaskStatus.CLAIMED
        task.claimed_by = int(truck_id)
        task.claimed_at = float(timestamp)
        return task

    def start_service(
        self, node_id: int, truck_id: int, timestamp: float
    ) -> CustomerTask:
        task = self.task_for_node(node_id)
        self._require_claimant(task, truck_id)
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(
                f"task {task.task_id} cannot start service from {task.status.value}"
            )
        task.status = TaskStatus.IN_SERVICE
        task.service_started_at = float(timestamp)
        return task

    def complete_service(
        self, node_id: int, truck_id: int, timestamp: float
    ) -> CustomerTask:
        task = self.task_for_node(node_id)
        self._require_claimant(task, truck_id)
        if task.status is not TaskStatus.IN_SERVICE:
            raise ValueError(
                f"task {task.task_id} cannot complete service from {task.status.value}"
            )
        task.status = TaskStatus.SERVED
        task.served_at = float(timestamp)
        task.served_by = int(truck_id)
        return task

    def release_claim(self, node_id: int, truck_id: int) -> CustomerTask:
        task = self.task_for_node(node_id)
        self._require_claimant(task, truck_id)
        if task.status is TaskStatus.SERVED:
            raise ValueError(f"served task {task.task_id} cannot be released")
        task.status = TaskStatus.UNASSIGNED
        task.claimed_by = None
        task.claimed_at = None
        task.service_started_at = None
        return task

    def all_served(self) -> bool:
        return all(task.is_served for task in self._tasks_by_id.values())

    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(
                task.status is status for task in self._tasks_by_id.values()
            )
            for status in TaskStatus
        }

    def snapshot(self) -> list[dict]:
        return [task.as_dict() for task in self.tasks()]

    @staticmethod
    def _require_claimant(task: CustomerTask, truck_id: int) -> None:
        if task.claimed_by != int(truck_id):
            raise ValueError(
                f"task {task.task_id} is claimed by {task.claimed_by}, "
                f"not truck {truck_id}"
            )
