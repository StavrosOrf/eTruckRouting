"""Core entities for the EV routing environment."""

from .customer import CustomerTask, FleetTaskRegistry, TaskStatus
from .truck import Truck
from .transportation_graph import TransportationGraph

__all__ = [
    "CustomerTask",
    "FleetTaskRegistry",
    "TaskStatus",
    "TransportationGraph",
    "Truck",
]
