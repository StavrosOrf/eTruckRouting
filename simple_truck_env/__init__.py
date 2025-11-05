"""
Simple Truck Routing Environment

A simplified single-agent environment for electric truck routing and charging.
"""

from simple_truck_env.event_driven_env import EventDrivenTruckEnv
from simple_truck_env.transportation_graph import TransportationGraph
from simple_truck_env.truck import Truck
from simple_truck_env.config_utils import (
    load_config,
    get_env_config,
    create_env_from_config,
    print_config_summary,
)

# Main environment is now the event-driven version
SimpleTruckEnv = EventDrivenTruckEnv

__all__ = [
    "EventDrivenTruckEnv",
    "SimpleTruckEnv",  # Alias for backward compatibility
    "TransportationGraph",
    "Truck",
    "load_config",
    "get_env_config",
    "create_env_from_config",
    "print_config_summary",
]
