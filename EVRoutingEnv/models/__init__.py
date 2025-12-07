"""Models package for the EV routing environment.

This package contains:
- core: Core entities (Truck, TransportationGraph)
- environment: Environment implementations (EventDrivenTruckEnv, CurriculumEnv, etc.)
- simulation: Simulation modules (ChargingStation, TrafficSimulator, ChargingCurve)
"""

# Import from subpackages for convenient access
from .core import Truck, TransportationGraph
from .environment import (
    EventDrivenTruckEnv,
    CurriculumEnvWrapper,
    CurriculumStrategy,
    UniformRandomStrategy,
    StagedCurriculumStrategy,
    MixedCurriculumStrategy,
    EventType,
    Event,
    EventHandler,
    create_truck,
)
from .simulation import ChargingStation, ChargingCurveModel, TrafficSimulator

__all__ = [
    # Core entities
    "Truck",
    "TransportationGraph",
    # Environments
    "EventDrivenTruckEnv",
    "CurriculumEnvWrapper",
    "CurriculumStrategy",
    "UniformRandomStrategy",
    "StagedCurriculumStrategy",
    "MixedCurriculumStrategy",
    # Event handling
    "EventType",
    "Event",
    "EventHandler",
    # Loaders
    "create_truck",
    # Simulation modules
    "ChargingStation",
    "ChargingCurveModel",
    "TrafficSimulator",
]
