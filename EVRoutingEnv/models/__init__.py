"""Models package for the EV routing environment.

This package contains:
- core: Core entities (Truck, TransportationGraph)
- environment: Environment implementations (EventDrivenTruckEnv, CurriculumEnv, etc.)
- simulation: Simulation modules (ChargingStation, TrafficSimulator, ChargingCurve)
"""

from importlib import import_module


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
    "CustomerTask",
    "FleetTaskRegistry",
    "TaskStatus",
]


_EXPORTS = {
    "Truck": ("EVRoutingEnv.models.core", "Truck"),
    "TransportationGraph": ("EVRoutingEnv.models.core", "TransportationGraph"),
    "EventDrivenTruckEnv": (
        "EVRoutingEnv.models.environment",
        "EventDrivenTruckEnv",
    ),
    "CurriculumEnvWrapper": (
        "EVRoutingEnv.models.environment",
        "CurriculumEnvWrapper",
    ),
    "CurriculumStrategy": (
        "EVRoutingEnv.models.environment",
        "CurriculumStrategy",
    ),
    "UniformRandomStrategy": (
        "EVRoutingEnv.models.environment",
        "UniformRandomStrategy",
    ),
    "StagedCurriculumStrategy": (
        "EVRoutingEnv.models.environment",
        "StagedCurriculumStrategy",
    ),
    "MixedCurriculumStrategy": (
        "EVRoutingEnv.models.environment",
        "MixedCurriculumStrategy",
    ),
    "EventType": ("EVRoutingEnv.models.environment", "EventType"),
    "Event": ("EVRoutingEnv.models.environment", "Event"),
    "EventHandler": ("EVRoutingEnv.models.environment", "EventHandler"),
    "create_truck": ("EVRoutingEnv.models.environment", "create_truck"),
    "ChargingStation": ("EVRoutingEnv.models.simulation", "ChargingStation"),
    "ChargingCurveModel": (
        "EVRoutingEnv.models.simulation",
        "ChargingCurveModel",
    ),
    "TrafficSimulator": ("EVRoutingEnv.models.simulation", "TrafficSimulator"),
    "CustomerTask": ("EVRoutingEnv.models.core", "CustomerTask"),
    "FleetTaskRegistry": ("EVRoutingEnv.models.core", "FleetTaskRegistry"),
    "TaskStatus": ("EVRoutingEnv.models.core", "TaskStatus"),
}


def __getattr__(name):
    """Load convenience exports lazily to keep submodules independently usable."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
