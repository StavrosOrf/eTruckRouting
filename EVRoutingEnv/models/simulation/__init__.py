"""Simulation modules for charging, traffic, and other dynamic behaviors."""

from importlib import import_module


__all__ = [
    "ChargingCurveModel",
    "ChargingStation",
    "DeliverySimulator",
    "ScenarioDescriptor",
    "ScenarioRandomStreams",
    "TrafficSimulator",
]

_EXPORTS = {
    "ChargingStation": ("charging_station", "ChargingStation"),
    "ChargingCurveModel": ("charging_curve", "ChargingCurveModel"),
    "TrafficSimulator": ("traffic_simulation", "TrafficSimulator"),
    "DeliverySimulator": ("delivery_simulator", "DeliverySimulator"),
    "ScenarioDescriptor": ("scenario", "ScenarioDescriptor"),
    "ScenarioRandomStreams": ("scenario", "ScenarioRandomStreams"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value
