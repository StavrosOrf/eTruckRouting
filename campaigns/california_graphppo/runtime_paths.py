"""Relocate filesystem references in trusted saved simulator state only.

Numerical arrays, tensors, RNG streams and optimizer/policy state are untouched.
This runs after trusted checkpoint deserialization, before environment restore.
"""

from pathlib import Path


def relocate_environments(state, mappings):
    seen = {}

    def visit(value):
        if isinstance(value, str):
            for old, new in mappings:
                if value == old or value.startswith(old + "/"):
                    return new + value[len(old):]
            return value
        if isinstance(value, Path):
            return type(value)(visit(str(value)))
        if value is None or isinstance(value, (int, float, bool, bytes)):
            return value
        module = type(value).__module__
        if module.startswith(("numpy", "torch")):
            return value
        if id(value) in seen:
            return seen[id(value)]
        seen[id(value)] = value
        if isinstance(value, dict):
            for key, child in list(value.items()):
                new_key = visit(key)
                new_value = visit(child)
                if new_key != key:
                    del value[key]
                value[new_key] = new_value
        elif isinstance(value, list):
            for index, child in enumerate(value):
                value[index] = visit(child)
        elif isinstance(value, tuple):
            children = [visit(child) for child in value]
            value = type(value)(*children) if hasattr(value, "_fields") else tuple(children)
            seen[id(value)] = value
        elif module.startswith(("EVRoutingEnv", "algo")) and hasattr(value, "__dict__"):
            for key, child in list(vars(value).items()):
                vars(value)[key] = visit(child)
        return value

    if "environments" in state:
        state["environments"] = visit(state["environments"])
    return state
