"""
Lightweight YAML parser implementing a minimal subset required by the EVPR configs.
Supports indentation-based dictionaries with scalar values and inline lists.
"""

import re
from typing import Any, Dict, List


def _strip_comments(line: str) -> str:
    in_string = False
    quote_char = None
    result = []
    for ch in line:
        if ch in ("'", '"'):
            if not in_string:
                in_string = True
                quote_char = ch
            elif ch == quote_char:
                in_string = False
        if ch == "#" and not in_string:
            break
        result.append(ch)
    return "".join(result)


def _parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [part.strip() for part in inner.split(",")]
        return [_parse_value(part) for part in parts]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


class Dumper:
    pass


class SafeDumper(Dumper):
    pass


class Loader:
    pass


class SafeLoader(Loader):
    pass


def safe_load(stream) -> Dict[str, Any]:
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = stream

    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = _strip_comments(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not value:
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _parse_value(value)

    return root


def load(stream):
    return safe_load(stream)


def dump(data, stream=None, Dumper=Dumper):
    import json
    text = json.dumps(data, indent=2)
    if stream is None:
        return text
    stream.write(text)
    return text


def safe_dump(data, stream=None, **kwargs):
    return dump(data, stream, Dumper=SafeDumper)


__all__ = ["safe_load", "load"]
