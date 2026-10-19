"""Atomic, optionally compressed checkpoints with explicit I/O diagnostics."""

from __future__ import annotations

import gzip
import io
import os
import time
from pathlib import Path

import torch


def save_torch_atomic(value, path, *, compress=False, attempts=4):
    """Serialize in memory, then publish only a flushed, complete file.

    Python file writes expose the underlying OSError instead of masking it in
    the C++ ZIP writer. Compression keeps duplicated simulator state from
    exhausting persistent storage. A failed write preserves the last checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(value, buffer)
    serialized = buffer.getbuffer()
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    for attempt in range(attempts):
        try:
            with temporary.open("wb") as output:
                if compress:
                    with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=1, mtime=0) as zipped:
                        zipped.write(serialized)
                else:
                    output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            return {"serialized_bytes": len(serialized), "stored_bytes": path.stat().st_size, "compressed": compress}
        except OSError as error:
            print(f"[checkpoint] write attempt {attempt + 1}/{attempts} failed for {path}: errno={error.errno} {error}", flush=True)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 == attempts:
                raise
            time.sleep(min(15, 2 ** attempt))


def load_trusted_training(path, *, map_location="cpu"):
    """Read our trusted full checkpoints in either original or gzip format."""
    path = Path(path)
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rb") as stream:
            payload = stream.read()
        return torch.load(io.BytesIO(payload), map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)


def write_text_atomic(text, path, *, attempts=4):
    """Publish policy configuration JSON without truncating a valid version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    for attempt in range(attempts):
        try:
            with temporary.open("w") as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            return
        except OSError as error:
            print(f"[checkpoint] JSON write attempt {attempt + 1}/{attempts} failed for {path}: errno={error.errno} {error}", flush=True)
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(15, 2 ** attempt))
