"""Check a completed train-only demonstration archive without loading it whole."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(root):
    started = time.perf_counter()
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = json.loads((root / "alns_planned_metadata.json").read_text())
    seeds = list(range(600000, 604096))
    if manifest["scenario_seeds"] != seeds or metadata["scenario_seeds"] != seeds:
        raise ValueError("teacher seed bank differs from the declared train-only bank")
    if metadata["split"] != "train" or metadata["scenarios"] != 4096 or metadata["keep_failures"]:
        raise ValueError("teacher collection split or success filtering differs")
    if metadata["episodes"] != 4096 or not 0 < metadata["successful_episodes"] <= 4096:
        raise ValueError("teacher episode accounting differs")
    for filename, expected in json.loads((root / "input_hashes.json").read_text()).items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"teacher source/config changed: {filename}")
    archive = root / "alns_planned.npz"
    if digest(archive) != metadata["archive_sha256"] or digest(root / "config.yaml") != metadata["config_sha256"]:
        raise ValueError("teacher archive/config hash differs")
    with np.load(archive) as values:
        actions = values["actions"]
        masks = values["masks"]
        if actions.shape != (metadata["transitions"],) or masks.shape != (len(actions), 62):
            raise ValueError("teacher action/mask shape differs")
        if actions.dtype.kind not in "iu" or masks.dtype != np.bool_:
            raise ValueError("teacher action/mask dtype differs")
        if np.any(actions < 0) or np.any(actions >= 62) or not masks[np.arange(len(actions)), actions].all():
            raise ValueError("teacher action is out of range or forbidden by its recorded mask")
    with zipfile.ZipFile(archive) as zipped, zipped.open("observations.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported teacher array format: {version}")
        if shape != (len(actions), 18945) or dtype != np.dtype("float32") or fortran:
            raise ValueError("teacher observations do not match the canonical 10T30D schema")
        count = 0
        while block := stream.read(8 * 1024 * 1024):
            array = np.frombuffer(block, dtype=dtype)
            if not np.isfinite(array).all():
                raise ValueError("teacher observations contain nonfinite values")
            count += array.size
        if count != int(np.prod(shape)):
            raise ValueError("teacher observation payload is incomplete")
    report = {
        "status": "completed", "split": "train", "scenarios": 4096,
        "successful_episodes": metadata["successful_episodes"], "transitions": len(actions),
        "observation_shape": shape, "all_observations_finite": True,
        "all_teacher_actions_allowed": True, "archive_sha256": metadata["archive_sha256"],
        "source_config_and_archive_hashes_verified": True,
        "validation_or_test_seeds_used": False, "seconds": time.perf_counter() - started,
    }
    (root / "dataset_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    audit(args.directory.resolve())


if __name__ == "__main__":
    main()
