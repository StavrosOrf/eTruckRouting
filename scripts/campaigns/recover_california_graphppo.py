"""Prepare minimal I/O recovery sources and inspect actual saved training state."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


COHORTS = ("ca10t3d_v1", "ca10t3d_progress_v3", "ca10t3d_imitation_v5", "ca10t3d_capacity_v7", "ca10t3d_objective_v8", "ca10t3d_retention_v9")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(workspace, root):
    if root.exists():
        raise FileExistsError("inspect existing recovery before preparing another")
    root.mkdir(parents=True)
    base = workspace / "results/california_graphppo"
    manifest = {"workspace": str(workspace), "cohorts": {}, "goal_achieved": False}
    old_save = '        temporary = path.with_suffix(path.suffix + ".tmp")\n        torch.save(state, temporary)\n        temporary.replace(path)'
    new_save = '        from algo.checkpoint_io import save_torch_atomic\n\n        save_torch_atomic(state, path, compress=True)'
    old_load = '        state = torch.load(path, map_location="cpu", weights_only=False)'
    new_load = '        from algo.checkpoint_io import load_trusted_training\n\n        state = load_trusted_training(path, map_location="cpu")'
    old_policy = '        torch.save(self.state_dict(), directory / f"{prefix}.pt")'
    new_policy = '        from algo.checkpoint_io import save_torch_atomic\n\n        save_torch_atomic(self.state_dict(), directory / f"{prefix}.pt")'
    for cohort in COHORTS:
        campaign = base / cohort
        promotions = [(int(path.stem.split("_", 1)[1]), path) for path in campaign.glob("promotion_*.json") if path.stem.split("_", 1)[1].isdigit()]
        budget, promotion_path = max(promotions)
        promotion = read(promotion_path)
        current = Path(promotion["wave"])
        previous = Path(read(current / "manifest.json")["source"])
        source = campaign / "source_checkpoint_io_v13"
        if source.exists():
            raise FileExistsError(source)
        shutil.copytree(previous, source, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        trainer = source / "algo/canonical_ppo.py"
        text = trainer.read_text()
        if text.count(old_save) != 1 or text.count(old_load) != 1:
            raise ValueError(f"unexpected checkpoint implementation: {cohort}")
        trainer.write_text(text.replace(old_save, new_save).replace(old_load, new_load))
        policy = source / "algo/canonical_policy.py"
        text = policy.read_text()
        if text.count(old_policy) != 1:
            raise ValueError(f"unexpected policy saver: {cohort}")
        policy.write_text(text.replace(old_policy, new_policy))
        shutil.copy2(workspace / "algo/checkpoint_io.py", source / "algo/checkpoint_io.py")
        changes = [str(path.relative_to(source)) for path in source.rglob("*.py") if not (previous / path.relative_to(source)).exists() or path.read_bytes() != (previous / path.relative_to(source)).read_bytes()]
        if set(changes) != {"algo/canonical_ppo.py", "algo/canonical_policy.py", "algo/checkpoint_io.py"}:
            raise ValueError(f"unexpected algorithm/source change: {cohort}")
        manifest["cohorts"][cohort] = {"campaign": str(campaign), "current": str(current), "budget": budget, "source": str(source), "previous_source": str(previous), "previous_training_job": promotion["training_job"], "changes": changes, "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.py")}}
    shutil.copy2(__file__, root / "recover.py")
    write(root / "manifest.json", manifest)
    print(root, flush=True)


def audit(root):
    import gc
    import types
    import zipfile
    from collections import defaultdict, deque

    import numpy as np
    import torch
    from algo.checkpoint_io import load_trusted_training, save_torch_atomic

    torch.set_num_threads(1)

    def equal_values(left, right, seen=None):
        if seen is None:
            seen = set()
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        pair = (id(left), id(right))
        if pair in seen:
            return True
        seen.add(pair)
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, np.ndarray):
            return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=left.dtype.kind in "fc")
        if isinstance(left, dict):
            factory_equal = not isinstance(left, defaultdict) or equal_values(left.default_factory, right.default_factory, seen)
            return factory_equal and left.keys() == right.keys() and all(equal_values(left[key], right[key], seen) for key in left)
        if isinstance(left, (tuple, list, deque)):
            if isinstance(left, deque) and left.maxlen != right.maxlen:
                return False
            return len(left) == len(right) and all(equal_values(a, b, seen) for a, b in zip(left, right))
        if isinstance(left, np.random.Generator):
            return equal_values(left.bit_generator.state, right.bit_generator.state, seen) and equal_values(left.bit_generator._seed_seq.state, right.bit_generator._seed_seq.state, seen)
        if isinstance(left, (type, types.FunctionType, types.BuiltinFunctionType)):
            return left == right
        if isinstance(left, types.MethodType):
            return left.__func__ == right.__func__ and equal_values(left.__self__, right.__self__, seen)
        if hasattr(left, "__dict__"):
            return equal_values(vars(left), vars(right), seen)
        if hasattr(left, "getstate"):
            return equal_values(left.getstate(), right.getstate(), seen)
        return bool(left == right)
    manifest = read(root / "manifest.json")
    records = []
    probe = None
    for cohort, settings in manifest["cohorts"].items():
        current = read(Path(settings["current"]) / "manifest.json")
        for run in current["runs"]:
            directory = Path(settings["campaign"]) / "training" / run["name"]
            path = directory / "training.pt"
            best_valid = zipfile.is_zipfile(directory / "best.pt")
            if best_valid:
                with zipfile.ZipFile(directory / "best.pt") as archive:
                    best_valid = archive.testzip() is None
            read(directory / "best_config.json")
            if not path.exists():
                if not best_valid:
                    raise ValueError(f"no valid policy or full state: {directory}")
                archive = root / "uncheckpointed_attempts" / cohort / run["name"]
                archive.mkdir(parents=True, exist_ok=True)
                for filename in ("best.pt", "best_config.json", "run_config.json", "resolved_config.json", "validation_history.json", "pretrain_history.json"):
                    if (directory / filename).exists():
                        shutil.copy2(directory / filename, archive / filename)
                records.append({"cohort": cohort, "name": run["name"], "checkpoint": str(path), "saved_steps": 0, "training_seed": run["seed"], "resume_mode": "repeat_original_initialization_no_full_checkpoint", "best_policy_archive_valid": True, "failed_attempt_archived": str(archive)})
                continue
            with path.open("rb") as stream:
                compressed = stream.read(2) == b"\x1f\x8b"
            if not compressed:
                with zipfile.ZipFile(path) as archive:
                    corrupt = archive.testzip()
                if corrupt is not None:
                    raise ValueError(f"corrupt saved training member: {path}, {corrupt}")
            state = load_trusted_training(path)
            repaired_best = False
            if not best_valid:
                archive = root / "damaged_selection_artifacts" / cohort / run["name"]
                archive.mkdir(parents=True, exist_ok=True)
                if not state.get("metadata", {}).get("revalidate_best_policy", False):
                    for filename in ("best.pt", "best_config.json", "validation_history.json", "training.pt"):
                        shutil.copy2(directory / filename, archive / filename)
                    state["metadata"] = {**state.get("metadata", {}), "best_score": (float("inf"), float("inf")), "validation_history": [], "revalidate_best_policy": True}
                    save_torch_atomic(state, path, compress=True)
                repaired_best = True
            record = {"cohort": cohort, "name": run["name"], "checkpoint": str(path), "bytes": path.stat().st_size, "saved_steps": state["total_steps"], "completed_updates": state["completed_updates"], "training_seed": state["ppo_config"]["seed"], "archive_crc_valid": True, "resume_mode": "full_state", "best_policy_revalidation_required": repaired_best}
            if record["training_seed"] != run["seed"]:
                raise ValueError("checkpoint seed differs from active run manifest")
            if probe is None:
                destination = root / "compressed_probe.pt"
                sizes = save_torch_atomic(state, destination, compress=True)
                restored = load_trusted_training(destination)
                for key, tensor in state["policy"].items():
                    if not torch.equal(tensor, restored["policy"][key]):
                        raise ValueError(f"compressed model tensor differs: {key}")
                if not equal_values(state["optimizer"], restored["optimizer"]):
                    raise ValueError("compressed optimizer state differs")
                for key in ("environments", "numpy_rng", "python_rng", "observations", "masks"):
                    if not equal_values(state[key], restored[key]):
                        raise ValueError(f"compressed checkpoint state differs: {key}")
                if not torch.equal(state["torch_rng"], restored["torch_rng"]) or state["total_steps"] != restored["total_steps"]:
                    raise ValueError("compressed RNG/interaction state differs")
                if state["cuda_rng"] is not None and any(not torch.equal(a, b) for a, b in zip(state["cuda_rng"], restored["cuda_rng"])):
                    raise ValueError("compressed accelerator RNG differs")
                probe = {**sizes, "source_checkpoint": str(path), "compressed_checkpoint": str(destination), "model_optimizer_environment_rng_roundtrip_equal": True}
                del restored
            records.append(record)
            del state
            gc.collect()
            write(root / "audit_progress.json", {"checked_checkpoints": len(records), "latest": record, "compression_probe": probe})
    write(root / "checkpoint_audit.json", {"status": "completed", "checkpoints": records, "compression_probe": probe, "goal_achieved": False})
    print(json.dumps({"checked_checkpoints": len(records), "compression_probe": probe}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.workspace.resolve(), args.directory.resolve())
    else:
        audit(args.directory.resolve())


if __name__ == "__main__":
    main()
