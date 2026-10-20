"""Prepare an explicitly derived runtime; never edit the immutable snapshot."""

import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(8 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


def relocate_workspace(workspace, manifest, account):
    origin = manifest["origin_workspace"]
    mappings = [(old, str(workspace / new)) for old, new in manifest["external_path_map"].items()]
    mappings.append((origin, str(workspace)))
    mappings.sort(key=lambda row: -len(row[0]))
    campaign = workspace / "results/california_graphppo"
    changes = []
    paths = [workspace / item["path"] for item in manifest["files"]]
    paths.extend((workspace / "scripts/campaigns").glob("*california*"))

    def record(path, before, reason):
        changes.append({"path": str(path.relative_to(workspace)), "origin_sha256": before,
                        "runtime_sha256": digest(path), "reason": reason})

    for path in dict.fromkeys(paths):
        if not path.is_file() or path.suffix not in {".py", ".json", ".jsonl", ".yaml", ".yml", ".sh", ".sbatch"}:
            continue
        original = path.read_text()
        relocated = original
        for old, new in mappings:
            relocated = relocated.replace(old, new)
        # Account changes apply to executable submission helpers, not to
        # historical job records. Run flags, model and simulator numbers stay.
        if path.suffix in {".py", ".sbatch", ".sh"}:
            relocated = relocated.replace("--account=tdsei17344", "--account=" + account)
            relocated = relocated.replace("--partition=gpu_h100", "--partition=gpu_mig")
            relocated = relocated.replace("--partition=cbuild", "--partition=gpu_mig")
        if original != relocated:
            before = digest(path)
            path.write_text(relocated)
            record(path, before, "filesystem relocation / SLURM account")

    for path in campaign.rglob("manifest.json"):
        if path.is_symlink():
            continue
        data = json.loads(path.read_text())
        if "runs" in data and "python" in data:
            before = digest(path)
            data.setdefault("slurm_resources", {})["account"] = account
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            record(path, before, "local SLURM account")
    # Job IDs from a different SLURM installation must never become live
    # dependencies, even if its scheduler happens to reuse a numeric ID.
    for path in campaign.rglob("training_dependencies.json"):
        before = digest(path)
        path.write_text(json.dumps({"afterany": [], "reason": "takeover: original scheduler dependencies are provenance only"}, indent=2) + "\n")
        record(path, before, "clear foreign scheduler dependencies")
    for root in campaign.glob("ca10t3d_*"):
        jobfile = root / "development_4m_jobs.json"
        summary = root / "development_4m_summary.json"
        if jobfile.exists() and not summary.exists():
            archive = root / "origin_scheduler"
            archive.mkdir(exist_ok=True)
            shutil.move(jobfile, archive / jobfile.name)
            changes.append({"path": str(jobfile.relative_to(workspace)), "reason": "archive incomplete foreign scoring job; fresh coordinator will schedule it"})

    # The trusted training files remain byte-identical. Only the runtime loader
    # adapts path strings in the live simulator objects they contain.
    wrapper = Path(__file__).with_name("runtime_paths.py").read_text()
    loaders = [path for path in paths if path.name == "checkpoint_io.py" and path.is_file()]
    for path in dict.fromkeys(loaders):
        before = digest(path)
        adapter = path.with_name("campaign_runtime_paths.py")
        adapter.write_text(wrapper)
        code = path.read_text()
        if "def load_trusted_training(" not in code:
            raise ValueError(f"unrecognized trusted checkpoint loader: {path}")
        code += ("\n\n# Explicit path-only runtime derivative; original source is in the handoff.\n"
                 "_campaign_original_load = load_trusted_training\n"
                 "def load_trusted_training(path, *, map_location='cpu'):\n"
                 "    from algo.campaign_runtime_paths import relocate_environments\n"
                 f"    return relocate_environments(_campaign_original_load(path, map_location=map_location), {mappings!r})\n")
        path.write_text(code)
        record(path, before, "path adapter for trusted saved environment objects")

    # Preserve origin lock files in the tracked snapshot. Derived runtime locks
    # record the actual relocated inputs, including nested lock dependencies.
    names = {"input_hashes.json", "source_lock.json", "completed_output_hashes.json",
             "prepared_inputs.json", "bootstrap_input_hashes.json"}
    locks = {path for path in paths if path.name in names and path.exists()}
    finished, visiting, missing = set(), set(), []

    def update(path):
        if path in finished:
            return
        if path in visiting:
            raise ValueError(f"cyclic input locks: {path}")
        visiting.add(path)
        data = json.loads(path.read_text())

        def replace(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(key, str) and key.startswith(str(workspace) + "/") and isinstance(child, str) and re.fullmatch(r"[0-9a-f]{64}", child):
                        target = Path(key)
                        if target in locks:
                            update(target)
                        if target.is_file():
                            value[key] = digest(target)
                        else:
                            missing.append({"lock": str(path), "input": key})
                    else:
                        replace(child)
            elif isinstance(value, list):
                for child in value:
                    replace(child)

        replace(data)
        before = digest(path)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        record(path, before, "derived runtime input hashes; original hashes preserved in snapshot")
        visiting.remove(path)
        finished.add(path)

    for path in sorted(locks):
        update(path)
    (campaign / "handoff_unresolved_historical_inputs.json").write_text(json.dumps(missing, indent=2) + "\n")
    return changes
