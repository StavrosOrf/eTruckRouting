"""Capture, check and restore the California GraphPPO campaign without publishing.

Only ``export`` reads the live campaign. ``restore`` refuses the original
workspace and an existing campaign. Large files use content-addressed Git LFS
chunks; the manifest contains original file hashes, not LFS pointer hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone


WORK = Path(__file__).resolve().parents[2]
DEFAULT = WORK / "campaigns/california_graphppo"
CAMPAIGN = Path("results/california_graphppo")
CHUNK_BYTES = 256 * 1024 * 1024
INLINE_BYTES = 1024 * 1024
SKIP = {"__pycache__", "runtime_cache", ".git", ".pytest_cache"}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(8 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


def safe(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    return root / relative


def export(destination):
    if (destination / "manifest.json").exists():
        raise FileExistsError("snapshot already sealed; choose a new destination")
    destination.mkdir(parents=True, exist_ok=True)
    origin = str(WORK)
    recovery = WORK / CAMPAIGN / "recovery_data_4096_v15/data"
    external = {str(recovery.resolve()): str(CAMPAIGN / "recovery_data_4096_v15/data")}
    files, links, excluded, captured = [], [], [], {}
    started = datetime.now(timezone.utc).isoformat()

    def capture(path, relative):
        # An open descriptor keeps one coherent inode if the trainer publishes
        # its next checkpoint with os.replace while this file is being read.
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if key in captured:
                previous = captured[key]
                if previous["storage"] == "snapshot":
                    target = safe(destination / "snapshot", relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(safe(destination / "snapshot", previous["path"]), target)
                files.append(dict(previous, path=str(relative)))
                return
            item = {"path": str(relative), "size": before.st_size,
                    "mode": stat.S_IMODE(before.st_mode),
                    "mtime_ns": before.st_mtime_ns}
            large = before.st_size >= INLINE_BYTES or path.suffix in {".pt", ".npz", ".npy"}
            digest = hashlib.sha256()
            if large:
                item.update(storage="chunks", chunks=[])
                while data := handle.read(CHUNK_BYTES):
                    digest.update(data)
                    chunk_sha = hashlib.sha256(data).hexdigest()
                    chunk = destination / "artifacts/chunks" / (chunk_sha + ".chunk")
                    if not chunk.exists():
                        chunk.parent.mkdir(parents=True, exist_ok=True)
                        temporary = chunk.with_suffix(".tmp")
                        temporary.write_bytes(data)
                        temporary.replace(chunk)
                    item["chunks"].append({"sha256": chunk_sha, "size": len(data)})
            else:
                data = handle.read()
                digest.update(data)
                target = safe(destination / "snapshot", relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(item["mode"])
                item["storage"] = "snapshot"
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"file changed in place during capture; retry export: {path}")
            item["sha256"] = digest.hexdigest()
            captured[key] = item
            files.append(item)
            if len(files) % 400 == 0:
                print(f"captured {len(files)} files", flush=True)

    def walk(directory, relative):
        for path in sorted(directory.iterdir()):
            rel = relative / path.name
            if path.name in SKIP or ".tmp." in path.name or path.suffix in {".pyc", ".lock", ".tmp"}:
                excluded.append(str(rel))
                continue
            if path.name == "data" and path.parent.name == "EVRoutingEnv":
                # Frozen sources import the common California inputs; unrelated
                # European matrices and visualizations are outside this task.
                links.append({"path": str(rel), "target": "EVRoutingEnv/data",
                              "origin_target": str(path.resolve()), "role": "shared case-study data"})
                continue
            if path.is_symlink():
                if path == recovery:
                    walk(path, rel)
                    continue
                resolved = path.resolve()
                try:
                    target = str(resolved.relative_to(WORK))
                except ValueError:
                    matching = [prefix for prefix in external if str(resolved).startswith(prefix + "/") or str(resolved) == prefix]
                    if not matching:
                        raise ValueError(f"unmapped external symlink: {path} -> {resolved}")
                    prefix = max(matching, key=len)
                    target = external[prefix] + str(resolved)[len(prefix):]
                if not resolved.exists():
                    raise FileNotFoundError(f"dangling campaign symlink: {path}")
                links.append({"path": str(rel), "target": target, "origin_target": os.readlink(path)})
            elif path.is_dir():
                walk(path, rel)
            elif path.is_file():
                capture(path, rel)

    walk(WORK / CAMPAIGN, CAMPAIGN)
    # Pin the only case-study inputs used by the campaign independently of the
    # checkout's future data revisions.
    california = Path("EVRoutingEnv/data/case_studies/california")
    walk(WORK / california, california)
    unique = {part["sha256"]: part["size"] for item in files for part in item.get("chunks", [])}
    manifest = {
        "version": 1, "origin_workspace": origin, "external_path_map": external,
        "started_at": started, "sealed_at": datetime.now(timezone.utc).isoformat(),
        "origin_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORK, text=True).strip(),
        "files": files, "symlinks": links, "excluded": excluded,
        "exclusion_policy": "runtime/scenario caches, Python bytecode, temporary writes, locks, non-California source data",
        "chunk_bytes": CHUNK_BYTES, "unique_chunk_count": len(unique),
        "unique_artifact_bytes": sum(unique.values()),
        "logical_file_bytes": sum(item["size"] for item in files),
        "snapshot_consistency": "per-file coherent capture; live jobs were not stopped; not a globally synchronized save",
        "github_publication": "NOT PUBLISHED: user will commit and push including Git LFS objects",
        "goal_achieved": False,
    }
    dump(destination / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("sealed_at", "unique_chunk_count", "unique_artifact_bytes", "logical_file_bytes")}))


def check(bundle, full=True):
    manifest = json.loads((bundle / "manifest.json").read_text())
    checked_chunks = set()
    for item in manifest["files"]:
        if item["storage"] == "snapshot":
            path = safe(bundle / "snapshot", item["path"])
            if path.stat().st_size != item["size"] or (full and sha(path) != item["sha256"]):
                raise ValueError(f"snapshot file differs: {path}")
        else:
            digest = hashlib.sha256()
            for part in item["chunks"]:
                path = bundle / "artifacts/chunks" / (part["sha256"] + ".chunk")
                if path.stat().st_size != part["size"]:
                    raise ValueError(f"missing LFS contents (run git lfs pull): {path}")
                if full:
                    # Full logical SHA is checked even when chunks are reused.
                    with path.open("rb") as handle:
                        chunk_digest = hashlib.sha256()
                        while data := handle.read(8 * 1024 * 1024):
                            digest.update(data)
                            chunk_digest.update(data)
                    if chunk_digest.hexdigest() != part["sha256"]:
                        raise ValueError(f"corrupt chunk: {path}")
                checked_chunks.add(part["sha256"])
            if full and digest.hexdigest() != item["sha256"]:
                raise ValueError(f"reconstructed file differs: {item['path']}")
    result = {"status": "passed", "full_sha256_verification": full,
              "files": len(manifest["files"]), "unique_chunks": len(checked_chunks),
              "goal_achieved": False}
    print(json.dumps(result))
    return manifest


def restore(bundle, workspace, account, relocate):
    manifest = check(bundle, full=False)
    workspace = workspace.resolve()
    if workspace == Path(manifest["origin_workspace"]):
        raise ValueError("restore is for a different checkout; never overwrite live training")
    if (workspace / CAMPAIGN).exists():
        raise FileExistsError("target campaign exists; restore into a fresh clone")
    records = []
    for item in manifest["files"]:
        target = safe(workspace, item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != item["sha256"]:
                raise FileExistsError(f"existing checkout input differs: {target}")
            continue
        digest = hashlib.sha256()
        temporary = target.with_name(target.name + ".handoff-tmp")
        with temporary.open("wb") as output:
            paths = [safe(bundle / "snapshot", item["path"])] if item["storage"] == "snapshot" else [bundle / "artifacts/chunks" / (part["sha256"] + ".chunk") for part in item["chunks"]]
            for path in paths:
                with path.open("rb") as handle:
                    while data := handle.read(8 * 1024 * 1024):
                        output.write(data)
                        digest.update(data)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"restore content mismatch: {target}")
        temporary.chmod(item["mode"])
        temporary.replace(target)
    for item in manifest["symlinks"]:
        link = safe(workspace, item["path"])
        target = safe(workspace, item["target"])
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
    if relocate:
        # Import from this versioned tool, not any snapshotted historical helper.
        sys.path.insert(0, str(bundle))
        from relocate import relocate_workspace
        records = relocate_workspace(workspace, manifest, account)
    dump(workspace / CAMPAIGN / "handoff_restore.json", {
        "status": "restored", "restored_at": datetime.now(timezone.utc).isoformat(),
        "bundle_sha256": sha(bundle / "manifest.json"), "origin_workspace": manifest["origin_workspace"],
        "workspace": str(workspace), "relocated": relocate, "changes": records,
        "account": account, "scheduler_jobs_submitted": 0,
        "original_scheduler_ids_are_provenance_only": True, "goal_achieved": False,
    })
    print(f"restored into {workspace}; no SLURM jobs submitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "check", "restore"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--account")
    parser.add_argument("--sizes-only", action="store_true")
    parser.add_argument("--original-paths", action="store_true", help="restore immutable bytes without preparing a relocated runtime")
    args = parser.parse_args()
    if args.action == "export":
        export(args.bundle.resolve())
    elif args.action == "check":
        check(args.bundle.resolve(), full=not args.sizes_only)
    else:
        if not args.workspace or not args.account:
            parser.error("restore requires --workspace and --account")
        restore(args.bundle.resolve(), args.workspace, args.account, not args.original_paths)


if __name__ == "__main__":
    main()
