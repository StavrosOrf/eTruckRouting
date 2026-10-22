"""The generalist campaign freezes its problems, arms and shards, composes sbatch, and packs jobs."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.campaigns import generalist_graphppo as campaign


def _bks(tmp_path) -> Path:
    """A best-known-solutions directory with the files prepare requires."""
    root = tmp_path / "bks"
    for name in campaign.PROBLEMS:
        (root / name).mkdir(parents=True)
        (root / name / "validation.json").write_text(json.dumps({"1000000000": {"best": {"hours": 1.0}}}))
        (root / name / "train_demonstrations.json").write_text("[]")
        (root / name / "summary.json").write_text("{}")
    return root


def test_prepare_freezes_problems_arms_bks_and_source(tmp_path) -> None:
    destination = tmp_path / "suite"
    bks = _bks(tmp_path)
    campaign.prepare(destination, "screen", "full", 3, bks_from=bks)
    manifest = campaign.read_manifest(destination)
    problems = json.loads((destination / "problems.json").read_text())
    assert [row["name"] for row in problems] == list(campaign.PROBLEMS)
    assert all((destination / "configs" / f"{row['name']}.yaml").exists() for row in problems)
    # The size probes are out of scope unless asked for.
    assert json.loads((destination / "probes.json").read_text()) == []
    assert not list((destination / "configs").glob("*_t5*")) and manifest["probes_enabled"] is False
    assert (destination / "source/algo/generalist_policy.py").exists()
    names = [run["name"] for run in manifest["runs"]]
    assert names[:7] == ["base_s0", "fleet_s0", "paired_s0", "full_s0", "full_bks_s0", "full_bks_retained_s0",
                         "full_dagger_s0"]
    assert len(manifest["runs"]) == 21
    # The BKS copy is frozen inside the campaign and every {bks} points at it.
    for name in campaign.PROBLEMS:
        assert (destination / "bks" / name / "validation.json").exists()
        assert (destination / "bks" / name / "summary.json").exists()
    assert manifest["bks"] == str(destination / "bks") and manifest["bks_from"] == str(bks)
    assert not any("{bks}" in flag for run in manifest["runs"] for flag in run["flags"])
    assert str(destination / "bks") in manifest["runs"][4]["flags"]
    assert "--demonstrations" not in manifest["runs"][3]["flags"]
    with pytest.raises(FileExistsError):
        campaign.prepare(destination, "screen", "full", 3, bks_from=bks)


def test_probes_are_written_only_on_request(tmp_path) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path), probes=True)
    probes = json.loads((destination / "probes.json").read_text())
    assert [row["name"] for row in probes] == list(campaign.PROBES)
    assert all((destination / "configs" / f"{row['name']}.yaml").exists() for row in probes)
    assert campaign.read_manifest(destination)["probes_enabled"] is True


def test_imitation_arms_need_best_known_solutions(tmp_path) -> None:
    with pytest.raises(SystemExit, match="needs best-known solutions"):
        campaign.prepare(tmp_path / "suite", "screen", "full", 1)
    assert not (tmp_path / "suite").exists()
    incomplete = tmp_path / "partial"
    (incomplete / "ca_sameday").mkdir(parents=True)
    with pytest.raises(SystemExit, match="lacks best-known solutions"):
        campaign.prepare(tmp_path / "suite", "screen", "full", 1, bks_from=incomplete)
    # The original arms need no BKS; a named set runs the listed recipes in order.
    campaign.prepare(tmp_path / "plain", "named", "full", 2, arms=["base", "full"])
    manifest = campaign.read_manifest(tmp_path / "plain")
    assert [run["name"] for run in manifest["runs"]] == ["base_s0", "full_s0", "base_s1", "full_s1"]
    assert manifest["arms"] == ["base", "full"] and manifest["bks"] is None
    with pytest.raises(SystemExit, match="needs --arms"):
        campaign.arms_for("named", "full", 1)
    with pytest.raises(SystemExit, match="unknown recipes"):
        campaign.arms_for("named", "full", 1, arms=["nope"])


def test_imitation_arms_override_the_common_flags(tmp_path) -> None:
    from scripts.training.train_generalist_ppo import build_argument_parser

    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    manifest = campaign.read_manifest(destination)
    parser = build_argument_parser()
    parsed = {}
    for run in manifest["runs"]:
        # The trainer argv as train() composes it: common flags first, the arm's flags last.
        argv = ["--problems", manifest["problems"], "--references", str(destination / "references"),
                *manifest["common_flags"], "--total-timesteps", "5120", "--seed", str(run["seed"]), *run["flags"]]
        parsed[run["arm"]] = parser.parse_args(argv)
    assert parsed["full"].bc_episodes == 64 and parsed["full"].demonstrations is None
    for arm in ("full_bks", "full_bks_retained", "full_dagger"):
        arguments = parsed[arm]
        assert arguments.bc_episodes == 256 and arguments.demonstrations == str(destination / "bks")
        assert arguments.reward_references == str(destination / "bks") and arguments.reward_reference == "scenario"
        assert arguments.fleet_assignment
    assert parsed["full_bks"].bc_coefficient == 0.0 and parsed["full_bks"].dagger_expert == []
    assert parsed["full_bks_retained"].bc_coefficient == 0.1
    assert parsed["full_bks_retained"].bc_anneal_timesteps == 2_000_000 and parsed["full_bks_retained"].dagger_expert == []
    dagger = parsed["full_dagger"]
    assert dagger.bc_coefficient == 0.1 and len(dagger.dagger_expert) == 5 and dagger.dagger_pretrain_rounds == 2
    assert dagger.dagger_every == 10 and dagger.dagger_beta == 0.5 and dagger.dagger_beta_anneal_timesteps == 2_000_000
    assert all("=algo.generalist_experts:" in spec for spec in dagger.dagger_expert)


def test_leave_one_problem_out_and_specialist_arms() -> None:
    lopo = campaign.arms_for("lopo", "full", 1)
    assert len(lopo) == len(campaign.PROBLEMS)
    for run, held in zip(lopo, campaign.PROBLEMS, strict=True):
        trained = run["flags"][run["flags"].index("--train-problems") + 1 :]
        assert held not in trained and len(trained) == len(campaign.PROBLEMS) - 1
        assert "--fleet-assignment" in run["flags"]
    specialists = campaign.arms_for("specialists", "base", 2)
    assert len(specialists) == 2 * len(campaign.PROBLEMS)
    assert all(run["flags"][-2] == "--train-problems" for run in specialists)


def test_reference_shards_cover_train_and_validation_only(tmp_path) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    tasks = campaign.reference_tasks(campaign.read_manifest(destination))
    per_problem = -(-campaign.SEEDS["train"] // campaign.SHARD) + -(-campaign.SEEDS["validation"] // campaign.SHARD)
    assert len(tasks) == per_problem * len(campaign.PROBLEMS)
    assert {split for _, split, _ in tasks} == {"train", "validation"}


def test_submit_composes_the_array_from_the_manifest(tmp_path, capsys) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    with pytest.raises(SystemExit, match="merge the references"):
        campaign.submit(destination, "train", 2_000_000, dry_run=True)
    (destination / "references" / "summary.json").write_text("{}")
    campaign.submit(destination, "train", 2_000_000, dry_run=True)
    command = capsys.readouterr().out.strip().splitlines()[-1]
    slurm = campaign.SLURM
    for part in (f"--partition={slurm['gpu_partition']}", f"--account={slurm['account']}", "--gpus-per-task=1",
                 f"--cpus-per-task={slurm['cpus_per_gpu']}", f"--mem-per-cpu={slurm['gpu_mem_per_cpu']}", "--array=0-6%20",
                 "--time=12:00:00", "generalist_graphppo_gpu.sbatch"):
        assert part in command
    # Unpacked: one run per array element; the arguments are budget, runs per GPU, then the groups.
    assert command.endswith(f"{destination} 2000000 1 0 1 2 3 4 5 6")


def test_submit_train_selects_the_runs_of_the_given_arms(tmp_path, capsys) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 2, bks_from=_bks(tmp_path))
    (destination / "references" / "summary.json").write_text("{}")
    campaign.submit(destination, "train", 8_000_000, dry_run=True, arms=["fleet", "full"])
    command = capsys.readouterr().out.strip().splitlines()[-1]
    assert "--array=0-3%20" in command and "--time=36:00:00" in command and command.endswith(" 8000000 1 1 3 8 10")
    with pytest.raises(SystemExit, match="unknown arms"):
        campaign.submit(destination, "train", 8_000_000, dry_run=True, arms=["nope"])


def test_submit_evaluate_composes_one_gpu_allocation(tmp_path, capsys) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    with pytest.raises(SystemExit, match="--tag"):
        campaign.submit(destination, "evaluate", None, dry_run=True)
    with pytest.raises(SystemExit, match="unknown runs"):
        campaign.submit(destination, "evaluate", None, dry_run=True, tag="2M", runs=["nope_s0"])
    campaign.submit(destination, "evaluate", None, dry_run=True, split="validation", tag="2M", runs=["full_s0"])
    command = capsys.readouterr().out.strip().splitlines()[-1]
    slurm = campaign.SLURM
    for part in (f"--partition={slurm['gpu_partition']}", f"--cpus-per-task={slurm['cpus_per_gpu']}", "--gpus-per-task=1",
                 f"--mem-per-cpu={slurm['gpu_mem_per_cpu']}", "generalist_graphppo_eval.sbatch"):
        assert part in command
    assert command.endswith(f"{destination} validation 2M full_s0")
    assert "--array" not in command


def test_prepare_copies_merged_references_from_another_campaign(tmp_path) -> None:
    source = tmp_path / "references"
    source.mkdir()
    bks = _bks(tmp_path)
    with pytest.raises(SystemExit, match="lacks merged references"):
        campaign.prepare(tmp_path / "early", "screen", "full", 1, references_from=source, bks_from=bks)
    for name in ("summary", *campaign.PROBLEMS):
        (source / f"{name}.json").write_text(json.dumps({"name": name}))
    destination = tmp_path / "pilot"
    campaign.prepare(destination, "screen", "full", 1, references_from=source, bks_from=bks)
    for name in ("summary", *campaign.PROBLEMS):
        assert json.loads((destination / "references" / f"{name}.json").read_text()) == {"name": name}
    assert campaign.read_manifest(destination)["references_from"] == str(source)


def _row(problem: str, seed: int, served: int, hours: float) -> dict:
    return {
        "problem": problem, "seed": seed, "dynamic": False, "success": served == 30, "failed": False,
        "hours": hours, "completed_fraction": served / 30, "tasks": 30, "served": served,
        "late": 0, "lost": 0, "unserved": 30 - served,
    }


def test_evaluation_rows_and_reports_are_kept_per_tag(tmp_path) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    folder = destination / "evaluation" / "validation"
    (folder / "baselines").mkdir(parents=True)
    (folder / "2M").mkdir()
    name = "tx_longhaul"
    campaign.write_json(folder / "baselines" / f"{name}__teacher.json", [_row(name, s, 30, 350.0) for s in (1, 2)])
    # Policy rows carry their score from evaluation; baseline scores are computed by the report.
    campaign.write_json(folder / "2M" / "full_s0.json", [{**_row(name, s, 30, 340.0), "score": 3.03} for s in (1, 2)])
    campaign.report(destination, "validation", "2M")
    text = (folder / "2M" / "report.md").read_text()
    assert "| full | 2 |" in text and "| baseline:teacher | 2 |" in text
    campaign.report(destination, "validation", "8M")
    text = (folder / "8M" / "report.md").read_text()
    assert "| full |" not in text and "| baseline:teacher | 2 |" in text
    assert not (folder / "report.md").exists()


def test_report_promotes_the_eligible_arms(tmp_path) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 2, bks_from=_bks(tmp_path))
    folder = destination / "evaluation" / "validation"
    (folder / "baselines").mkdir(parents=True)
    (folder / "2M").mkdir()
    for name in campaign.PROBLEMS:
        campaign.write_json(folder / "baselines" / f"{name}__teacher.json", [_row(name, s, 30, 350.0) for s in (1, 2)])

    def rows(served_by_problem: dict, score: float) -> list[dict]:
        return [{**_row(name, s, served_by_problem.get(name, 30), 340.0), "score": score}
                for name in campaign.PROBLEMS for s in (1, 2)]

    for seed in (0, 1):
        campaign.write_json(folder / "2M" / f"full_s{seed}.json", rows({}, 3.0))
        campaign.write_json(folder / "2M" / f"fleet_s{seed}.json", rows({}, 2.0 + seed))
        campaign.write_json(folder / "2M" / f"base_s{seed}.json", rows({"tx_longhaul": 28}, 4.0))
        campaign.write_json(folder / "2M" / f"paired_s{seed}.json",
                            [row for row in rows({}, 5.0) if row["problem"] != "pnw_longhaul_hos"])
    campaign.report(destination, "validation", "2M", promote_count=2)
    result = json.loads((folder / "2M" / "promotion.json").read_text())
    assert result["promoted"] == ["full", "fleet"]
    assert result["arms"]["base"]["eligible"] is False and "tx_longhaul: served" in result["arms"]["base"]["reasons"][0]
    assert result["arms"]["paired"]["eligible"] is False and "rows missing" in result["arms"]["paired"]["reasons"][0]
    assert result["arms"]["fleet"]["mean_score"] == pytest.approx(2.5) and result["arms"]["fleet"]["seeds"] == 2
    assert "full_bks" not in result["arms"] and result["teacher"]["tx_longhaul"]["served_fraction"] == 1.0
    text = (folder / "2M" / "report.md").read_text()
    assert "## Promotion" in text and "| full | 2 | 3.000 | yes |" in text


# ---------------------------------------------------------------------------
# Job packing
# ---------------------------------------------------------------------------


def _prepared(tmp_path, seeds: int = 3):
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", seeds, bks_from=_bks(tmp_path))
    (destination / "references" / "summary.json").write_text("{}")
    return destination


def _commands(capsys) -> list[str]:
    return [line for line in capsys.readouterr().out.strip().splitlines() if line.startswith("sbatch ")]


def test_packed_training_asks_for_one_node_with_k_gpus_and_k_cpu_shares(tmp_path, capsys) -> None:
    destination = _prepared(tmp_path)
    slurm = campaign.SLURM
    campaign.submit(destination, "train", 2_000_000, dry_run=True, pack=3)
    (command,) = _commands(capsys)
    for part in ("--nodes=1", "--ntasks=1", "--gpus-per-task=3", f"--cpus-per-task={3 * slurm['cpus_per_gpu']}",
                 f"--mem-per-cpu={slurm['gpu_mem_per_cpu']}", "--array=0-6%20", "--signal=B:USR1@300"):
        assert part in command.split()
    assert command.endswith(f"{destination} 2000000 1 0,1,2 3,4,5 6,7,8 9,10,11 12,13,14 15,16,17 18,19,20")
    # Two runs per GPU halve the GPUs, not the CPUs.
    campaign.submit(destination, "train", 2_000_000, dry_run=True, pack=7, runs_per_gpu=2)
    (command,) = _commands(capsys)
    assert "--gpus-per-task=4" in command and f"--cpus-per-task={7 * slurm['cpus_per_gpu']}" in command
    assert "--array=0-2%20" in command
    assert command.endswith(" 2000000 2 0,1,2,3,4,5,6 7,8,9,10,11,12,13 14,15,16,17,18,19,20")


def test_a_short_last_group_gets_its_own_smaller_array(tmp_path, capsys) -> None:
    destination = _prepared(tmp_path)
    campaign.submit(destination, "train", 2_000_000, dry_run=True, pack=5)
    full, short = _commands(capsys)
    assert "--gpus-per-task=5" in full and "--array=0-3%20" in full
    assert full.endswith(" 0,1,2,3,4 5,6,7,8,9 10,11,12,13,14 15,16,17,18,19")
    assert "--gpus-per-task=1" in short and "--array=0-0%20" in short and short.endswith(" 20")
    assert "--cpus-per-task=18" in short


def test_packing_groups_only_unfinished_runs(tmp_path, capsys, monkeypatch) -> None:
    destination = _prepared(tmp_path)
    finished = {"base_s0", "paired_s0", "full_s1"}
    monkeypatch.setattr(campaign, "completed_run", lambda directory, run, budget: run["name"] in finished)
    campaign.submit(destination, "train", 2_000_000, dry_run=True, pack=3, arms=["base", "paired", "full"])
    (command,) = _commands(capsys)
    runs = campaign.read_manifest(destination)["runs"]
    pending = [index for index, run in enumerate(runs) if run["arm"] in {"base", "paired", "full"}
               and run["name"] not in finished]
    assert command.endswith(" 2000000 1 " + " ".join(
        ",".join(map(str, pending[start:start + 3])) for start in range(0, len(pending), 3)))


def test_pack_and_stride_groups() -> None:
    assert campaign.pack_groups([3, 4, 5, 6, 7], 2) == [[3, 4], [5, 6], [7]]
    groups = campaign.strided_groups(list(range(45)), 4)
    assert len(groups) == 4 and sorted(sum(groups, [])) == list(range(45))
    assert groups[0][:3] == [0, 4, 8]
    assert campaign.strided_groups([1, 2], 5) == [[1], [2]]
    with pytest.raises(SystemExit):
        campaign.pack_groups([1], 0)


def test_cpu_slices_are_disjoint_contiguous_and_cover_the_allocation() -> None:
    cpus = [*range(16, 32), *range(48, 64)]
    slices = campaign.cpu_slices(set(cpus), 2)
    assert slices == [list(range(16, 32)), list(range(48, 64))]
    uneven = campaign.cpu_slices(range(10), 3)
    assert [len(part) for part in uneven] == [4, 3, 3] and sum(uneven, []) == list(range(10))
    with pytest.raises(SystemExit, match="cannot give"):
        campaign.cpu_slices(range(8), 3)
    assert campaign.compact([0, 1, 2, 3, 7, 9, 10]) == "0-3,7,9-10"


def test_gpu_assignment_pins_each_run_or_shares_a_gpu() -> None:
    assert campaign.gpu_assignment(["2", "3"], 2, 1) == ["2", "3"]
    assert campaign.gpu_assignment(["0", "1"], 4, 2) == ["0", "0", "1", "1"]
    assert campaign.gpu_assignment(["0", "1"], 3, 2) == ["0", "0", "1"]
    assert campaign.gpu_assignment([], 2, 1) == [None, None]
    with pytest.raises(SystemExit, match="need 2 GPUs"):
        campaign.gpu_assignment(["0"], 2, 1)


def test_plan_pack_gives_each_unfinished_run_its_own_slice_and_gpu(tmp_path, monkeypatch) -> None:
    destination = _prepared(tmp_path, seeds=1)
    monkeypatch.setattr(campaign, "completed_run", lambda directory, run, budget: run["name"] == "fleet_s0")
    plan = campaign.plan_pack(destination, [0, 1, 2], 2_000_000, 1, range(48), ["0", "1", "2"])
    assert [item["name"] for item in plan] == ["base_s0", "paired_s0"]
    assert [item["gpu"] for item in plan] == ["0", "1"]
    assert plan[0]["cpus"] == list(range(24)) and plan[1]["cpus"] == list(range(24, 48))
    command = plan[1]["command"]
    assert command[2:] == ["train", "--directory", str(destination), "--run", "2", "--budget", "2000000"]
    assert campaign.plan_pack(destination, [1], 2_000_000, 1, range(48), ["0"]) == []


_CHILD = r"""
import json, os, signal, sys, time
target = sys.argv[1]
def stop(signum, frame):
    json.dump({"cpus": sorted(os.sched_getaffinity(0)), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "signal": signum}, open(target, "w"))
    sys.exit(0)
signal.signal(signal.SIGUSR1, stop)
open(target + ".ready", "w").close()
for _ in range(600):
    time.sleep(0.1)
sys.exit(3)
"""


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity")
def test_supervise_pins_children_and_forwards_the_checkpoint_signal_to_all(tmp_path, monkeypatch) -> None:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    targets = [tmp_path / "a.json", tmp_path / "b.json"]
    plan = [
        {"name": name, "cpus": [cpu], "gpu": gpu, "command": [sys.executable, "-c", _CHILD, str(target)]}
        for name, cpu, gpu, target in zip(("a", "b"), cpus[:2], ("0", "1"), targets, strict=True)
    ]

    def interrupt() -> None:
        deadline = time.time() + 30
        while time.time() < deadline and not all(Path(f"{target}.ready").exists() for target in targets):
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)  # as at the time limit; children get USR1

    before = signal.getsignal(signal.SIGTERM)
    thread = threading.Thread(target=interrupt)
    thread.start()
    codes = campaign.supervise(plan)
    thread.join()
    assert codes == [0, 0]
    for target, cpu, gpu in zip(targets, cpus[:2], ("0", "1"), strict=True):
        record = json.loads(target.read_text())
        assert record == {"cpus": [cpu], "gpu": gpu, "signal": int(signal.SIGUSR1)}
    assert signal.getsignal(signal.SIGTERM) is before


def test_packed_references_split_pending_shards_over_gpu_shares(tmp_path, capsys) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    manifest = campaign.read_manifest(destination)
    tasks = campaign.reference_tasks(manifest)
    campaign.submit(destination, "references", None, dry_run=True, pack=3)
    (command,) = _commands(capsys)
    slurm = campaign.SLURM
    # A CPU stage on a GPU partition still holds one GPU.
    for part in (f"--partition={slurm['cpu_partition']}", f"--cpus-per-task={slurm['cpu_node_cores']}",
                 f"--mem-per-cpu={slurm['cpu_mem_per_cpu']}", "--nodes=1", "--array=0-2%20",
                 f"--gpus-per-task={slurm['cpu_gpus']}"):
        assert part in command.split()
    assert "generalist_graphppo_cpu.sbatch" in command
    groups = [list(map(int, group.split(","))) for group in command.split(" references ")[1].split()]
    assert len(groups) == 3 and sorted(sum(groups, [])) == list(range(len(tasks)))
    # Finished shards are not resubmitted.
    shards = destination / "references" / "shards"
    shards.mkdir(parents=True)
    for name, split, shard in tasks[:40]:
        (shards / f"{name}_{split}_{shard:03d}.json").write_text("[]")
    campaign.submit(destination, "references", None, dry_run=True, pack=3, cpus=40)
    (command,) = _commands(capsys)
    assert "--cpus-per-task=40" in command and command.endswith(" references 40,43 41,44 42")
    for name, split, shard in tasks[40:]:
        (shards / f"{name}_{split}_{shard:03d}.json").write_text("[]")
    with pytest.raises(SystemExit, match="every reference shard exists"):
        campaign.submit(destination, "references", None, dry_run=True, pack=3)


def _fake_campaign(tmp_path) -> tuple:
    """A manifest whose interpreter records its arguments instead of running them."""
    source = tmp_path / "source"
    source.mkdir()
    record = tmp_path / "arguments.txt"
    fake = tmp_path / "python"
    fake.write_text(f"#!/bin/bash\necho \"$@\" > {record}\n")
    fake.chmod(0o755)
    directory = tmp_path / "campaign"
    (directory / "logs").mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"python": str(fake), "source": str(source)}))
    return directory, record


@pytest.mark.parametrize("index, group", [("0", "0,1"), ("2", "4,5")])
def test_the_gpu_wrapper_trains_the_group_of_its_array_index(tmp_path, index, group) -> None:
    directory, record = _fake_campaign(tmp_path)
    environment = {**os.environ, "SLURM_ARRAY_TASK_ID": index, "SLURM_JOB_ID": "7", "PATH": "/usr/bin:/bin"}
    subprocess.run(
        ["bash", str(campaign.ROOT / "scripts/campaigns/generalist_graphppo_gpu.sbatch"),
         str(directory), "2000000", "2", "0,1", "2,3", "4,5"],
        env=environment, check=True, capture_output=True,
    )
    assert record.read_text().split() == [
        "scripts/campaigns/generalist_graphppo.py", "train-pack", "--directory", str(directory),
        "--runs", group, "--budget", "2000000", "--runs-per-gpu", "2",
    ]


def test_the_cpu_wrapper_runs_the_shards_of_its_array_index_on_all_cores(tmp_path) -> None:
    directory, record = _fake_campaign(tmp_path)
    environment = {**os.environ, "SLURM_ARRAY_TASK_ID": "1", "SLURM_CPUS_PER_TASK": "48", "PATH": "/usr/bin:/bin"}
    subprocess.run(
        ["bash", str(campaign.ROOT / "scripts/campaigns/generalist_graphppo_cpu.sbatch"),
         str(directory), "references", "0,3", "1,4", "2"],
        env=environment, check=True, capture_output=True,
    )
    assert record.read_text().split() == [
        "scripts/campaigns/generalist_graphppo.py", "references", "--directory", str(directory),
        "--tasks", "1,4", "--workers", "48",
    ]


def test_every_composed_command_fits_snellius(tmp_path, capsys) -> None:
    destination = _prepared(tmp_path)
    campaign.submit(destination, "references", None, dry_run=True, pack=2)
    campaign.submit(destination, "train", 2_000_000, dry_run=True, pack=3)
    campaign.submit(destination, "evaluate", None, dry_run=True, tag="2M")
    commands = _commands(capsys)
    assert len(commands) == 3
    for command in commands:
        options = {part.split("=")[0]: part.partition("=")[2] for part in command.split() if part.startswith("--")}
        assert options["--account"] == "tdsei17344" and options["--partition"] == "gpu_a100"
        # Every job on a GPU partition holds a GPU, the CPU stage included; the
        # site takes --mem-per-cpu and --gpus-per-task.
        assert int(options["--gpus-per-task"]) >= 1 and "--mem-per-cpu" in options
        assert "--mem" not in options and "--gpus" not in options and "--gres" not in options
        if "--array" in options:
            assert options["--array"].endswith("%20")


def test_an_overflow_partition_changes_the_cores_per_gpu(tmp_path, capsys) -> None:
    destination = _prepared(tmp_path, seeds=1)
    campaign.submit(destination, "train", 2_000_000, dry_run=True, pack=2, partition="gpu_h100")
    full, short = _commands(capsys)
    assert "--partition=gpu_h100" in full and "--cpus-per-task=32" in full and "--gpus-per-task=2" in full
    assert "--cpus-per-task=16" in short and "--gpus-per-task=1" in short
    campaign.submit(destination, "evaluate", None, dry_run=True, tag="2M", partition="gpu_mig")
    (command,) = _commands(capsys)
    assert "--partition=gpu_mig" in command and "--cpus-per-task=9" in command
    campaign.submit(destination, "evaluate", None, dry_run=True, tag="2M", partition="gpu_a100", cpus_per_gpu=12)
    (command,) = _commands(capsys)
    assert "--partition=gpu_a100" in command and "--cpus-per-task=12" in command
    with pytest.raises(SystemExit, match="neither"):
        campaign.submit(destination, "train", 2_000_000, dry_run=True, partition="rome")


def test_test_only_checks_each_command_without_submitting(tmp_path, monkeypatch, capsys) -> None:
    destination = _prepared(tmp_path, seeds=1)
    calls = []

    class Result:
        returncode, stdout, stderr = 0, "", "sbatch: Job 1 to start at 2026-10-22T18:00:00 using 18 processors\n"

    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs: calls.append(command) or Result())
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: pytest.fail("must not submit"))
    campaign.submit(destination, "train", 2_000_000, dry_run=False, test_only=True)
    assert len(calls) == 1 and calls[0][:3] == ["sbatch", "--parsable", "--test-only"]
    assert not (destination / "logs" / "submissions.jsonl").exists()
    assert "Job 1 to start" in capsys.readouterr().out


def test_reference_jobs_claim_shards_and_take_over_what_is_left(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "suite"
    campaign.prepare(destination, "screen", "full", 1, bks_from=_bks(tmp_path))
    tasks = campaign.reference_tasks(campaign.read_manifest(destination))
    shards = destination / "references" / "shards"
    shards.mkdir(parents=True)

    def stem(task):
        name, split, shard = tasks[task]
        return shards / f"{name}_{split}_{shard:03d}"

    done = []

    def fake_run(directory, task, workers):
        assert stem(task).with_suffix(".claim").exists()
        stem(task).with_suffix(".json").write_text("[]")
        done.append(task)

    monkeypatch.setattr(campaign, "run_references", fake_run)
    for task in range(3, len(tasks)):
        if task not in (5, 7):
            stem(task).with_suffix(".json").write_text("[]")
    stem(7).with_suffix(".claim").write_text("other job")  # another job is on it
    campaign.reference_pack(destination, [2, 0], workers=4)
    # Own share first, then the unclaimed rest; the claimed shard is left alone.
    assert done == [2, 0, 1, 5]
    assert [path.name for path in shards.glob("*.claim")] == [stem(7).with_suffix(".claim").name]
    assert not stem(7).with_suffix(".json").exists()
    # A stale claim (its job died) is taken over.
    old = time.time() - 4000
    os.utime(stem(7).with_suffix(".claim"), (old, old))
    campaign.reference_pack(destination, [], workers=4)
    assert done[-1] == 7 and not any(shards.glob("*.claim"))
    # Without stealing, only the job's own share runs.
    stem(2).with_suffix(".json").unlink()
    stem(3).with_suffix(".json").unlink()
    campaign.reference_pack(destination, [3], workers=4, steal=False)
    assert done[-1] == 3 and not stem(2).with_suffix(".json").exists()


def test_a_shard_claim_is_exclusive(tmp_path) -> None:
    claim = tmp_path / "x.claim"
    assert campaign.claim_shard(claim) and not campaign.claim_shard(claim)
    assert campaign.claim_shard(claim, stale_seconds=0)
