"""Mixed-problem PPO plumbing: reward identity, references, replay, resume."""

import numpy as np
import pytest
import torch
import yaml
from generalist_harness import small_config

from algo.canonical_ppo import PPOConfig
from algo.generalist_observation import ObservationSpec
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from algo.generalist_ppo import (
    GeneralistPPO,
    ReferenceTable,
    RegretReferenceTable,
    SuiteProblem,
    SuiteReward,
    SuiteVecEnv,
    balance_advantages,
    balanced_slots,
    evaluate_policy_suite,
    replay_demonstrations,
    run_reference_episodes,
)


COMPONENTS = ("static", "hos", "same_day")


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    folder = tmp_path_factory.mktemp("suite")
    problems = []
    for component in COMPONENTS:
        path = folder / f"{component}.yaml"
        path.write_text(yaml.safe_dump(small_config(component, trucks=2, stops=4)))
        problems.append(SuiteProblem(component, str(path)))
    rows = {problem.name: {row["seed"]: row for row in run_reference_episodes(problem, range(6))} for problem in problems}
    return problems, rows


def _policy() -> GeneralistActorCritic:
    torch.manual_seed(0)
    return GeneralistActorCritic(GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4))


@pytest.mark.parametrize("mode", ["problem", "scenario"])
def test_the_return_of_an_episode_is_its_score(suite, mode) -> None:
    problems, rows = suite
    reward = SuiteReward(reference=mode)
    envs = SuiteVecEnv(problems, [(index, [1]) for index in range(len(problems))],
                       ObservationSpec(charger_budget=4), reward, ReferenceTable(rows), cycle=False)
    generator = np.random.default_rng(0)
    observations, totals, records = envs.reset(), np.zeros(len(problems)), []
    while any(observation is not None for observation in observations):
        actions = [
            None if observation is None else int(generator.choice(observation.action_index))
            for observation in observations
        ]
        observations, rewards, _, finished = envs.step(actions)
        totals += rewards
        records.extend((index, record) for index, record in enumerate(finished) if record is not None)
    envs.close()
    assert len(records) == len(problems)
    for index, record in records:
        assert totals[index] == pytest.approx(record["score"], abs=1e-4)


def test_scenario_references_are_paired_and_fall_back_to_the_median(suite) -> None:
    problems, rows = suite
    table = ReferenceTable(rows)
    name = problems[0].name
    paired, fallback = table.scale(name, 2, "scenario")
    assert paired.hours == pytest.approx(rows[name][2]["hours"]) and not fallback
    median, fallback = table.scale(name, 10**6, "scenario")
    assert median == table.medians[name] and fallback
    assert table.scale(name, 2, "problem") == (table.medians[name], False)


def test_slots_balance_problems_on_disjoint_seed_slices() -> None:
    seeds = [list(range(100, 112)), list(range(200, 212))]
    slots = balanced_slots(2, 6, seeds)
    assert [problem for problem, _ in slots] == [0, 1, 0, 1, 0, 1]
    for problem in (0, 1):
        streams = [set(stream) for owner, stream in slots if owner == problem]
        assert set().union(*streams) == set(seeds[problem])
        assert sum(map(len, streams)) == len(seeds[problem])
    with pytest.raises(ValueError, match="multiple"):
        balanced_slots(2, 5, seeds)
    with pytest.raises(ValueError, match="at least"):
        balanced_slots(2, 6, [[1], [2]])


def test_replay_rebuilds_the_teacher_decisions(suite) -> None:
    problems, rows = suite
    spec = ObservationSpec(charger_budget=10_000)
    episodes = [(index, seed, rows[problem.name][seed]["actions"]) for index, problem in enumerate(problems) for seed in (0, 1)]
    observations, labels, coverage = replay_demonstrations(problems, spec, episodes)
    assert coverage["misses"] == {} and coverage["kept"] == coverage["steps"] == len(labels)
    recorded = [action for _, _, actions in episodes for action in actions]
    assert [int(o.action_index[label]) for o, label in zip(observations, labels, strict=True)] == recorded


def test_evaluation_runs_every_seed_once(suite) -> None:
    problems, rows = suite
    seeds = [[0, 1, 2], [3], [4, 5]]
    found = evaluate_policy_suite(_policy(), problems, seeds, ReferenceTable(rows), SuiteReward(), envs_per_problem=2)
    assert sorted((row["problem"], row["seed"]) for row in found) == sorted(
        (problem.name, seed) for problem, stream in zip(problems, seeds, strict=True) for seed in stream
    )


def test_balanced_advantages_standardize_each_problem() -> None:
    generator = torch.Generator().manual_seed(0)
    advantages = torch.randn(32, 4, generator=generator)
    advantages[:, [0, 2]] = advantages[:, [0, 2]] * 50.0 + 10.0
    advantages[:, [1, 3]] = advantages[:, [1, 3]] * 0.1 - 3.0
    balanced = balance_advantages(advantages, [0, 1, 0, 1])
    for columns in ([0, 2], [1, 3]):
        group = balanced[:, columns]
        assert group.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert group.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="problem ids"):
        balance_advantages(advantages, [0, 1, 0])


def test_balanced_advantages_fall_back_to_global_statistics_on_small_groups() -> None:
    generator = torch.Generator().manual_seed(1)
    advantages = torch.randn(4, 4, generator=generator)  # 8 samples per problem < 32
    balanced = balance_advantages(advantages, [0, 1, 0, 1])
    whole = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    assert torch.allclose(balanced, whole)


def test_regret_reference_table_measures_the_bks_teacher_spread() -> None:
    static = {"tasks": 30, "dynamic": False, "served": 30}
    dynamic = {"tasks": 40, "dynamic": True}
    teacher = ReferenceTable({
        "static_p": {0: {"hours": 300.0, **static}, 1: {"hours": 220.0, **static}},
        "dyn_p": {0: {"hours": 36.0, "served": 10, **dynamic}},
    })
    bks = ReferenceTable({
        "static_p": {0: {"hours": 100.0, **static}, 1: {"hours": 120.0, **static}, 2: {"hours": 110.0, **static}},
        "dyn_p": {0: {"hours": 36.0, "served": 25, **dynamic}},
        "lonely": {0: {"hours": 50.0, **static}},
    })
    table = RegretReferenceTable(teacher=teacher, bks=bks)
    paired, fallback = table.scale("static_p", 0, "scenario")
    assert paired.hours == pytest.approx(200.0) and paired.tasks == 30.0 and not fallback
    # Teacher and BKS hours coincide: the floor of 10 % of the BKS hours holds the unit.
    served, fallback = table.scale("dyn_p", 0, "scenario")
    assert served.hours == pytest.approx(3.6) and served.tasks == pytest.approx(15.0) and not fallback
    # A seed missing from the teacher table takes the median spread, flagged in scenario mode.
    median, fallback = table.scale("static_p", 2, "scenario")
    assert median == table.spread_medians["static_p"] and fallback
    assert table.scale("static_p", 0, "problem") == (table.spread_medians["static_p"], False)
    # A problem with no overlapping seeds keeps the plain BKS units.
    lonely, fallback = table.scale("lonely", 0, "scenario")
    assert lonely.hours == pytest.approx(50.0) and lonely.tasks == 30.0 and not fallback


def test_the_return_equals_the_score_with_regret_units(suite) -> None:
    problems, rows = suite
    better = {
        name: {seed: {**row, "hours": row["hours"] * 0.6} for seed, row in values.items()}
        for name, values in rows.items()
    }
    references = RegretReferenceTable(teacher=ReferenceTable(rows), bks=ReferenceTable(better))
    reward = SuiteReward(reference="scenario")
    envs = SuiteVecEnv(problems, [(index, [1]) for index in range(len(problems))],
                       ObservationSpec(charger_budget=4), reward, references, cycle=False)
    generator = np.random.default_rng(0)
    observations, totals, records = envs.reset(), np.zeros(len(problems)), []
    while any(observation is not None for observation in observations):
        actions = [
            None if observation is None else int(generator.choice(observation.action_index))
            for observation in observations
        ]
        observations, rewards, _, finished = envs.step(actions)
        totals += rewards
        records.extend((index, record) for index, record in enumerate(finished) if record is not None)
    envs.close()
    assert len(records) == len(problems)
    for index, record in records:
        assert totals[index] == pytest.approx(record["score"], abs=1e-4)


def test_ppo_updates_checkpoints_and_resumes(suite, tmp_path) -> None:
    problems, rows = suite
    config = PPOConfig(total_timesteps=3 * 2 * 8 * 3, num_envs=6, rollout_steps=8, minibatch_size=16, epochs=2, seed=3)
    policy_config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4, fleet_assignment=True)
    reward = SuiteReward(reference="scenario")

    def trainer():
        return GeneralistPPO(policy_config, config, problems, ReferenceTable(rows), reward, [list(range(6))] * 3)

    first = trainer()
    first.learn(lambda record: record["update"] == 2)
    assert first.completed_updates == 2 and first.total_steps == 2 * 48
    first.save_training(tmp_path / "training.pt", {"note": 1})
    positions = first.envs.positions()[0]
    first.close()

    second = trainer()
    assert second.load_training(tmp_path / "training.pt") == {"note": 1}
    assert second.envs.positions()[0] == positions
    assert torch.equal(
        next(iter(second.policy.state_dict().values())), next(iter(first.policy.state_dict().values()))
    )
    history = second.learn()
    second.close()
    assert [record["update"] for record in history] == [1, 2, 3]
    assert all(np.isfinite(record["policy_loss"]) for record in history)


def test_the_trainer_records_throughput_in_its_history(suite, tmp_path) -> None:
    import json
    from dataclasses import asdict

    from scripts.training.train_generalist_ppo import main

    problems, rows = suite
    (tmp_path / "problems.json").write_text(json.dumps([asdict(problem) for problem in problems]))
    for name, values in rows.items():
        (tmp_path / f"{name}.json").write_text(json.dumps({str(seed): row for seed, row in values.items()}))
    main([
        "--problems", str(tmp_path / "problems.json"), "--references", str(tmp_path),
        "--num-envs", "3", "--rollout-steps", "4", "--minibatch-size", "6", "--epochs", "1",
        "--hidden-dim", "16", "--encoder-layers", "1", "--charger-budget", "4", "--bc-episodes", "0",
        "--train-scenarios", "6", "--validation-scenarios", "1", "--validate-every", "100",
        "--total-timesteps", "24", "--output", str(tmp_path / "training"), "--run-name", "cli",
    ])
    history = json.loads((tmp_path / "training" / "cli" / "training_history.json").read_text())
    assert [record["update"] for record in history] == [1, 2]
    assert all(record["decisions_per_second"] > 0 for record in history)
    assert all("gpu_peak_gb" not in record for record in history)


def test_ppo_learns_with_balanced_advantages(suite) -> None:
    problems, rows = suite
    config = PPOConfig(total_timesteps=6 * 8, num_envs=6, rollout_steps=8, minibatch_size=16, epochs=1, seed=5)
    trainer = GeneralistPPO(
        GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4),
        config, problems, ReferenceTable(rows), SuiteReward(), [list(range(6))] * 3,
        balanced_advantages=True,
    )
    history = trainer.learn()
    trainer.close()
    assert len(history) == 1 and np.isfinite(history[0]["policy_loss"])


def test_the_trainer_refuses_regret_normalization_without_reward_references(suite, tmp_path) -> None:
    import json
    from dataclasses import asdict

    from scripts.training.train_generalist_ppo import main

    problems, rows = suite
    (tmp_path / "problems.json").write_text(json.dumps([asdict(problem) for problem in problems]))
    for name, values in rows.items():
        (tmp_path / f"{name}.json").write_text(json.dumps({str(seed): row for seed, row in values.items()}))
    with pytest.raises(SystemExit, match="needs --reward-references"):
        main([
            "--problems", str(tmp_path / "problems.json"), "--references", str(tmp_path),
            "--regret-normalized-reward", "--output", str(tmp_path / "training"), "--run-name", "refused",
        ])
