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
    SuiteProblem,
    SuiteReward,
    SuiteVecEnv,
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
