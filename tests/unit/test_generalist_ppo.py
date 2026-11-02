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


def test_group_slots_run_each_seed_slice_on_consecutive_sub_environments() -> None:
    seeds = [list(range(100, 112)), list(range(200, 212))]
    slots = balanced_slots(2, 8, seeds, group_size=2)
    assert [problem for problem, _ in slots] == [0, 0, 1, 1, 0, 0, 1, 1]
    assert all(slots[index] == slots[index + 1] for index in range(0, 8, 2))
    for problem in (0, 1):
        streams = [set(stream) for owner, stream in slots[::2] if owner == problem]
        assert set().union(*streams) == set(seeds[problem]) and sum(map(len, streams)) == len(seeds[problem])
    assert balanced_slots(2, 4, seeds, group_size=1) == balanced_slots(2, 4, seeds)
    with pytest.raises(ValueError, match="times the group size"):
        balanced_slots(2, 6, seeds, group_size=4)
    with pytest.raises(ValueError, match="group_size > 1"):
        PPOConfig(advantage_mode="group")


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


def test_advantages_are_normalized_within_each_problem() -> None:
    from algo.generalist_ppo import normalized_by_group

    advantages = torch.tensor([[100.0, 0.001, 300.0, 0.003], [200.0, 0.002, 400.0, 0.004]])
    normalized = normalized_by_group(advantages, torch.tensor([0, 1, 0, 1]))
    for rows in ([0, 2], [1, 3]):
        part = normalized[:, rows]
        assert abs(float(part.mean())) < 1e-5
        assert abs(float(part.std(unbiased=False)) - 1.0) < 1e-3


def test_group_advantages_are_leave_one_out_returns_with_a_gae_fallback() -> None:
    from algo.generalist_ppo import group_advantages

    gae = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    rewards = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [5.0, 2.0], [5.0, 9.0], [5.0, 9.0]])
    # Both sub-environments finish seed 7 (returns 3 and 8); the first then
    # finishes seed 8, whose partner is still running at the rollout's end.
    finished = [(2, 0, {"problem": "p", "seed": 7}), (3, 1, {"problem": "p", "seed": 7}), (5, 0, {"problem": "p", "seed": 8})]
    advantages, counts = group_advantages(gae, rewards, finished, [True, True], 2)
    assert advantages[:3, 0].tolist() == [-5.0] * 3 and advantages[:4, 1].tolist() == [5.0] * 4
    assert torch.equal(advantages[3:, 0], gae[3:, 0]) and torch.equal(advantages[4:, 1], gae[4:, 1])
    assert counts == {"group/episodes": 2, "group/fallback_steps": 5, "group/fallback_fraction": 5 / 12}
    # The second sub-environment's step-0 state continues an earlier episode: nothing pairs.
    advantages, counts = group_advantages(gae, rewards, finished, [True, False], 2)
    assert torch.equal(advantages, gae) and counts["group/fallback_steps"] == 12


def test_group_mode_pairs_the_rollouts_by_seed_and_logs_its_fallback(suite) -> None:
    problems, rows = suite
    config = PPOConfig(
        total_timesteps=6 * 32, num_envs=6, rollout_steps=32, minibatch_size=64, epochs=1, seed=3,
        group_size=2, advantage_mode="group",
    )
    policy_config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4)
    trainer = GeneralistPPO(policy_config, config, problems, ReferenceTable(rows), SuiteReward(), [list(range(6))] * 3)
    assert trainer.advantage_groups.tolist() == [0, 0, 1, 1, 2, 2]
    history = trainer.learn()
    trainer.close()
    record = history[-1]
    assert record["group/episodes"] >= 2 and 0.0 < record["group/fallback_fraction"] < 1.0
    assert record["group/fallback_steps"] == round(record["group/fallback_fraction"] * 6 * 32)
    assert np.isfinite(record["policy_loss"])


def test_a_zero_policy_gradient_weight_leaves_the_action_head_to_the_other_terms(suite) -> None:
    problems, rows = suite
    policy_config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4)

    def after_one_update(weight: float) -> tuple[bool, bool]:
        config = PPOConfig(
            total_timesteps=12, num_envs=3, rollout_steps=4, minibatch_size=12, epochs=1, seed=3,
            policy_gradient_weight=weight, entropy_coefficient=0.0,
        )
        trainer = GeneralistPPO(policy_config, config, problems, ReferenceTable(rows), SuiteReward(), [list(range(6))] * 3)
        before = {name: tensor.clone() for name, tensor in trainer.policy.state_dict().items()}
        record = trainer.learn()[-1]
        trainer.close()
        assert np.isfinite(record["policy_loss"]) and record["policy_loss"] != 0.0
        after = trainer.policy.state_dict()
        moved = lambda prefix: any(not torch.equal(before[name], after[name]) for name in before if name.startswith(prefix))
        return moved("action_head."), moved("value_head.")

    assert after_one_update(0.0) == (False, True)
    assert after_one_update(1.0) == (True, True)


def test_training_filters_need_the_mask_and_the_plan(monkeypatch) -> None:
    from types import SimpleNamespace

    from scripts.training import train_generalist_ppo as train

    for name in ("EVRP_SAFETY_MASK", "EVRP_SAFETY_MARGIN", "EVRP_RETIRE_GUARD", "EVRP_PUBLIC_WAKE"):
        monkeypatch.setenv(name, "0" if name.endswith("MARGIN") else "off")  # restored after the test
    with pytest.raises(SystemExit, match="need --safety-mask"):
        train.configure_filters(SimpleNamespace(safety_mask=False, safety_margin=None, retire_guard=True, plan_conditioned=True, public_wake=False))
    with pytest.raises(SystemExit, match="needs --plan-conditioned"):
        train.configure_filters(SimpleNamespace(safety_mask=True, safety_margin=None, retire_guard=True, plan_conditioned=False, public_wake=False))
    train.configure_filters(SimpleNamespace(safety_mask=True, safety_margin=0.0, retire_guard=True, plan_conditioned=True, public_wake=True))
    import os

    assert os.environ["EVRP_SAFETY_MASK"] == "on" and os.environ["EVRP_RETIRE_GUARD"] == "on"
    assert os.environ["EVRP_PUBLIC_WAKE"] == "on"


def test_a_waiting_truck_shows_only_its_public_wake_up(monkeypatch) -> None:
    from types import SimpleNamespace

    from EVRoutingEnv.models.core.requests import public_wake_enabled
    from EVRoutingEnv.models.environment.request_engine import RequestEngine

    requests = [SimpleNamespace(reveal_time=1.0, release_time=1.2), SimpleNamespace(reveal_time=0.1, release_time=0.1)]
    engine = SimpleNamespace(_known_schedule=[1.0, 2.0], registry=SimpleNamespace(revealed=lambda: requests))
    assert RequestEngine.public_wake(engine, 0.3, 0.5, 0.4, 0.8) == 0.8  # the wait
    assert RequestEngine.public_wake(engine, 0.9, 0.5, 0.95, 1.4) == 1.0  # a scheduled boundary
    assert RequestEngine.public_wake(engine, 1.05, 0.5, 1.1, 1.5) == 1.2  # a known release
    # A boundary and a release that passed while the truck kept waiting were empty: public news.
    assert RequestEngine.public_wake(engine, 0.3, 5.0, 1.5, 2.0) == 2.0
    assert RequestEngine.public_wake(engine, 0.3, 5.0, 2.5, 5.3) == 5.3
    assert RequestEngine.public_wake(engine, 0.3, 5.0, 1.2, 2.0) == 1.2  # known news due this instant
    assert RequestEngine.public_wake(engine, 0.3, 5.0, 1.5, 1.5) == 1.5  # the wake-up itself is due
    assert public_wake_enabled() is False
    monkeypatch.setenv("EVRP_PUBLIC_WAKE", "on")
    assert public_wake_enabled() is True


def test_the_deployed_arms_warm_start_from_their_own_seed() -> None:
    from scripts.campaigns import generalist_graphppo as campaign

    runs = campaign.arms_for("named", "full", 2, bks="/b", arms=["ppo_deployed", "ppo_deployed_anchor"])
    assert [run["name"] for run in runs] == [
        "ppo_deployed_s0", "ppo_deployed_anchor_s0", "ppo_deployed_s1", "ppo_deployed_anchor_s1",
    ]
    for run in runs:
        flags = run["flags"]
        assert flags[flags.index("--init-from") + 1].endswith(f"full_dagger_plan_charge_s{run['seed']}")
        assert flags[flags.index("--plan-dropout", flags.index("--plan-dropout") + 1) + 1] == "0"
        assert "--retire-guard" in flags and "--safety-mask" in flags and "--advantage-by-problem" in flags
        assert "--public-wake" in flags
    assert "--dagger-expert" not in runs[0]["flags"] and "--dagger-expert" in runs[1]["flags"]


def test_the_family_critic_arm_warm_starts_and_a_mismatched_actor_is_refused() -> None:
    from scripts.campaigns import generalist_graphppo as campaign

    runs = campaign.arms_for("named", "full", 1, bks="/b", arms=["ppo_deployed_anchor_critics"])
    flags = runs[0]["flags"]
    assert flags[flags.index("--critic-heads") + 1] == "family" and "--init-from" in flags and "--dagger-expert" in flags


def test_the_variance_reduction_arms_override_the_common_flags() -> None:
    from scripts.campaigns import generalist_graphppo as campaign
    from scripts.training.train_generalist_ppo import build_argument_parser

    arms = ["ppo_deployed_anchor", "ppo_deployed_anchor_bigbatch", "ppo_deployed_anchor_group", "imitation_only_deployed"]
    parsed = {}
    for run in campaign.arms_for("named", "full", 1, bks="/b", arms=arms):
        assert "--init-from" in run["flags"] and "--dagger-expert" in run["flags"] and "--advantage-by-problem" in run["flags"]
        # The job runs COMMON first and the arm's flags last, so the arm's value wins.
        arguments = build_argument_parser().parse_args(["--references", "/r", *campaign.COMMON, *run["flags"]])
        parsed[run["arm"]] = (
            arguments.rollout_steps, arguments.minibatch_size, arguments.accumulate_minibatches, arguments.group_size,
            arguments.advantage_mode, arguments.policy_gradient_weight, arguments.rollout_on_cpu,
        )
    assert parsed["ppo_deployed_anchor"] == (128, 256, 1, 1, "gae", 1.0, False)
    assert parsed["ppo_deployed_anchor_bigbatch"] == (512, 256, 4, 1, "gae", 1.0, True)
    assert parsed["ppo_deployed_anchor_group"] == (512, 256, 4, 4, "group", 1.0, True)
    assert parsed["imitation_only_deployed"] == (128, 256, 1, 1, "gae", 0.0, False)


def test_accumulated_minibatches_take_the_step_of_one_big_minibatch(suite) -> None:
    """Four accumulated minibatches of 12 move the weights like one minibatch of 48."""
    problems, rows = suite
    reward = SuiteReward(reference="scenario")
    weights = []
    for minibatch, accumulate in ((48, 1), (12, 4)):
        config = PPOConfig(total_timesteps=6 * 8, num_envs=6, rollout_steps=8, minibatch_size=minibatch, epochs=1, seed=3,
                           accumulate_minibatches=accumulate, advantage_by_problem=True, target_kl=None)
        policy_config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4, fleet_assignment=True)
        trainer = GeneralistPPO(policy_config, config, problems, ReferenceTable(rows), reward, [list(range(6))] * 3)
        trainer.learn()
        weights.append({name: value.detach().clone() for name, value in trainer.policy.named_parameters()})
        trainer.close()
    for name in weights[0]:
        assert torch.allclose(weights[0][name], weights[1][name], atol=1e-5, rtol=1e-4), name


def test_a_rollout_kept_on_the_cpu_trains_like_one_on_the_device(suite) -> None:
    """``rollout_on_cpu`` only changes where the collated rollout lives: the updates are the same."""
    problems, rows = suite
    reward = SuiteReward(reference="scenario")
    histories = []
    for on_cpu in (False, True):
        config = PPOConfig(total_timesteps=2 * 6 * 8, num_envs=6, rollout_steps=8, minibatch_size=16, epochs=1, seed=3,
                           rollout_on_cpu=on_cpu)
        policy_config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=4, fleet_assignment=True)
        trainer = GeneralistPPO(policy_config, config, problems, ReferenceTable(rows), reward, [list(range(6))] * 3)
        histories.append(trainer.learn())
        trainer.close()
    assert [h["policy_loss"] for h in histories[0]] == pytest.approx([h["policy_loss"] for h in histories[1]])
    assert [h["value_loss"] for h in histories[0]] == pytest.approx([h["value_loss"] for h in histories[1]])
