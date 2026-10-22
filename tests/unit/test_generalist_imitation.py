"""Imitation for the generalist policy: demonstrations, retained imitation, DAgger, resume."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
import yaml
from generalist_harness import small_config

from algo import generalist_ppo
from algo.canonical_ppo import PPOConfig
from algo.generalist_imitation import (
    ImitationConfig,
    ImitationData,
    demonstration_episodes,
    peek_imitation_state,
    read_rows,
    replay_dagger_archive,
    trainable_archive,
)
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from algo.generalist_ppo import (
    GeneralistPPO,
    ReferenceTable,
    SuiteProblem,
    SuiteReward,
    replay_demonstrations,
    run_reference_episode,
    run_reference_episodes,
)


COMPONENTS = ("static", "hos", "same_day")
SEEDS = list(range(6))
# Large enough that every charger is a candidate, so no teacher action is a miss.
POLICY = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, charger_budget=10_000)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    folder = tmp_path_factory.mktemp("suite")
    problems = []
    for component in COMPONENTS:
        path = folder / f"{component}.yaml"
        path.write_text(yaml.safe_dump(small_config(component, trucks=2, stops=4)))
        problems.append(SuiteProblem(component, str(path)))
    rows = {problem.name: {row["seed"]: row for row in run_reference_episodes(problem, SEEDS)} for problem in problems}
    return problems, rows


def _trainer(suite, imitation=None, *, workers=1, updates=2, seed=3, policy=POLICY) -> GeneralistPPO:
    problems, rows = suite
    config = PPOConfig(
        total_timesteps=3 * 8 * updates, num_envs=3, rollout_steps=8, minibatch_size=12, epochs=2, seed=seed,
    )
    return GeneralistPPO(
        policy, config, problems, ReferenceTable(rows), SuiteReward(reference="scenario"), [SEEDS] * 3,
        rollout_workers=workers, imitation=imitation,
    )


def _demonstrations(suite, count=2):
    problems, rows = suite
    episodes, _ = demonstration_episodes(problems, [SEEDS] * 3, count, references=ReferenceTable(rows))
    observations, labels, _ = replay_demonstrations(problems, POLICY.observation_spec, episodes)
    return observations, labels


# ---------------------------------------------------------------------------
# Demonstrations and expert outcomes
# ---------------------------------------------------------------------------


def test_a_demonstrations_folder_is_read_per_problem_with_reference_fallback(suite, tmp_path) -> None:
    problems, rows = suite
    static, hos = rows["static"], rows["hos"]
    (tmp_path / "static").mkdir()
    (tmp_path / "hos").mkdir()
    # A list of rows, one of them failed and one for a seed outside the train list.
    listed = [{**static[4], "failed": True}, {**static[5], "failed": False}, {**static[3], "failed": False},
              {**static[2], "seed": 99}]
    (tmp_path / "static" / "train_demonstrations.json").write_text(json.dumps(listed))
    # A seed-keyed dict whose rows carry only actions and the summary nested under "best".
    keyed = {str(seed): {"best": {**{k: v for k, v in hos[seed].items() if k != "seed"}, "failed": False}}
             for seed in (1, 0)}
    (tmp_path / "hos" / "train_demonstrations.json").write_text(json.dumps(keyed))

    episodes, sources = demonstration_episodes(
        problems, [SEEDS] * 3, 2, demonstrations=tmp_path, references=ReferenceTable(rows),
    )
    chosen = {(problems[position].name, seed): actions for position, seed, actions in episodes}
    # Train-seed order, failed rows skipped, foreign seeds ignored, at most two per problem.
    assert [seed for name, seed in chosen if name == "static"] == [3, 5]
    assert [seed for name, seed in chosen if name == "hos"] == [0, 1]
    assert chosen[("hos", 1)] == hos[1]["actions"]
    usable = [seed for seed in SEEDS if not rows["same_day"][seed]["failed"]]
    assert [seed for name, seed in chosen if name == "same_day"] == usable[:2]
    assert sources["static"]["failed_skipped"] == 1 and sources["static"]["source"].endswith("train_demonstrations.json")
    assert sources["same_day"]["source"] == "references"

    # Validation or test seeds in a demonstrations file are refused.
    (tmp_path / "static" / "train_demonstrations.json").write_text(json.dumps([{**static[3], "seed": 2_000_000_007}]))
    with pytest.raises(ValueError, match="train namespace"):
        demonstration_episodes(problems, [SEEDS] * 3, 2, demonstrations=tmp_path, references=ReferenceTable(rows))
    (tmp_path / "x.json").write_text(json.dumps({"rows": [{"actions": [1]}]}))
    with pytest.raises(ValueError, match="no seed"):
        read_rows(tmp_path / "x.json")


def test_references_accept_the_bks_folder_for_paired_rewards(suite, tmp_path) -> None:
    problems, rows = suite
    for problem in problems:
        folder = tmp_path / problem.name
        folder.mkdir()
        expert = {str(seed): {**row, "hours": row["hours"] * 0.5} for seed, row in rows[problem.name].items()}
        (folder / "train_demonstrations.json").write_text(json.dumps(expert))
        best = {"method": "x", "success": True, "failed": False, "dynamic": False, "tasks": 4, "served": 4,
                "late": 0, "lost": 0, "unserved": 0, "hours": 7.0}
        (folder / "validation.json").write_text(json.dumps(
            {"1000000003": {"best": best, "methods": {"x": best}, "bound": {"kind": "none", "value": 0.0}}}
        ))
    table = ReferenceTable.load(tmp_path, problems)
    paired, fallback = table.scale("static", 2, "scenario")
    assert paired.hours == pytest.approx(rows["static"][2]["hours"] * 0.5) and not fallback
    assert table.scale("static", 1_000_000_003, "scenario")[0].hours == pytest.approx(7.0)
    # A merged references file still wins over a folder of the same name.
    (tmp_path / "static.json").write_text(json.dumps({str(seed): row for seed, row in rows["static"].items()}))
    assert ReferenceTable.load(tmp_path, problems).scale("static", 2, "scenario")[0].hours == pytest.approx(
        rows["static"][2]["hours"]
    )


# ---------------------------------------------------------------------------
# Behaviour cloning and retained imitation
# ---------------------------------------------------------------------------


def test_behaviour_cloning_loss_falls_on_a_tiny_config(suite) -> None:
    observations, labels = _demonstrations(suite, count=3)
    trainer = _trainer(suite)
    try:
        history = trainer.pretrain(
            observations, labels, epochs=8, learning_rate=3e-3, batch_size=16, holdout_fraction=0.0,
        )
    finally:
        trainer.close()
    assert history[-1]["train_loss"] < 0.7 * history[0]["train_loss"]


def test_the_coefficient_anneals_linearly() -> None:
    config = ImitationConfig(bc_coefficient=1.0, bc_coefficient_final=0.2, bc_anneal_timesteps=100)
    assert [config.bc_weight(steps) for steps in (0, 50, 100, 500)] == pytest.approx([1.0, 0.6, 0.2, 0.2])
    assert ImitationConfig(bc_coefficient=0.3).bc_weight(10**9) == 0.3
    beta = ImitationConfig(dagger_beta=1.0, dagger_beta_final=0.0, dagger_beta_anneal_timesteps=10)
    assert beta.beta(5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ImitationConfig(dagger_beta=1.5)
    with pytest.raises(ValueError, match="module:factory"):
        ImitationConfig(dagger_expert=("not an expert",))
    table = ImitationConfig(dagger_expert=("teacher", "hos=none", "static=greedy_dispatch"))
    assert [table.expert_for(name) for name in COMPONENTS] == ["greedy_dispatch", None, "teacher"]


def _one_update(suite, imitation, demonstrations) -> GeneralistPPO:
    trainer = _trainer(suite, imitation)
    if demonstrations is not None:
        trainer.set_demonstrations(*demonstrations)
    trainer.learn(lambda record: record["update"] == 1)
    trainer.close()
    return trainer


def test_the_auxiliary_loss_changes_the_update_only_when_lambda_is_positive(suite) -> None:
    demonstrations = _demonstrations(suite)
    plain = _one_update(suite, None, None)
    zero = _one_update(suite, ImitationConfig(bc_coefficient=0.0), demonstrations)
    positive = _one_update(suite, ImitationConfig(bc_coefficient=0.5), demonstrations)
    for name, value in plain.policy.state_dict().items():
        assert torch.equal(value, zero.policy.state_dict()[name]), name
    assert any(
        not torch.equal(value, positive.policy.state_dict()[name]) for name, value in plain.policy.state_dict().items()
    )
    assert "bc_loss" not in zero.history[0] and zero.history[0]["bc_coefficient"] == 0.0
    assert positive.history[0]["bc_loss"] > 0 and positive.history[0]["bc_coefficient"] == 0.5

    # The gradient itself: the cross-entropy term reaches the action head and nothing else is needed.
    positive.policy.zero_grad()
    loss, _ = positive.imitation_loss()
    loss.backward()
    head = [parameter.grad for name, parameter in positive.policy.named_parameters() if name.startswith("action_head")]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in head)
    assert any(grad.abs().sum() > 0 for grad in head)
    assert all(
        parameter.grad is None or not parameter.grad.abs().sum()
        for name, parameter in positive.policy.named_parameters() if name.startswith("value_head")
    )


# ---------------------------------------------------------------------------
# DAgger
# ---------------------------------------------------------------------------


def test_dagger_with_the_teacher_labels_every_state_and_reproduces_it_at_beta_one(suite) -> None:
    problems, _ = suite
    imitation = ImitationConfig(
        dagger_expert=("teacher",), dagger_episodes=2, dagger_envs_per_problem=1, dagger_beta=1.0,
    )
    trainer = _trainer(suite, imitation, workers=2)
    try:
        record = trainer.dagger_round()
        archive = trainer.imitation_data.archive()
        observations, labels = trainer.imitation_data.flat()
        assert record["dagger/labelled"] == record["dagger/decisions"] == len(labels) > 0
        assert record["dagger/expert_executed_fraction"] == 1.0 and record["dagger/misses"] == {}
        # DAgger seeds come from the end of each train stream.
        assert sorted((row["problem"], row["seed"]) for row in archive) == sorted(
            (problem.name, seed) for problem in problems for seed in (5, 4)
        )
        by_name = {problem.name: problem for problem in problems}
        for row in archive:
            teacher = run_reference_episode(by_name[row["problem"]], row["seed"])["actions"]
            assert row["labels"] == row["actions"] == teacher
        executed = [action for row in archive for action in row["actions"]]
        assert [int(o.action_index[label]) for o, label in zip(observations, labels, strict=True)] == executed

        # Mixed control: still one label per visited state, and both policies act.
        mixed = trainer.dagger_round(beta=0.5)
        rows = trainer.imitation_data.archive()[len(archive):]
        assert mixed["dagger/labelled"] == mixed["dagger/decisions"]
        assert all(label >= 0 for row in rows for label in row["labels"])
        assert 0.0 < mixed["dagger/expert_executed_fraction"] < 1.0
        assert {f"dagger/{name}/agreement" for name in COMPONENTS} <= set(mixed)
        assert trainer.dagger_rounds == 2
    finally:
        trainer.close()


BUILT = []


def declining_teacher(problem, env):
    """A ``module:factory`` expert: the greedy teacher, declining every third state."""
    from algo.generalist_ppo import build_teacher

    BUILT.append(problem.name)
    teacher, calls = build_teacher(problem.teacher, env), [0]

    def act(env):
        calls[0] += 1
        if calls[0] % 3 == 0:
            raise RuntimeError("declined")
        return teacher(env)

    return act


def test_an_imported_expert_is_built_per_episode_and_may_decline_states(suite) -> None:
    BUILT.clear()
    imitation = ImitationConfig(
        dagger_expert=("test_generalist_imitation:declining_teacher",), dagger_episodes=2,
        dagger_envs_per_problem=1, dagger_beta=1.0,
    )
    trainer = _trainer(suite, imitation)
    try:
        record = trainer.dagger_round()
    finally:
        trainer.close()
    assert sorted(BUILT) == sorted(name for name in COMPONENTS for _ in range(2))
    rows = trainer.imitation_data.archive()
    declined = sum(label < 0 for row in rows for label in row["labels"])
    assert declined == record["dagger/expert_errors"] > 0
    assert record["dagger/labelled"] == record["dagger/decisions"] - declined == len(trainer.imitation_data)
    with pytest.raises(ModuleNotFoundError):
        _trainer(suite, ImitationConfig(dagger_expert=("no_such_module:factory",)))


def test_the_dagger_set_keeps_its_newest_episodes() -> None:
    data = ImitationData(max_dagger_states=5)
    data.set_demonstrations(["d"], [0])
    for index, size in enumerate((3, 2, 4)):
        data.add_episode({"seed": index}, ["o"] * size, [1] * size)
    assert [row["seed"] for row in data.archive()] == [2] and data.dagger_states == 4 and len(data) == 5
    observations, labels = data.sample(np.random.default_rng(0), 64)
    assert len(observations) == 64 and set(labels.tolist()) == {0, 1}


def test_checkpoint_and_resume_rebuild_the_dagger_set(suite, tmp_path) -> None:
    problems, _ = suite
    imitation = ImitationConfig(
        bc_coefficient=0.5, dagger_expert=("teacher",), dagger_every=1, dagger_episodes=1,
        dagger_envs_per_problem=1, dagger_beta=0.5,
    )
    demonstrations = _demonstrations(suite)
    first = _trainer(suite, imitation, updates=3)
    first.set_demonstrations(*demonstrations)
    first.learn(lambda record: record["update"] == 2)
    first.save_training(tmp_path / "training.pt")
    observations, labels = first.imitation_data.flat()
    positions = first.envs.dagger("positions")
    rng_state = first._bc_rng.bit_generator.state
    first.close()
    assert first.dagger_rounds == 2 and len(first.imitation_data.archive()) == 6

    saved = peek_imitation_state(tmp_path / "training.pt")
    archive = trainable_archive(saved, problems)
    replayed = replay_dagger_archive(problems, POLICY.observation_spec, archive, workers=2)
    for prebuilt in (replayed, None):  # rebuilt before forking, or in process
        second = _trainer(suite, imitation, updates=3)
        second.set_demonstrations(*demonstrations)
        second.load_training(tmp_path / "training.pt", dagger_replay=prebuilt)
        rebuilt, rebuilt_labels = second.imitation_data.flat()
        assert np.array_equal(rebuilt_labels, labels)
        for live, again in zip(observations, rebuilt, strict=True):
            assert np.array_equal(live.edges, again.edges) and np.array_equal(live.actions, again.actions)
            assert np.array_equal(live.action_index, again.action_index)
        assert second.envs.dagger("positions") == positions and second.dagger_rounds == 2
        assert second._bc_rng.bit_generator.state == rng_state
        if prebuilt is None:
            history = second.learn()
            assert [record["update"] for record in history] == [1, 2, 3]
            assert history[-1]["dagger/round"] == 3 and history[-1]["bc_loss"] > 0
        second.close()

    # The old format still loads both ways.
    plain = _trainer(suite, None, updates=3)
    plain.load_training(tmp_path / "training.pt")
    plain.save_training(tmp_path / "plain.pt")
    plain.close()
    assert peek_imitation_state(tmp_path / "plain.pt") is None
    third = _trainer(suite, imitation, updates=3)
    third.load_training(tmp_path / "plain.pt")
    assert len(third.imitation_data) == 0 and third.dagger_rounds == 0
    third.close()


def test_workers_with_dagger_slots_fork_before_the_policy_moves(suite, monkeypatch) -> None:
    events = []
    original_init = generalist_ppo.SuiteVecEnv.__init__
    original_to = GeneralistActorCritic.to

    def recording_init(self, *args, **kwargs):
        events.append(("fork", len(kwargs.get("dagger_slots") or ())))
        original_init(self, *args, **kwargs)

    def recording_to(self, *args, **kwargs):
        events.append(("to", str(args[0])))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(generalist_ppo.SuiteVecEnv, "__init__", recording_init)
    monkeypatch.setattr(GeneralistActorCritic, "to", recording_to)
    imitation = ImitationConfig(dagger_expert=("teacher", "hos=none"), dagger_envs_per_problem=2)
    trainer = _trainer(suite, imitation, workers=2)
    try:
        assert events == [("fork", 4), ("to", "cpu")]
        assert sorted({trainer.problems[slot.problem].name for slot in trainer.dagger_slots}) == ["same_day", "static"]
    finally:
        trainer.close()


def test_the_trainer_cli_clones_demonstrations_runs_dagger_and_resumes(suite, tmp_path, monkeypatch) -> None:
    from scripts.training import train_generalist_ppo as cli

    problems, rows = suite
    (tmp_path / "problems.json").write_text(json.dumps([asdict(problem) for problem in problems]))
    references = tmp_path / "references"
    references.mkdir()
    for name, values in rows.items():
        (references / f"{name}.json").write_text(json.dumps({str(seed): row for seed, row in values.items()}))
    demonstrations = tmp_path / "bks" / "static"
    demonstrations.mkdir(parents=True)
    (demonstrations / "train_demonstrations.json").write_text(json.dumps(list(rows["static"].values())))
    common = [
        "--problems", str(tmp_path / "problems.json"), "--references", str(references),
        "--num-envs", "3", "--rollout-steps", "4", "--minibatch-size", "6", "--epochs", "1",
        "--hidden-dim", "16", "--encoder-layers", "1", "--charger-budget", "10000",
        "--bc-episodes", "2", "--pretrain-epochs", "1", "--demonstrations", str(tmp_path / "bks"),
        "--bc-coefficient", "0.5", "--bc-anneal-timesteps", "100", "--bc-batch-size", "8",
        "--dagger-expert", "teacher", "hos=none", "--dagger-every", "2", "--dagger-episodes", "1",
        "--dagger-envs-per-problem", "1", "--dagger-pretrain-rounds", "1", "--dagger-pretrain-epochs", "1",
        "--train-scenarios", "6", "--validation-scenarios", "1", "--validate-every", "100",
        "--rollout-workers", "2", "--output", str(tmp_path / "training"), "--run-name", "cli",
    ]
    cli.main([*common, "--total-timesteps", "24"])
    run = tmp_path / "training" / "cli"
    history = json.loads((run / "training_history.json").read_text())
    pretrain = json.loads((run / "pretrain_history.json").read_text())
    assert [record["update"] for record in history] == [1, 2]
    assert history[0]["dagger/round"] == 2 and "dagger/round" not in history[1]
    assert history[1]["bc_coefficient"] == pytest.approx(0.5 * (1 - 24 / 100))
    assert pretrain["sources"]["static"]["source"].endswith("train_demonstrations.json")
    assert len(pretrain["dagger_rounds"]) == 1 and pretrain["coverage"]["misses"] == {}
    assert json.loads((run / "run_config.json").read_text())["imitation"]["dagger_expert"] == ["teacher", "hos=none"]

    # On resume the demonstrations and the DAgger archive are replayed before any worker is forked.
    events = []

    def record(name, function):
        def wrapper(*args, **kwargs):
            events.append(name)
            return function(*args, **kwargs)
        return wrapper

    for name in ("replay_demonstrations", "replay_dagger_archive", "evaluation_envs", "GeneralistPPO"):
        monkeypatch.setattr(cli, name, record(name, getattr(cli, name)))
    original_to = GeneralistActorCritic.to
    monkeypatch.setattr(GeneralistActorCritic, "to", lambda self, *a, **k: events.append("to") or original_to(self, *a, **k))
    cli.main([*common, "--total-timesteps", "36", "--resume", str(run / "training.pt")])
    assert events == ["replay_demonstrations", "replay_dagger_archive", "evaluation_envs", "GeneralistPPO", "to"]
    history = json.loads((run / "training_history.json").read_text())
    assert [record["update"] for record in history] == [1, 2, 3]
    assert history[2]["dagger/round"] == 3 and history[2]["bc_loss"] > 0
