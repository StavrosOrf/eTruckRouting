"""Imitation for the generalist policy: demonstrations, retained imitation, DAgger, resume."""

import json
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import torch
import yaml
from generalist_harness import small_config

from algo import generalist_ppo
from algo.canonical_ppo import PPOConfig
from algo.generalist_imitation import (
    DaggerSlots,
    ImitationConfig,
    ImitationData,
    dagger_slots,
    demonstration_episodes,
    episode_counts,
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


def _trainer(suite, imitation=None, *, workers=1, updates=2, seed=3, policy=POLICY, together=False, steps=8) -> GeneralistPPO:
    problems, rows = suite
    config = PPOConfig(
        total_timesteps=3 * steps * updates, num_envs=3, rollout_steps=steps, minibatch_size=12, epochs=2, seed=seed,
        step_together=together,
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


def test_a_decision_cap_keeps_whole_episodes_and_at_least_one(suite) -> None:
    problems, rows = suite
    references = ReferenceTable(rows)
    full, full_sources = demonstration_episodes(problems, [SEEDS] * 3, 6, references=references)
    lengths = {name: [len(actions) for position, _, actions in full if problems[position].name == name] for name in rows}
    name = problems[0].name
    cap = lengths[name][0] + lengths[name][1]  # exactly the first two episodes fit
    capped, sources = demonstration_episodes(problems, [SEEDS] * 3, 6, references=references, max_decisions=cap)
    taken = [len(actions) for position, _, actions in capped if problems[position].name == name]
    assert taken == lengths[name][:2] and sources[name]["decisions"] == cap
    assert all(sources[other]["decisions"] <= max(cap, lengths[other][0]) for other in rows)
    # A cap below one episode still keeps that first episode, whole.
    one, sources = demonstration_episodes(problems, [SEEDS] * 3, 6, references=references, max_decisions=1)
    assert [len(actions) for position, _, actions in one if problems[position].name == name] == lengths[name][:1]
    assert sources[name]["used"] == 1 and full_sources[name]["decisions"] == sum(lengths[name])


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
    loss, accuracy = positive.imitation_gradient()
    assert loss > 0 and 0.0 <= accuracy <= 1.0
    head = [parameter.grad for name, parameter in positive.policy.named_parameters() if name.startswith("action_head")]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in head)
    assert any(grad.abs().sum() > 0 for grad in head)
    assert all(
        parameter.grad is None or not parameter.grad.abs().sum()
        for name, parameter in positive.policy.named_parameters() if name.startswith("value_head")
    )


def test_minibatches_in_chunks_give_the_gradient_of_the_whole_minibatch(suite) -> None:
    """A pair budget that splits every minibatch (and its labelled batch) leaves the update unchanged."""
    from algo.generalist_ppo import PairBudget

    demonstrations = _demonstrations(suite)
    trainer = _trainer(suite, ImitationConfig(bc_coefficient=0.5, bc_batch_size=6))
    try:
        trainer.set_demonstrations(*demonstrations)
        trainer._observations = trainer.envs.reset()
        trainer._observe(trainer._observations)
        rollout = trainer._collect_rollout()
        assert len(rollout["observations"]) == 24 and rollout["counts"].shape == (24, 3)
        policy = {name: value.clone() for name, value in trainer.policy.state_dict().items()}
        optimizer = {key: value for key, value in trainer.optimizer.state_dict().items()}
        import copy

        optimizer = copy.deepcopy(optimizer)
        results = []
        for pairs in (None, 150):  # 150 pairs: two or three of these small graphs per chunk
            trainer.policy.load_state_dict(policy)
            trainer.optimizer.load_state_dict(copy.deepcopy(optimizer))
            trainer.budget = PairBudget(pairs)
            trainer._bc_rng = np.random.default_rng(7)
            np.random.seed(11)
            statistics = trainer._optimize(rollout)
            results.append((statistics, {name: value.clone() for name, value in trainer.policy.state_dict().items()}))
        (whole, whole_weights), (split, split_weights) = results
        assert len(trainer.budget.chunks(rollout["counts"][:12])) > 2
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "bc_loss", "bc_accuracy"):
            assert split[key] == pytest.approx(whole[key], rel=1e-4, abs=1e-6), key
        for name, value in whole_weights.items():
            assert torch.allclose(value, split_weights[name], atol=1e-5, rtol=1e-4), name
    finally:
        trainer.close()


def test_behaviour_cloning_in_chunks_matches_whole_minibatches(suite) -> None:
    from algo.generalist_ppo import PairBudget

    observations, labels = _demonstrations(suite, count=3)
    trainer = _trainer(suite)
    try:
        # The gradient of a labelled minibatch is the same whether it is scored whole or in chunks.
        results = []
        for pairs in (None, 150):
            trainer.budget = PairBudget(pairs)
            trainer.policy.zero_grad()
            sums = trainer._labelled_sums(observations, labels, 1.0 / len(labels))
            silent = trainer._labelled_sums(observations, labels, None)
            assert silent == pytest.approx(sums, rel=1e-5)
            results.append((sums, [None if p.grad is None else p.grad.clone() for p in trainer.policy.parameters()]))
        sizes = np.asarray([(len(o.trucks), len(o.tasks), len(o.chargers)) for o in observations])
        assert len(trainer.budget.chunks(sizes)) > 5
        (whole, whole_gradients), (split, split_gradients) = results
        assert split == pytest.approx(whole, rel=1e-5)
        used = 0
        for mine, theirs in zip(split_gradients, whole_gradients, strict=True):
            assert (mine is None) == (theirs is None)
            if mine is not None:
                used += 1
                assert torch.allclose(mine, theirs, atol=1e-6, rtol=1e-4)
        assert used > 4
    finally:
        trainer.close()
    # Cloning itself: the first minibatch scores the same, and training follows the same course
    # (Adam's default epsilon amplifies rounding noise, so later losses agree loosely).
    histories, statistics = [], []
    for pairs in (None, 150):
        trainer = _trainer(suite)
        try:
            trainer.budget = PairBudget(pairs)
            histories.append(trainer.pretrain(observations, labels, epochs=2, learning_rate=3e-3, batch_size=10_000))
            normalizer = trainer.policy.encoder.normalizer
            statistics.append((float(normalizer.context_count.item()), normalizer.context_mean.clone(),
                               normalizer.edge_mean.clone(), normalizer.edge_var.clone()))
        finally:
            trainer.close()
    # The input statistics fold every observation exactly once, in whatever chunks.
    assert statistics[0][0] == statistics[1][0] == len(observations)
    for whole, split in zip(statistics[0][1:], statistics[1][1:], strict=True):
        assert torch.allclose(whole, split, atol=1e-5, rtol=1e-3)
    assert histories[1][0]["train_loss"] == pytest.approx(histories[0][0]["train_loss"], rel=1e-4)
    for whole, split in zip(*histories, strict=True):
        assert split["train_loss"] == pytest.approx(whole["train_loss"], rel=1e-2)
        assert split["holdout_loss"] == pytest.approx(whole["holdout_loss"], rel=1e-2) and split["seconds"] > 0


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


def test_dagger_environments_are_built_only_for_the_slots_that_run_an_episode(suite) -> None:
    problems, _ = suite
    config = ImitationConfig(dagger_expert=("teacher",), dagger_episodes=1, dagger_envs_per_problem=4)
    slots = dagger_slots(problems, [list(range(8))] * len(problems), config)
    assert len(slots) == 4 * len(problems)
    workers = DaggerSlots(problems, slots, POLICY.observation_spec)
    try:
        assert all(env is None for env in workers.envs)
        live = workers.start(episode_counts(slots, config.dagger_episodes), 1.0, (0, 0))
        running = [index for index, item in enumerate(live) if item is not None]
        # One episode per problem: one slot of the four runs, and only it holds an environment.
        assert len(running) == len(problems)
        assert [index for index, env in enumerate(workers.envs) if env is not None] == running
        records = []
        while any(item is not None for item in live):
            actions = [None if item is None else int(item[0].action_index[item[1]]) for item in live]
            stepped = workers.step(actions)
            live = [None if result is None else result[0] for result in stepped]
            records.extend(result[1] for result in stepped if result is not None and result[1] is not None)
        assert sorted(record["problem"] for record in records) == sorted(problem.name for problem in problems)
        assert all(record["seconds"] > 0 and record["decisions"] > 0 for record in records)
        assert [index for index, env in enumerate(workers.envs) if env is not None] == running
    finally:
        workers.close()


def test_the_slots_a_dagger_round_runs_are_spread_over_the_workers(suite) -> None:
    imitation = ImitationConfig(dagger_expert=("teacher",), dagger_episodes=1, dagger_envs_per_problem=4, dagger_beta=1.0)
    trainer = _trainer(suite, imitation, workers=2)
    try:
        counts = episode_counts(trainer.dagger_slots, imitation.dagger_episodes)
        running = [index for index, count in enumerate(counts) if count]
        assert running == [0, 4, 8]
        spans = trainer.envs._dagger_spans
        assert sorted(index for span in spans for index in span) == list(range(12))
        # Listing order would put all three running slots on the first of the two workers.
        assert sorted(sum(index in span for index in running) for span in spans) == [1, 2]
        record = trainer.dagger_round()
        assert record["dagger/episodes"] == 3 and record["dagger/labelled"] == record["dagger/decisions"] > 0
        assert {row["problem"] for row in trainer.imitation_data.archive()} == set(COMPONENTS)
        assert all(record[f"dagger/{name}/seconds"] > 0 and record[f"dagger/{name}/decisions"] > 0 for name in COMPONENTS)
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Workers at their own pace
# ---------------------------------------------------------------------------


def _same_observations(mine, theirs) -> bool:
    return len(mine) == len(theirs) and all(
        np.array_equal(getattr(a, name), getattr(b, name))
        for a, b in zip(mine, theirs, strict=True) for name in ("trucks", "tasks", "chargers", "edges", "action_index")
    )


def test_worker_by_worker_and_lockstep_collect_the_same_rollouts(suite) -> None:
    collected = []
    for together in (True, False):
        trainer = _trainer(suite, workers=2, together=together, steps=40)
        try:
            assert len(generalist_ppo.Pace(trainer.envs, together=together).groups) == (1 if together else 2)
            trainer._observations = trainer.envs.reset()
            trainer._observe(trainer._observations)
            rollouts = [trainer._collect_rollout(), trainer._collect_rollout()]
            episodes = {name: [(row["seed"], row["score"]) for row in rows] for name, rows in trainer.episodes.items()}
            collected.append((rollouts, episodes, trainer.total_steps, trainer.reward_scale.scale, trainer._fresh.tolist()))
        finally:
            trainer.close()
    (lockstep, episodes, steps, scale, fresh), (alone, *others) = collected
    assert others[0] == episodes and sum(len(rows) for rows in episodes.values()) >= 3  # episodes ended and restarted
    assert others[1] == steps == 2 * 40 * 3 and others[3] == fresh
    assert others[2] == pytest.approx(scale, rel=1e-9)
    for mine, theirs in zip(lockstep, alone, strict=True):
        assert _same_observations(mine["observations"], theirs["observations"])
        assert torch.equal(mine["actions"], theirs["actions"])
        for key in ("logprobs", "values", "advantages", "returns"):
            assert torch.allclose(mine[key], theirs[key], rtol=1e-4, atol=1e-5), key


def test_worker_by_worker_and_lockstep_dagger_rounds_label_the_same_episodes(suite) -> None:
    imitation = ImitationConfig(dagger_expert=("teacher",), dagger_episodes=3, dagger_envs_per_problem=2, dagger_beta=0.5)
    archives = []
    for together in (True, False):
        trainer = _trainer(suite, imitation, workers=2, together=together)
        try:
            record = trainer.dagger_round()
            observations, labels = trainer.imitation_data.flat()
            archives.append((trainer.imitation_data.archive(), list(observations), list(labels), record))
        finally:
            trainer.close()
    (rows, observations, labels, record), (other_rows, other_observations, other_labels, other_record) = archives
    assert len(rows) == 9 and 0.0 < record["dagger/expert_executed_fraction"] < 1.0
    assert other_rows == rows and other_labels == labels and _same_observations(other_observations, observations)
    assert {key: value for key, value in other_record.items() if "seconds" not in key} == {
        key: value for key, value in record.items() if "seconds" not in key
    }


def test_worker_by_worker_and_lockstep_validation_give_the_same_rows(suite) -> None:
    problems, rows = suite
    policy = GeneralistActorCritic(POLICY)
    outcomes = [
        generalist_ppo.evaluate_policy_suite(
            policy, problems, [SEEDS] * 3, ReferenceTable(rows), SuiteReward(reference="scenario"),
            workers=2, envs_per_problem=2, together=together,
        )
        for together in (True, False)
    ]
    assert [(row["problem"], row["seed"]) for row in outcomes[0]] == [(name, seed) for name in COMPONENTS for seed in SEEDS]
    assert outcomes[1] == outcomes[0]


def test_slots_are_dealt_by_cost_and_the_rollout_stays_the_same(suite) -> None:
    assert generalist_ppo.pack_slots(None, 5, 2) == [[0, 2, 4], [1, 3]]
    assert generalist_ppo.pack_slots([5.0, 1.0, 1.0, 1.0, 1.0, 1.0], 6, 2) == [[0], [1, 2, 3, 4, 5]]
    loads = [10.0, 0.0]
    assert generalist_ppo.pack_slots([3.0, 2.0, 1.0], 3, 2, loads) == [[], [0, 1, 2]] and loads == [10.0, 6.0]
    costs = {"static": {"step": 0.1, "episode": 2.0, "dagger": 3.0}, "same_day": {"step": 0.9, "episode": 30.0, "dagger": 60.0}}
    assert generalist_ppo.slot_weights(costs, ["static", "hos", "same_day"], "step") == [0.1, 0.9, 0.9]  # unknown: the median
    assert generalist_ppo.slot_weights(costs, ["static"], "episode", [4]) == [8.0]
    assert generalist_ppo.slot_weights(None, ["static"], "step") is None

    problems, rows = suite
    imitation = ImitationConfig(dagger_expert=("teacher",), dagger_episodes=1, dagger_envs_per_problem=2, dagger_beta=0.5)
    collected = []
    for slot_costs in (None, {"static": {"step": 0.1, "dagger": 1.0}, "hos": {"step": 0.1, "dagger": 1.0},
                              "same_day": {"step": 5.0, "dagger": 9.0}}):
        config = PPOConfig(total_timesteps=6 * 16 * 2, num_envs=6, rollout_steps=16, minibatch_size=12, epochs=2, seed=3)
        trainer = GeneralistPPO(
            POLICY, config, problems, ReferenceTable(rows), SuiteReward(reference="scenario"), [SEEDS] * 3,
            rollout_workers=2, imitation=imitation, slot_costs=slot_costs,
        )
        try:
            spans, dagger_spans = trainer.envs._spans, trainer.envs._dagger_spans
            trainer._observations = trainer.envs.reset()
            trainer._observe(trainer._observations)
            rollout = trainer._collect_rollout()
            trainer.dagger_round()
            collected.append((spans, dagger_spans, rollout, trainer.imitation_data.archive()))
        finally:
            trainer.close()
    (plain, plain_dagger, first, archive), (packed, packed_dagger, second, other_archive) = collected
    # Slots 2 and 5 run the costly problem: they are dealt first, one to each worker, then the light ones.
    assert plain == [[0, 2, 4], [1, 3, 5]] and packed == [[0, 2, 3], [1, 4, 5]]
    assert sorted(index for span in packed_dagger for index in span) == list(range(6)) and packed_dagger != plain_dagger
    assert _same_observations(first["observations"], second["observations"]) and torch.equal(first["actions"], second["actions"])
    assert torch.allclose(first["returns"], second["returns"], rtol=1e-4, atol=1e-5) and other_archive == archive


def test_sampling_from_given_noise_does_not_depend_on_the_batch(suite) -> None:
    observations, _ = _demonstrations(suite, count=1)
    policy = GeneralistActorCritic(POLICY)
    noise = np.random.default_rng(5).random(len(observations))
    whole, _, _ = generalist_ppo.act_in_chunks(policy, observations, "cpu", uniform=noise)
    single = [
        int(generalist_ppo.act_in_chunks(policy, [observation], "cpu", uniform=noise[index : index + 1])[0])
        for index, observation in enumerate(observations)
    ]
    assert whole.tolist() == single and len(set(single)) > 1
    for observation, position in zip(observations, single, strict=True):
        assert 0 <= position < len(observation.action_index)
    # The draws follow the policy's distribution: both ends of the unit interval stay inside the candidates.
    edges, _, _ = generalist_ppo.act_in_chunks(policy, observations[:2] * 2, "cpu", uniform=[0.0, 0.0, 1.0 - 1e-12, 1.0 - 1e-12])
    assert all(0 <= int(position) < len(observation.action_index) for position, observation in zip(edges, observations[:2] * 2, strict=True))


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


def test_the_replay_store_matches_the_in_memory_replay_and_is_shared(suite, tmp_path, monkeypatch) -> None:
    from algo.generalist_imitation import ImitationData, ObservationStore
    from algo.generalist_observation import ObservationSpec

    problems, rows = suite
    spec = ObservationSpec(charger_budget=10_000)
    episodes = [(index, seed, rows[problem.name][seed]["actions"]) for index, problem in enumerate(problems) for seed in (0, 1)]
    observations, labels, coverage = generalist_ppo.replay_demonstrations(problems, spec, episodes)
    store, stored_labels, stored_coverage = generalist_ppo.replay_demonstrations(
        problems, spec, episodes, workers=2, store=tmp_path / "store",
    )
    assert isinstance(store, ObservationStore) and len(store) == len(observations) == len(stored_labels) > 4
    assert stored_labels == labels and stored_coverage == coverage and len(store.files) > 1
    # The store knows every observation's size without reading it; a store written without the
    # counts computes them once and leaves them beside its index.
    from algo.generalist_observation import node_counts

    assert np.array_equal(store.counts, node_counts(observations)) and np.array_equal(node_counts(store), store.counts)
    (tmp_path / "store" / "counts.npy").unlink()
    older = ObservationStore(tmp_path / "store")
    assert np.array_equal(older.counts, store.counts) and (tmp_path / "store" / "counts.npy").exists()
    picks = [2, 0, len(store) - 1]
    for mine, theirs in zip(store.read_many(picks), [observations[row] for row in picks], strict=True):
        assert np.array_equal(mine.edges, theirs.edges) and np.array_equal(mine.action_index, theirs.action_index)
        assert np.array_equal(mine.tasks, theirs.tasks) and mine.active_truck == theirs.active_truck
    assert np.array_equal(store[1].actions, observations[1].actions) and len(store[1:3]) == 2
    # The imitation set reads the store where it is and samples across it and the DAgger states.
    data = ImitationData(max_dagger_states=100)
    data.set_demonstrations(store, stored_labels)
    data.add_episode({"problem": "x"}, observations[:2], labels[:2])
    flat, flat_labels = data.flat()
    assert data.demonstrations[0] is store and len(flat) == len(store) + 2 and len(data) == len(flat)
    assert list(flat_labels[-2:]) == labels[:2] and flat[len(store)] is observations[0]
    assert np.array_equal(flat.counts, np.concatenate([store.counts, node_counts(observations[:2])]))
    sampled, sampled_labels = data.sample(np.random.default_rng(0), 7)
    assert len(sampled) == 7 == len(sampled_labels)
    # A campaign's runs share one store per replay: a present store is loaded, a different view gets its own.
    calls = []
    original = generalist_ppo.replay_demonstrations
    monkeypatch.setattr(generalist_ppo, "replay_demonstrations", lambda *a, **k: calls.append(1) or original(*a, **k))
    root = tmp_path / "shared"
    first = generalist_ppo.load_or_replay_store(root, problems, spec, episodes)
    again = generalist_ppo.load_or_replay_store(root, problems, spec, episodes)
    assert len(calls) == 1 and again[0].directory == first[0].directory and again[1] == labels
    other = generalist_ppo.load_or_replay_store(root, problems, ObservationSpec(charger_budget=5), episodes)
    assert len(calls) == 2 and other[0].directory != first[0].directory
    assert sorted(path.name for path in root.iterdir()) == sorted([first[0].directory.name, other[0].directory.name])


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
        "--demonstration-store", str(tmp_path / "store"),
        # Small enough that acting, minibatches, the labelled batches and validation all run in several chunks.
        "--pair-budget", "150",
    ]
    cli.main([*common, "--total-timesteps", "24"])
    stores = list((tmp_path / "store").iterdir())
    assert len(stores) == 1 and (stores[0] / "meta.json").exists()
    run = tmp_path / "training" / "cli"
    # The complete state after pretraining is kept: a control forked from it starts at update 0 with the same
    # validation row and trains with its own flags (here the surrogate switched off).
    control = tmp_path / "training" / "control"
    control.mkdir()
    shutil.copy2(run / "training_update0.pt", control / "training.pt")
    forked = [*common[: common.index("--run-name")], "--run-name", "control", *common[common.index("--run-name") + 2:]]
    cli.main([*forked, "--total-timesteps", "24", "--policy-gradient-weight", "0", "--resume", str(control / "training.pt")])
    main_validation = json.loads((run / "validation_history.json").read_text())
    control_validation = json.loads((control / "validation_history.json").read_text())
    assert control_validation[0]["update"] == 0 and control_validation[0]["selection_score"] == main_validation[0]["selection_score"]
    assert [record["update"] for record in json.loads((control / "training_history.json").read_text())] == [1, 2]
    assert json.loads((control / "run_config.json").read_text())["ppo_config"]["policy_gradient_weight"] == 0.0
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

    for name in ("load_or_replay_store", "replay_dagger_archive", "evaluation_envs", "GeneralistPPO"):
        monkeypatch.setattr(cli, name, record(name, getattr(cli, name)))
    monkeypatch.setattr(generalist_ppo, "replay_demonstrations", lambda *a, **k: pytest.fail("the store is loaded, not replayed"))
    original_to = GeneralistActorCritic.to
    monkeypatch.setattr(GeneralistActorCritic, "to", lambda self, *a, **k: events.append("to") or original_to(self, *a, **k))
    cli.main([*common, "--total-timesteps", "36", "--resume", str(run / "training.pt")])
    assert events == ["load_or_replay_store", "replay_dagger_archive", "evaluation_envs", "GeneralistPPO", "to"]
    history = json.loads((run / "training_history.json").read_text())
    assert [record["update"] for record in history] == [1, 2, 3]
    assert history[2]["dagger/round"] == 3 and history[2]["bc_loss"] > 0
