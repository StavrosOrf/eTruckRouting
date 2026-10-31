"""Track A decision-time additions: the feasibility mask and the campaign's decision settings."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from generalist_harness import small_env

from algo.generalist_search import (
    _KINDS,
    retire_guard_enabled,
    retire_guarded,
    safe_actions,
    safety_margin,
    safety_mask_enabled,
)
from EVRoutingEnv.baselines.heuristics.canonical_baselines import RandomFeasiblePolicy
from EVRoutingEnv.state.feasibility import ActionKind


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in ("EVRP_SAFETY_MASK", "EVRP_SAFETY_MARGIN", "EVRP_RETIRE_GUARD", "EVRP_PLAN_MODE", "EVRP_PLAN_CACHE", "EVRP_PLAN_DROPOUT"):
        monkeypatch.delenv(name, raising=False)


# Nodes: 0 depot, 1 charger, 2 customer near the charger, 3 customer far from everything.
ENERGY = np.array([
    [0.0, 10.0, 20.0, 50.0],
    [10.0, 0.0, 10.0, 60.0],
    [20.0, 10.0, 0.0, 60.0],
    [50.0, 60.0, 60.0, 0.0],
])
HOURS = ENERGY / 10.0
INDEX = {node: node for node in range(4)}


def _env(*, battery: float, clock: float = 0.0, horizon: float = 30.0, node: int = 0, charge_hours: float = 1.0):
    truck = SimpleNamespace(truck_id=0, current_node=node, current_battery=battery, battery_capacity=100.0)
    return SimpleNamespace(
        active_truck_id=0, trucks=[truck], global_clock=clock, max_time=horizon,
        joint_instance=SimpleNamespace(depot_node=0), charging_nodes=[1],
        charging_station=SimpleNamespace(charger_power_kw={1: 50.0}),
        traffic_config={"enable_traffic": True, "enable_energy_uncertainty": True, "max_energy_multiplier": 1.2},
        charging_action_mode="target_soc", dynamic_requests=False, request_engine=None,
        _certificate_charge_transition=lambda here, battery, value: (value * 100.0, charge_hours, value),
    )


def _rows(*actions):
    kinds = np.array([_KINDS.index(kind) for kind, _, _, _ in actions])
    targets = np.array([target for _, target, _, _ in actions], dtype=float)
    values = np.array([value for _, _, value, _ in actions], dtype=float)
    energies = np.array([energy for _, _, _, energy in actions], dtype=float)
    return kinds, targets, values, energies


def test_a_customer_move_needs_the_energy_to_reach_a_refuge_afterwards() -> None:
    # 70 kWh: customer 2 (20 kWh) leaves 50 >= 1.2 * 10 to the charger; customer 3 (50 kWh) leaves 20 < 1.2 * 50.
    rows = _rows((ActionKind.CUSTOMER, 2, 0, 20.0), (ActionKind.CUSTOMER, 3, 0, 50.0), (ActionKind.CHARGER, 1, 0, 10.0))
    keep = safe_actions(_env(battery=70.0), *rows, np.ones(3, bool), INDEX, ENERGY, HOURS)
    assert keep.tolist() == [True, False, True]


def test_a_move_must_leave_time_to_get_home_before_the_horizon(monkeypatch) -> None:
    # At t=20 with 100 kWh: customer 3 is 5 h out and 5 h back (30 > 29.5 with the margin); customer 2 is 2 + 2 h.
    rows = _rows((ActionKind.CUSTOMER, 2, 0, 20.0), (ActionKind.CUSTOMER, 3, 0, 30.0))
    keep = safe_actions(_env(battery=100.0, clock=20.0), *rows, np.ones(2, bool), INDEX, ENERGY, HOURS)
    assert keep.tolist() == [True, True]
    monkeypatch.setenv("EVRP_SAFETY_MARGIN", "0.5")
    keep = safe_actions(_env(battery=100.0, clock=20.0), *rows, np.ones(2, bool), INDEX, ENERGY, HOURS)
    assert keep.tolist() == [True, False]


def test_the_way_home_may_need_a_charging_stop_and_a_long_charge_can_be_too_late() -> None:
    # From customer 2 with 25 kWh left the depot (1.2 * 20 = 24) is in reach: home is 2 h direct.
    rows = _rows((ActionKind.CUSTOMER, 2, 0, 0.0))
    assert safe_actions(_env(battery=25.0, clock=26.0, node=2), *rows, np.ones(1, bool), INDEX, ENERGY, HOURS).all()
    # With 15 kWh it is not: 1 h to the charger, (12 - 3) / 50 h of charge, 1 h home -> 28.18 > 28 at t=26.
    keep = safe_actions(_env(battery=15.0, clock=26.0, horizon=28.0, node=2), *_rows(
        (ActionKind.CUSTOMER, 2, 0, 0.0), (ActionKind.DEPOT, 0, 0, 12.0)), np.ones(2, bool), INDEX, ENERGY, HOURS)
    assert keep.tolist() == [False, True]
    # At the charger: a 3 h charge cannot be followed by the 1 h drive home before t=30.
    env = _env(battery=50.0, clock=26.5, node=1, charge_hours=3.0)
    rows = _rows((ActionKind.CHARGE, 1, 1.0, 0.0), (ActionKind.DEPOT, 0, 0, 10.0))
    assert safe_actions(env, *rows, np.ones(2, bool), INDEX, ENERGY, HOURS).tolist() == [False, True]


def test_the_hard_mask_stands_when_every_action_would_be_removed() -> None:
    rows = _rows((ActionKind.CUSTOMER, 3, 0, 50.0))
    feasible = np.ones(1, bool)
    assert safe_actions(_env(battery=60.0), *rows, feasible, INDEX, ENERGY, HOURS).tolist() == [True]


def test_the_energy_rule_stands_when_the_time_rule_would_remove_everything_it_leaves() -> None:
    rows = _rows((ActionKind.CUSTOMER, 3, 0, 50.0), (ActionKind.CHARGER, 1, 0, 10.0))
    feasible = np.ones(2, bool)
    env = _env(battery=60.0, clock=29.9)  # no time for either; the customer also strands the truck
    assert safe_actions(env, *rows, feasible, INDEX, ENERGY, HOURS).tolist() == [False, True]


def test_the_retirement_guard_keeps_a_planned_truck_out_while_it_can_do_anything_else(monkeypatch) -> None:
    kinds = np.array([_KINDS.index(ActionKind.CUSTOMER), _KINDS.index(ActionKind.DEPOT)])
    both, only_depot = np.array([True, True]), np.array([False, True])
    assert retire_guarded(kinds, both, [3]).tolist() == [True, False]
    assert retire_guarded(kinds, both, []).tolist() == [True, True]
    assert retire_guarded(kinds, both, None).tolist() == [True, True]
    assert retire_guarded(kinds, only_depot, [3]).tolist() == [False, True]
    assert retire_guard_enabled() is False
    monkeypatch.setenv("EVRP_RETIRE_GUARD", "on")
    assert retire_guard_enabled() is True
    monkeypatch.setenv("EVRP_RETIRE_GUARD", "maybe")
    with pytest.raises(ValueError, match="EVRP_RETIRE_GUARD"):
        retire_guard_enabled()


def test_settings_are_read_from_the_environment(monkeypatch) -> None:
    assert not safety_mask_enabled() and safety_margin() == 0.0
    monkeypatch.setenv("EVRP_SAFETY_MASK", "on")
    monkeypatch.setenv("EVRP_SAFETY_MARGIN", "1.5")
    assert safety_mask_enabled() and safety_margin() == 1.5
    monkeypatch.setenv("EVRP_SAFETY_MASK", "yes")
    with pytest.raises(ValueError):
        safety_mask_enabled()


@pytest.mark.parametrize("component", ["static", "public", "hos", "same_day"])
def test_the_masked_view_offers_a_subset_of_the_plain_one(component, monkeypatch) -> None:
    env = small_env(component, trucks=2, stops=4)
    observation, _ = env.reset(seed=11)
    policy = RandomFeasiblePolicy(11)
    for _ in range(40):
        monkeypatch.setenv("EVRP_SAFETY_MASK", "on")
        monkeypatch.setenv("EVRP_SAFETY_MARGIN", "1.0")
        masked = env._get_observation()
        monkeypatch.setenv("EVRP_SAFETY_MASK", "off")
        plain = env._get_observation()
        assert masked.num_actions >= 1 and set(masked.action_index.tolist()) <= set(plain.action_index.tolist())
        observation, _, terminated, truncated, _ = env.step(policy(env))
        if terminated or truncated:
            break


def test_the_campaign_freezes_decision_settings_per_tag(tmp_path, monkeypatch, capsys) -> None:
    from scripts.campaigns import generalist_graphppo as campaign

    with pytest.raises(SystemExit, match="needs --safety-mask"):
        campaign.decision_from(False, 1.0)
    assert campaign.decision_from(False, None) is None
    settings = campaign.decision_from(True, 1.0)
    assert settings == {"safety_mask": True, "safety_margin": 1.0}
    folder = tmp_path / "evaluation" / "validation" / "8M_mask1"
    campaign.write_decision(folder, settings)
    campaign.write_decision(folder, settings)  # idempotent
    with pytest.raises(SystemExit, match="other settings"):
        campaign.write_decision(folder, campaign.decision_from(True, 0.0))
    monkeypatch.setenv("EVRP_SAFETY_MASK", "off")
    assert campaign.apply_decision(folder) == settings
    import os

    assert os.environ["EVRP_SAFETY_MASK"] == "on" and os.environ["EVRP_SAFETY_MARGIN"] == "1.0"
    assert campaign.apply_decision(tmp_path) == {} and os.environ["EVRP_SAFETY_MASK"] == "off"
    assert json.loads((folder / campaign.DECISION).read_text()) == settings
    with pytest.raises(SystemExit, match="needs --safety-mask"):
        campaign.decision_from(False, None, retire_guard=True)
    guarded = campaign.decision_from(True, 0.0, retire_guard=True)
    assert guarded == {"safety_mask": True, "safety_margin": 0.0, "retire_guard": True}
    campaign.write_decision(tmp_path / "guarded", guarded)
    campaign.apply_decision(tmp_path / "guarded")
    assert os.environ["EVRP_RETIRE_GUARD"] == "on"
    campaign.apply_decision(folder)
    assert os.environ["EVRP_RETIRE_GUARD"] == "off"


def test_the_evaluation_wrapper_runs_the_code_frozen_with_a_tag(tmp_path) -> None:
    import os
    import subprocess

    from scripts.campaigns import generalist_graphppo as campaign

    record = tmp_path / "ran.txt"
    fake = tmp_path / "python"
    fake.write_text(f"#!/bin/bash\necho \"$PWD $@\" > {record}\n")
    fake.chmod(0o755)
    directory, training_source, tag_source = tmp_path / "campaign", tmp_path / "source", tmp_path / "tag_source"
    training_source.mkdir()
    tag_source.mkdir()
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"python": str(fake), "source": str(training_source)}))
    environment = {**os.environ, "SLURM_CPUS_PER_TASK": "18", "PATH": "/usr/bin:/bin"}
    wrapper = str(campaign.ROOT / "scripts/campaigns/generalist_graphppo_eval.sbatch")
    subprocess.run(["bash", wrapper, str(directory), "validation", "8M", "a_s0"], env=environment, check=True)
    assert record.read_text().split()[0] == str(training_source)
    folder = directory / "evaluation" / "validation" / "8M_mask"
    folder.mkdir(parents=True)
    (folder / campaign.DECISION).write_text(json.dumps({"safety_mask": True, "source": str(tag_source)}))
    subprocess.run(["bash", wrapper, str(directory), "validation", "8M_mask", "a_s0"], env=environment, check=True)
    ran = record.read_text().split()
    assert ran[0] == str(tag_source) and ran[ran.index("--tag") + 1] == "8M_mask"


def _policy():
    from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig

    config = GeneralistPolicyConfig(hidden_dim=16, encoder_layers=1, action_head_layers=1, charger_budget=4)
    return GeneralistActorCritic(config).eval()


def _finish(env, seed: int, limit: int | None = None) -> float:
    policy = RandomFeasiblePolicy(seed)
    steps = 0
    while env.active_truck_id is not None and not env._episode_ended and (limit is None or steps < limit):
        steps += 1
        _, _, terminated, truncated, _ = env.step(policy(env))
        if terminated or truncated:
            break
    return float(env.global_clock)


def test_a_fresh_copy_keeps_the_past_and_redraws_the_future() -> None:
    from algo.generalist_search import fresh_clone

    env = small_env("static", trucks=2, stops=4)
    env.reset(seed=5)
    policy = RandomFeasiblePolicy(5)
    for _ in range(3):
        env.step(policy(env))
    copies = [fresh_clone(env, seed) for seed in (101, 101, 202)]
    for copy in copies:
        assert copy.global_clock == env.global_clock and copy.active_truck_id == env.active_truck_id
        assert [t.current_node for t in copy.trucks] == [t.current_node for t in env.trucks]
    ends = [_finish(copy, 9) for copy in copies]
    assert ends[0] == ends[1]  # the same seed draws the same future
    assert ends[2] != ends[0] and _finish(env, 9) != ends[0]  # a new seed, and the episode's own draws, differ


def test_dynamic_problems_are_never_searched() -> None:
    from algo.generalist_search import fresh_clone

    env = small_env("same_day", trucks=2, stops=4)
    env.reset(seed=5)
    with pytest.raises(ValueError, match="static problems"):
        fresh_clone(env, 1)


@pytest.mark.parametrize("search", ["lookahead", "best_of_n"])
def test_both_searches_play_a_whole_episode(search) -> None:
    from algo.generalist_search import search_episode

    env = small_env("static", trucks=2, stops=4)
    summary = search_episode(env, _policy(), 7, search, k=2, n=2)
    assert summary["steps"] > 0 and summary["termination_reason"] is not None
    with pytest.raises(ValueError, match="unknown search"):
        search_episode(env, _policy(), 7, "beam")


def test_search_settings_are_part_of_the_decision() -> None:
    from scripts.campaigns import generalist_graphppo as campaign

    assert campaign.decision_from(True, 1.0, "lookahead", 4, 16) == {
        "safety_mask": True, "safety_margin": 1.0, "search": "lookahead", "k": 4, "n": None,
    }
    assert campaign.decision_from(False, None, "best_of_n", 4, 8) == {"search": "best_of_n", "k": None, "n": 8}


def _charging_seeker(env, rng):
    """A feasible action, preferring to queue, charge or drive to a charger (to load the public stations)."""
    decisions = env.last_action_feasibility
    feasible = [index for index, decision in enumerate(decisions) if decision.feasible]
    for kind in (ActionKind.QUEUE, ActionKind.CHARGE, ActionKind.CHARGER):
        options = [index for index in feasible if decisions[index].action_kind is kind]
        if options and rng.random() < 0.8:
            return int(rng.choice(options))
    return int(rng.choice(feasible))


@pytest.mark.parametrize("seed", [3, 11, 19])
def test_copies_at_busy_public_stations_keep_every_promised_reservation(seed) -> None:
    # Copies taken while trucks queue or hold predicted reservations play on
    # without breaking a reservation. (The crash this guards against, a promised
    # reservation moved into the past when a lagging station was redrawn from
    # its own clock, came from a 5 h validation search on congested Texas and
    # is too rare to reproduce here; the fix advances each station to now first.)
    from generalist_harness import small_config

    from algo.generalist_observation import GeneralistEnv, ObservationSpec
    from algo.generalist_search import fresh_clone

    config = small_config("public", trucks=3, stops=6)
    config["public_charging"]["exogenous_traffic"]["utilization"] = 0.97
    # One charger per view: most stations are not observed for a while, so their clocks lag.
    env = GeneralistEnv(config, ObservationSpec(charger_budget=1))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    copies = 0
    for step in range(600):
        stations = env.public_stations
        if any(stations.queue_position(truck.truck_id) is not None or truck.truck_id in stations.scheduled
               for truck in env.trucks):
            copy = fresh_clone(env, 1000 + step)
            copy_rng = np.random.default_rng(step)
            for _ in range(200):
                if copy.active_truck_id is None or copy._episode_ended:
                    break
                _, _, terminated, truncated, _ = copy.step(_charging_seeker(copy, copy_rng))
                if terminated or truncated:
                    break
            copies += 1
        _, _, terminated, truncated, _ = env.step(_charging_seeker(env, rng))
        if terminated or truncated or env.active_truck_id is None or copies >= 8:
            break
    assert copies > 0  # a truck was queued or held a predicted reservation when copied


def test_a_broken_copy_falls_back_to_the_policy_and_is_counted(monkeypatch) -> None:
    import algo.generalist_search as search

    env = small_env("static", trucks=2, stops=4)
    observation, _ = env.reset(seed=7)
    policy = _policy()
    greedy = int(observation.action_index[int(np.argmax(search._logits(policy, observation)))])

    def broken(*args, **kwargs):
        raise RuntimeError("reservation for truck 0 at 1.0 was missed")

    monkeypatch.setattr(search, "fresh_clone", broken)
    action, fell_back = search.lookahead_action(env, observation, policy, 3, 1)
    if observation.num_actions > 1:
        assert fell_back and action == greedy
    summary = search.search_episode(env, policy, 7, "lookahead", k=3)
    assert summary["search_fallbacks"] >= 1 and summary["termination_reason"] is not None
    key, routes = search.best_of_n_routes(env, policy, 2, 1)
    assert key is None and routes is None
