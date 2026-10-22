"""The planner experts re-plan from any state and fit DAgger's expert interface."""

import pytest
from generalist_harness import small_env

from algo import generalist_experts as experts
from algo.generalist_imitation import ImitationConfig, check_experts, resolve_expert
from algo.generalist_ppo import SuiteProblem
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility


def _decisions(env):
    return joint_action_feasibility(env)


def test_the_longhaul_expert_replans_after_the_learner_deviates() -> None:
    env = small_env("static", trucks=2, stops=4)
    env.reset(seed=3)
    problem = SuiteProblem("static", "unused.yaml")
    expert = resolve_expert("algo.generalist_experts:longhaul_plan", problem, env)
    assert callable(expert.act) and callable(expert.reset)
    expert.reset(env, 3)
    first = expert.act(env)
    assert _decisions(env)[first].feasible
    env.step(first)  # the first truck follows the plan
    # The next truck claims a customer the plan gave to someone else.
    truck = int(env.active_truck_id)
    own = set(expert.routes.get(truck, ()))
    foreign = [
        index for index, decision in enumerate(_decisions(env))
        if decision.feasible and decision.action_kind is ActionKind.CUSTOMER and int(decision.target_node) not in own
    ]
    if not foreign:
        pytest.skip("the plan gave every open customer to the active truck on this seed")
    env.step(foreign[0])
    again = expert.act(env)
    assert _decisions(env)[again].feasible
    assert expert.stats["deviations"] >= 1 and expert.stats["replans"] >= 1
    with pytest.raises(ValueError, match="static problem"):
        experts.longhaul_plan(problem, small_env("same_day"))


def test_the_rolling_expert_drives_a_same_day_episode_and_is_wrapped_as_a_callable() -> None:
    env = small_env("same_day", trucks=2, stops=4)
    env.reset(seed=1)
    problem = SuiteProblem("same_day", "unused.yaml")
    expert = resolve_expert("algo.generalist_experts:rolling_request", problem, env)
    assert callable(expert.act)
    steps = 0
    for _ in range(40):
        action = expert.act(env)
        assert _decisions(env)[action].feasible
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert steps > 1
    assert experts.is_dynamic(env) and not experts.is_dynamic(small_env("static"))
    with pytest.raises(ValueError, match="dynamic request mode"):
        experts.rolling_request(problem, small_env("static"))
    bound = experts.rolling_request(problem, env, clairvoyant=True)
    assert bound.parameters.clairvoyant and not experts.rolling_request(problem, env).parameters.clairvoyant
    assert type(experts.for_problem(problem, env)) is type(bound)
    assert type(experts.for_problem(problem, small_env("static"))).__name__ == "LonghaulPlanPolicy"


def test_the_campaign_experts_resolve_for_every_suite_problem() -> None:
    from scripts.campaigns.generalist_graphppo import DYNAMIC, EXPERTS, PROBLEMS

    config = ImitationConfig(dagger_expert=tuple(EXPERTS))
    problems = [SuiteProblem(name, "unused.yaml") for name in PROBLEMS]
    table = check_experts(config, problems)
    for name in PROBLEMS:
        factory = "rolling_request" if name in DYNAMIC else "longhaul_plan"
        assert table[name] == f"algo.generalist_experts:{factory}"
