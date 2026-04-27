"""
Quick check that MaskablePPO uses the detour-aware feasible action mask.
It verifies that ActionMasker(mask_fn) matches get_action_mask with detour enabled
and can run multiple full episodes to ensure masks stay consistent.
"""

import argparse
import numpy as np

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import get_action_mask


def mask_fn(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return get_action_mask(base)


def build_env(cfg_path: str, use_detour: bool):
    env = EventDrivenTruckEnv(config=cfg_path, verbose=False, enable_plotting=False)
    env.use_detour_mask = use_detour
    return ActionMasker(Monitor(env), mask_fn)


def run_episode(env, episode_idx: int, max_steps: int) -> dict:
    obs = env.reset()
    steps = 0
    rewards = []
    while True:
        wrapper_mask = env.action_masks()
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        direct_mask = get_action_mask(base_env)

        if not np.array_equal(wrapper_mask, direct_mask):
            diff_idx = np.where(wrapper_mask != direct_mask)[0].tolist()
            return {"ok": False, "episode": episode_idx, "reason": f"mask mismatch at steps={steps}, idx={diff_idx}"}

        feasible_idx = np.flatnonzero(wrapper_mask)
        if feasible_idx.size == 0:
            return {"ok": False, "episode": episode_idx, "reason": "no feasible actions"}

        action = int(feasible_idx[0])
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_result

        rew_val = reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
        rewards.append(float(rew_val))
        steps += 1

        if done or steps >= max_steps:
            return {"ok": True, "episode": episode_idx, "steps": steps, "return": float(sum(rewards))}


def main():
    parser = argparse.ArgumentParser(description="Check detour mask usage for MaskablePPO")
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config.yaml", help="Env config file")
    parser.add_argument("--no-detour", action="store_true", help="Disable detour mask (for comparison)")
    parser.add_argument("--episodes", type=int, default=100, help="How many episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Per-episode step cap")
    args = parser.parse_args()

    use_detour = not args.no_detour
    env = build_env(args.config, use_detour)

    print(f"Detour mask enabled: {use_detour}")

    results = []
    for ep in range(args.episodes):
        res = run_episode(env, ep, args.max_steps)
        results.append(res)
        if res["ok"]:
            print(f"Episode {ep}: ok, steps={res['steps']}, return={res['return']:.3f}")
        else:
            print(f"Episode {ep}: FAIL -> {res['reason']}")
            raise SystemExit(1)

    ok_eps = sum(1 for r in results if r["ok"])
    print(f"\nAll episodes succeeded: {ok_eps}/{len(results)}")


if __name__ == "__main__":
    main()
