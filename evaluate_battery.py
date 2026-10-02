"""
evaluate_battery.py
----------------------
Rollout + metrics for the battery-on-continuous-environment experiment.
Extracted from sanity_check_battery.py's inline rollout()/as_*_policy_fn()
helpers so run_experiment_battery.py can share the exact same evaluation
code path. Mirrors evaluate_oversight.py's structure and the original
project's evaluate.py.
"""

import numpy as np

from env_battery_continuous import ContinuousNavBatteryEnv


def as_policy_fn(sb3_model):
    """Adapts a plain SB3 model (Architecture A, or a lone untrained g
    for the goal-nulling scenario) to the (obs) -> action interface."""
    def fn(obs):
        action, _ = sb3_model.predict(obs, deterministic=True)
        return action
    return fn


def as_dual_loop_policy_fn(dual_loop_policy):
    """Adapts a DualLoopPolicy (Architecture B) to the
    (obs, battery_level) -> action interface -- DualLoopPolicy needs the
    raw battery level, mirroring the original project's exact same
    distinction."""
    def fn(obs, battery_level):
        return dual_loop_policy.predict(obs, battery_level, deterministic=True)
    return fn


def as_learned_gate_policy_fn(learned_gate_policy):
    """Adapts a LearnedGatePolicy_Battery (Architecture C) to the
    (obs) -> action interface."""
    def fn(obs):
        return learned_gate_policy.predict(obs, deterministic=True)
    return fn


def rollout(policy_fn, n_episodes, seed=None, adversarial_weight=0.0, needs_battery_level=False):
    """
    Runs n_episodes on a fresh ContinuousNavBatteryEnv and aggregates
    metrics. Mirrors evaluate_oversight.py's rollout() closely, with
    crash/battery terminology instead of violation/oversight.

    needs_battery_level=True switches policy_fn's expected signature
    from (obs) -> action to (obs, battery_level) -> action, for
    DualLoopPolicy (Architecture B) specifically.
    """
    env = ContinuousNavBatteryEnv(adversarial_weight=adversarial_weight)
    successes = crashes = 0
    battery_levels, lengths = [], []

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        steps = 0
        info = {}
        for _ in range(env.max_episode_steps):
            if needs_battery_level:
                action = policy_fn(obs, env.battery)
            else:
                action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        successes += int(info.get("reached_goal", False))
        crashes += int(info.get("crashed", False))
        battery_levels.append(info.get("battery", 0.0))
        lengths.append(steps)

    return {
        "n_eval_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "crash_rate": crashes / n_episodes,
        "avg_battery_level": float(np.mean(battery_levels)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def evaluate_cell(architecture, network_size, timesteps, seed, condition,
                   policy_fn, n_episodes, adversarial_weight=0.0, needs_battery_level=False):
    """
    Runs rollout() and packages the result into one row matching this
    experiment's results CSV schema. Mirrors evaluate_oversight.py's
    evaluate_cell() exactly.
    """
    metrics = rollout(policy_fn, n_episodes, seed=seed, adversarial_weight=adversarial_weight,
                       needs_battery_level=needs_battery_level)
    return {
        "architecture": architecture,
        "network_size": str(network_size),
        "training_timesteps": timesteps,
        "seed": seed,
        "condition": condition,
        **metrics,
    }
