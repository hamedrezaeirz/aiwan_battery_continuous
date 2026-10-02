"""
sanity_check_battery.py
--------------------------
Run this BEFORE committing to any multi-day VPS sweep for the battery-
on-continuous-environment experiment. Mirrors sanity_check_oversight.py
closely, with one CRITICAL ADDITIONAL check that oversight's version
didn't need: verifying the assumption (stated but NOT yet verified in
train_battery.py's train_rho docstring) that rho does NOT need a
training-density boost the way rho_oversight did, since battery drains
continuously every step rather than activating rarely and stochastically.

What this checks, in order:

  1. Reference baselines: a random policy, and a hand-coded "greedy
     toward goal, ignore battery entirely" heuristic -- same role as
     sanity_check_oversight.py's baselines.
  2. Learning curves for Architecture A, g, and rho -- using this
     experiment's own rollout(), at the SAME budget for all three
     (unlike rho_oversight, which needed a special larger budget --
     this check is specifically what determines whether rho here
     also needs special treatment or not).
  3. Gate learning curve + routing diagnostic for Architecture C,
     bucketed by BATTERY LEVEL (not a binary flag, since battery is
     graded) -- mirrors the original project's sanity_check_gate.py
     battery-bucket diagnostic, not env_oversight's binary-bucket one.
  4. Architecture B (sigmoid DualLoopPolicy) evaluated on the same
     g/rho as C, for direct comparison.

Usage:
    python sanity_check_battery.py

NOTE: needs stable-baselines3 + torch installed. Only syntax-checked
before being handed off (this container has no SB3/torch installed) --
run this on the machine where train_battery.py's own smoke test was
already confirmed to run (either the local machine or the VPS).
"""

import numpy as np
import matplotlib.pyplot as plt

from env_battery_continuous import (
    ContinuousNavBatteryEnv,
    RewardWrapperA_Battery,
    RewardWrapperGoalOnly_Battery,
    RewardWrapperBatteryOnly,
    LearnedGateEnv_Battery,
)
from train_battery import _make_ppo, train_g, train_rho
from arbitration_battery import DualLoopPolicy, LearnedGatePolicy_Battery

CHECK_TIMESTEPS = 50_000     # SAME budget for A/g/rho -- unlike the oversight
                              # experiment, rho is NOT expected to need a
                              # special larger budget (see module docstring);
                              # this check is what actually verifies that,
                              # rather than assuming it.
CHECK_INTERVAL = 5_000
CHECK_NET_SIZE = [64]
CHECK_SEED = 0
N_EVAL_EPISODES = 30
BATTERY_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
N_ROUTING_EPISODES = 50


def rollout(policy_fn, n_episodes, seed=None, adversarial_weight=0.0, needs_battery_level=False):
    """
    Mirrors env_oversight's evaluate_oversight.rollout(), adapted for
    battery/crash instead of oversight/violation terminology.

    needs_battery_level=True switches policy_fn's signature from
    (obs) -> action to (obs, battery_level) -> action, for
    DualLoopPolicy (Architecture B) specifically, mirroring the
    original project's exact same distinction.
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


def as_policy_fn(sb3_model):
    def fn(obs):
        action, _ = sb3_model.predict(obs, deterministic=True)
        return action
    return fn


def as_dual_loop_policy_fn(dual_loop_policy):
    def fn(obs, battery_level):
        return dual_loop_policy.predict(obs, battery_level, deterministic=True)
    return fn


def as_learned_gate_policy_fn(learned_gate_policy):
    def fn(obs):
        return learned_gate_policy.predict(obs, deterministic=True)
    return fn


def random_baseline(n_episodes=100):
    env = ContinuousNavBatteryEnv()
    return rollout(lambda obs: env.action_space.sample(), n_episodes, seed=2000)


def greedy_ignore_battery_baseline(n_episodes=100):
    """Hand-coded, ignoring battery ENTIRELY on purpose -- always move
    directly toward the goal at full speed. High crash_rate here is
    expected and fine; this is a ceiling for success_rate."""
    def policy_fn(obs):
        agent_x, agent_y, goal_x, goal_y = obs[0], obs[1], obs[2], obs[3]
        direction = np.array([goal_x - agent_x, goal_y - agent_y])
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / norm).astype(np.float32)
    return rollout(policy_fn, n_episodes, seed=3000)


def learning_curve(build_env_fn, hp, label, total_timesteps=CHECK_TIMESTEPS):
    env = build_env_fn()
    model = _make_ppo(env, CHECK_NET_SIZE, CHECK_SEED, hp)

    steps_done, success_rates, crash_rates = [], [], []
    n_chunks = total_timesteps // CHECK_INTERVAL
    for i in range(n_chunks):
        model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
        metrics = rollout(as_policy_fn(model), n_episodes=N_EVAL_EPISODES, seed=1000 + i)
        steps_done.append((i + 1) * CHECK_INTERVAL)
        success_rates.append(metrics["success_rate"])
        crash_rates.append(metrics["crash_rate"])
        print(f"  [{label}] step={steps_done[-1]:>7} "
              f"success_rate={metrics['success_rate']:.2f} "
              f"crash_rate={metrics['crash_rate']:.2f}")

    return steps_done, success_rates, crash_rates


def gate_learning_curve(g_model, rho_model, hp):
    env = LearnedGateEnv_Battery(ContinuousNavBatteryEnv(), g_model, rho_model)
    gate_model = _make_ppo(env, CHECK_NET_SIZE, CHECK_SEED, hp)

    steps_done, success_rates, crash_rates = [], [], []
    n_chunks = CHECK_TIMESTEPS // CHECK_INTERVAL
    for i in range(n_chunks):
        gate_model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
        composed = LearnedGatePolicy_Battery(g_model, rho_model, gate_model)
        metrics = rollout(as_learned_gate_policy_fn(composed), n_episodes=N_EVAL_EPISODES, seed=1000 + i)
        steps_done.append((i + 1) * CHECK_INTERVAL)
        success_rates.append(metrics["success_rate"])
        crash_rates.append(metrics["crash_rate"])
        print(f"  [gate] step={steps_done[-1]:>7} "
              f"success_rate={metrics['success_rate']:.2f} "
              f"crash_rate={metrics['crash_rate']:.2f}")

    return steps_done, success_rates, crash_rates, gate_model


def routing_diagnostic(g_model, rho_model, gate_model, n_episodes=N_ROUTING_EPISODES):
    """
    Mirrors the ORIGINAL project's sanity_check_gate.py routing
    diagnostic exactly -- bucketed by battery level (graded), not a
    binary flag, since battery is graded here (unlike oversight).
    """
    env = ContinuousNavBatteryEnv()
    bucket_counts = {b: [0, 0] for b in BATTERY_BUCKETS}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=4000 + ep)
        for _ in range(env.max_episode_steps):
            gate_action, _ = gate_model.predict(obs, deterministic=True)
            gate_action = int(gate_action)
            battery_pct = 100.0 * env.battery / 100.0  # BATTERY_MAX = 100.0
            for lo, hi in BATTERY_BUCKETS:
                if lo <= battery_pct < hi or (hi == 100 and battery_pct == 100):
                    bucket_counts[(lo, hi)][1] += 1
                    bucket_counts[(lo, hi)][0] += gate_action
                    break
            sub_model = rho_model if gate_action == 1 else g_model
            real_action, _ = sub_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(real_action)
            if terminated or truncated:
                break

    print("\nRouting diagnostic -- fraction of steps the gate deferred to rho, by battery level:")
    fractions = []
    for (lo, hi) in BATTERY_BUCKETS:
        n_rho, n_total = bucket_counts[(lo, hi)]
        frac = n_rho / n_total if n_total > 0 else float("nan")
        fractions.append(frac)
        print(f"  battery in [{lo:>3}, {hi:>3}): n={n_total:>5}  fraction routed to rho = {frac:.2f}")

    spread = np.nanmax(fractions) - np.nanmin(fractions)
    if spread < 0.05:
        print(f"  WARNING: spread across buckets is only {spread:.2f} -- the gate may have "
              f"collapsed onto a state-independent rule. Worth a closer look before the real sweep.")
    else:
        print(f"  Spread across buckets: {spread:.2f} -- routing does vary with battery level.")
    return fractions


if __name__ == "__main__":
    import json
    import os as _os

    HP_PATH = "results/hyperparams_battery.json"
    if _os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            hp = json.load(f)
        print(f"NOTE: loaded TUNED hyperparameters from {HP_PATH}: {hp}\n")
    else:
        hp = {"learning_rate": 3e-4, "gamma": 0.99, "log_std_init": -0.5, "lambda_battery_a": 1.0}
        print(f"NOTE: {HP_PATH} not found -- using placeholder hyperparameters {hp}. "
              f"Run tune_hyperparams_battery.py first for tuned values.\n")

    print("=" * 70)
    print("STEP 1: Reference baselines (no learning involved)")
    print("=" * 70)
    rand = random_baseline()
    print(f"  random policy:          success_rate={rand['success_rate']:.2f} "
          f"crash_rate={rand['crash_rate']:.2f}")
    greedy = greedy_ignore_battery_baseline()
    print(f"  greedy-ignore-battery:   success_rate={greedy['success_rate']:.2f} "
          f"crash_rate={greedy['crash_rate']:.2f} (ignores battery on purpose -- "
          f"a HIGH crash_rate here is expected and fine)")

    print("\n" + "=" * 70)
    print("STEP 2: Learning curves, Architecture A / g / rho -- SAME budget for all three")
    print("(this is what verifies whether rho needs special treatment, unlike assuming it doesn't)")
    print("=" * 70)
    print("\nArchitecture A:")
    steps_a, succ_a, crash_a = learning_curve(
        lambda: RewardWrapperA_Battery(ContinuousNavBatteryEnv(), lambda_battery=hp["lambda_battery_a"]),
        hp, "A")

    print("\ng (goal-only):")
    steps_g, succ_g, _ = learning_curve(
        lambda: RewardWrapperGoalOnly_Battery(ContinuousNavBatteryEnv()), hp, "g")

    print("\nrho (battery-only):")
    steps_r, _, crash_r = learning_curve(
        lambda: RewardWrapperBatteryOnly(ContinuousNavBatteryEnv()), hp, "rho")

    if crash_r[-1] > 0.3:
        print(f"\n  WARNING: rho's final crash_rate ({crash_r[-1]:.2f}) is still high after "
              f"{CHECK_TIMESTEPS} steps at the SAME budget as A/g. This suggests the "
              f"assumption in train_battery.py's train_rho docstring (that battery's "
              f"continuous nature means no training-density boost is needed) may be "
              f"WRONG -- consider a diagnose_rho_battery_instability.py analogous to "
              f"the oversight experiment's diagnose_rho_instability.py before trusting "
              f"this architecture's results.")
    else:
        print(f"\n  rho's crash_rate ({crash_r[-1]:.2f}) looks reasonable at the standard "
              f"budget -- the assumption that battery's continuous nature avoids the "
              f"sparse-gradient problem rho_oversight had appears to hold, but keep this "
              f"in mind if later results look inconsistent.")

    print("\n" + "=" * 70)
    print("STEP 3 & 4: Gate learning curve + routing diagnostic (Architecture C), "
          "Architecture B comparison")
    print("=" * 70)
    print(f"\nTraining g and rho fully ({CHECK_TIMESTEPS} steps each, held fixed for the gate)...")
    g_model = train_g(CHECK_NET_SIZE, CHECK_TIMESTEPS, CHECK_SEED, hp)
    rho_model = train_rho(CHECK_SEED, hp, net_size=CHECK_NET_SIZE, timesteps=CHECK_TIMESTEPS)

    print("\nEvaluating Architecture B (sigmoid gate, DualLoopPolicy) on this SAME g/rho:")
    # k/b_mid now come from the TUNED hp (gate_k, gate_b_mid), not hardcoded
    # placeholders -- see tune_hyperparams_battery.py.
    gate_k = hp.get("gate_k", 0.2)
    gate_b_mid = hp.get("gate_b_mid", 30.0)
    dual = DualLoopPolicy(g_model, rho_model, k=gate_k, b_mid=gate_b_mid)
    b_metrics = rollout(as_dual_loop_policy_fn(dual), n_episodes=100, seed=5000,
                         needs_battery_level=True)
    print(f"  B: success_rate={b_metrics['success_rate']:.2f} crash_rate={b_metrics['crash_rate']:.2f}")

    print("\nLearning curve, gate (Architecture C):")
    steps_c, succ_c, crash_c, gate_model = gate_learning_curve(g_model, rho_model, hp)

    routing_diagnostic(g_model, rho_model, gate_model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps_a, succ_a, marker="o", label="A")
    axes[0].plot(steps_g, succ_g, marker="o", label="g")
    axes[0].plot(steps_c, succ_c, marker="o", label="C (learned gate)")
    axes[0].axhline(b_metrics["success_rate"], color="tab:orange", linestyle="-.", label="B (sigmoid gate)")
    axes[0].axhline(rand["success_rate"], color="gray", linestyle="--", label="random baseline")
    axes[0].axhline(greedy["success_rate"], color="black", linestyle=":", label="greedy-ignore-battery")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("success_rate")
    axes[0].set_title("Does success_rate actually climb?")
    axes[0].legend(fontsize="small")

    axes[1].plot(steps_a, crash_a, marker="o", label="A")
    axes[1].plot(steps_r, crash_r, marker="o", label="rho")
    axes[1].plot(steps_c, crash_c, marker="o", label="C (learned gate)")
    axes[1].axhline(b_metrics["crash_rate"], color="tab:orange", linestyle="-.", label="B (sigmoid gate)")
    axes[1].axhline(rand["crash_rate"], color="gray", linestyle="--", label="random baseline")
    axes[1].set_xlabel("training steps")
    axes[1].set_ylabel("crash_rate")
    axes[1].set_title("Does crash_rate actually drop?")
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    import os
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sanity_check_battery_learning_curves.png", dpi=150)
    print("\nSaved results/sanity_check_battery_learning_curves.png")
