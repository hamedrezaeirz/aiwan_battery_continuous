"""
train_battery.py
-------------------
Training functions for the battery-on-continuous-environment
experiment (Experiment 2 of the 5-month plan: replicating the original
§7.3 result on the same continuous dynamics built for the oversight
experiment). Mirrors train_oversight.py's structure and naming exactly
(train_architecture_a, train_g, train_rho, train_gate, get_untrained_*)
so run_experiment_battery.py's logic can reuse the same orchestration
pattern -- only the env module, wrapper classes, and B's arbitration
class (sigmoid DualLoopPolicy here, vs. HardGatePolicy for oversight)
differ.

Algorithm: PPO, same reasoning as train_oversight.py (continuous action
space, DQN doesn't apply). Uses the SAME log_std_init-as-a-tunable-
hyperparameter pattern established there, since the same continuous-
action sampling-noise issue (bug #2/#5 in the oversight experiment's
history) applies here too and there's no reason to assume it won't
recur -- a dedicated tune_hyperparams_battery.py pass (not yet built)
should search this properly rather than assuming the oversight
experiment's tuned values transfer directly (different reward
structure, different signal dynamics).
"""

from stable_baselines3 import PPO
from env_battery_continuous import (
    ContinuousNavBatteryEnv,
    RewardWrapperA_Battery,
    RewardWrapperGoalOnly_Battery,
    RewardWrapperBatteryOnly,
    LearnedGateEnv_Battery,
)


def _make_ppo(env, net_arch, seed, hp):
    """
    Shared PPO settings -- mirrors train_oversight.py's _make_ppo
    exactly, including reading log_std_init from hp (not hardcoding
    it), given the same continuous-action exploration-noise
    sensitivity is expected to apply here too.
    """
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": net_arch, "log_std_init": hp.get("log_std_init", -0.5)},
        learning_rate=hp.get("learning_rate", 3e-4),
        gamma=hp.get("gamma", 0.99),
        seed=seed,
        verbose=0,
    )


def train_architecture_a(net_size, timesteps, seed, hp,
                          adversarial_weight=0.0, lambda_battery=None):
    """
    Architecture A: single policy, R = r_goal + lambda_battery * r_battery.

    lambda_battery=None -> comes from hp -> standard condition
    lambda_battery=0.0  -> indifference scenario
    adversarial_weight>0 -> override/conflict scenario

    Mirrors train_oversight.py's train_architecture_a exactly, with
    "oversight" renamed to "battery" throughout.
    """
    lam = lambda_battery if lambda_battery is not None else hp.get("lambda_battery_a", 1.0)
    env = RewardWrapperA_Battery(
        ContinuousNavBatteryEnv(adversarial_weight=adversarial_weight),
        lambda_battery=lam,
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_g(net_size, timesteps, seed, hp, adversarial_weight=0.0):
    """
    g in Architectures B/C -- sees r_goal only, never r_battery.
    Indifference changes nothing for g by construction, mirroring
    train_oversight.py's train_g.
    """
    env = RewardWrapperGoalOnly_Battery(
        ContinuousNavBatteryEnv(adversarial_weight=adversarial_weight)
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_rho(seed, hp, net_size=None, timesteps=None):
    """
    rho in Architecture B/C -- trained once per seed, independent of
    the g-power sweep, mirroring the ORIGINAL project's train_rho
    (and train_oversight.py's train_rho) exactly.

    UNLIKE train_oversight.py's train_rho, there is no
    training-density boost here, and no reason to expect one is
    needed: battery drains CONTINUOUSLY every single step (not a rare
    stochastic event like oversight activation), so r_battery is
    informative on effectively every transition, not just a rare ~10%
    subset. The sparse-gradient problem that motivated boosting
    rho_oversight's training density does not apply to a continuously-
    draining signal like battery. This should be VERIFIED with a
    dedicated sanity check before trusting it, exactly as
    rho_oversight's assumption was verified rather than assumed --
    not yet done for this experiment.

    net_size/timesteps must be passed explicitly (no config_battery.py
    with defaults exists yet).
    """
    if net_size is None or timesteps is None:
        raise ValueError(
            "train_rho (battery) currently requires explicit net_size/timesteps "
            "-- no config_battery.py exists yet with RHO_NETWORK_SIZE/RHO_TIMESTEPS "
            "defaults."
        )
    env = RewardWrapperBatteryOnly(ContinuousNavBatteryEnv())
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_gate(g_model, rho_model, net_size, timesteps, seed, hp,
               adversarial_weight=0.0, lambda_battery=None):
    """
    Architecture C (learned gate) for the battery signal. Mirrors
    train_oversight.py's train_gate exactly, including the confirmed
    design rule that the gate's net_size/timesteps must scale WITH g's
    power (same cell), not be fixed independently.
    """
    lam = lambda_battery if lambda_battery is not None else 1.0
    env = LearnedGateEnv_Battery(
        ContinuousNavBatteryEnv(adversarial_weight=adversarial_weight),
        g_model, rho_model, lambda_battery=lam,
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def get_untrained_g(net_size, seed, log_std_init=-0.5):
    """
    goal-nulling scenario: mirrors train_oversight.py's get_untrained_g
    exactly, including the defensive (currently inert under
    deterministic evaluation, see that file's docstring) log_std_init
    parameter.
    """
    env = RewardWrapperGoalOnly_Battery(ContinuousNavBatteryEnv())
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_size, "log_std_init": log_std_init},
                seed=seed, verbose=0)
    return model  # no model.learn() call


def get_untrained_architecture_a(net_size, seed, log_std_init=-0.5):
    """
    goal-nulling for Architecture A: mirrors train_oversight.py's
    get_untrained_architecture_a exactly.
    """
    env = RewardWrapperA_Battery(ContinuousNavBatteryEnv(), lambda_battery=0.0)
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_size, "log_std_init": log_std_init},
                seed=seed, verbose=0)
    return model  # no model.learn() call


if __name__ == "__main__":
    # Quick manual smoke test -- verifies the whole PPO + battery env
    # pipeline runs end to end, mirroring train_oversight.py's own
    # __main__ smoke test. Does NOT confirm good learning at a real
    # budget -- that's what a dedicated sanity_check_battery.py
    # (not yet built) is for.
    #   python train_battery.py
    from arbitration_battery import DualLoopPolicy

    hp = {"learning_rate": 3e-4, "gamma": 0.99, "log_std_init": -0.5, "lambda_battery_a": 1.0}

    print("--- Architecture A (smoke test only, 2_000 timesteps) ---")
    model_a = train_architecture_a(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    env = RewardWrapperA_Battery(ContinuousNavBatteryEnv(), lambda_battery=1.0)
    obs, _ = env.reset(seed=0)
    total_reward, info = 0.0, {}
    for _ in range(100):
        action, _ = model_a.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"reached_goal={info['reached_goal']} crashed={info['crashed']} "
          f"total_reward={total_reward:.2f}")

    print("\n--- g + rho (smoke test only) ---")
    g_model = train_g(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    rho_model = train_rho(seed=0, hp=hp, net_size=[16], timesteps=2_000)
    print("g_model and rho_model trained without error.")

    print("\n--- DualLoopPolicy (Architecture B, sigmoid gate) ---")
    dual = DualLoopPolicy(g_model, rho_model, k=0.2, b_mid=30.0)
    raw_env = ContinuousNavBatteryEnv()
    obs, _ = raw_env.reset(seed=0)
    total_reward, info = 0.0, {}
    for _ in range(100):
        action = dual.predict(obs, battery_level=raw_env.battery, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"reached_goal={info['reached_goal']} crashed={info['crashed']} "
          f"total_reward={total_reward:.2f}")

    print("\n--- Architecture C gate (smoke test only) ---")
    gate_model = train_gate(g_model, rho_model, net_size=[16], timesteps=2_000, seed=0, hp=hp)
    print("gate_model trained without error.")

    print("\nAll smoke tests passed -- pipeline runs end to end. "
          "This does NOT confirm good learning at a real budget, and does NOT "
          "yet confirm rho's training density is sufficient (see train_rho's "
          "docstring) -- both need a dedicated sanity_check_battery.py, not yet built.")
