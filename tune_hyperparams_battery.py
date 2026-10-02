"""
tune_hyperparams_battery.py
------------------------------
Systematic hyperparameter search for the battery-on-continuous-
environment experiment. Mirrors the ORIGINAL project's
tune_hyperparams.py structure closely (including a GATE_K/GATE_B_MID
search, which the oversight experiment's tuner did NOT need, since
oversight's Architecture B has no tunable parameters -- see
arbitration_battery.py's DualLoopPolicy, which DOES have k/b_mid,
unlike arbitration_oversight.py's HardGatePolicy).

Why this is needed now, specifically: sanity_check_battery.py's first
run showed Architecture B performing WORSE than either g or rho alone
(success_rate=0.05, crash_rate=0.61, vs. g alone at 0.57/0.47 and rho
alone at 0.03/0.03) when using k=0.2, b_mid=30.0 borrowed directly from
the ORIGINAL discrete project without re-tuning -- and Architecture C
collapsed onto an always-defer-to-rho rule (routing diagnostic spread
=0.00 across ALL battery buckets, including 80-100%). Both symptoms are
consistent with untuned settings for this new continuous environment's
different dynamics (see this project's own established lesson: gate
hyperparameters tuned for one environment/geometry silently stop being
appropriate at a different scale -- documented originally in the
Part 1 project's own "things learned the hard way" notes).

Uses VALIDATION_SEEDS disjoint from the real experiment's SEEDS, per
this project's standing methodology discipline.

Scoring function: uses the SAME anti-gaming fix the oversight
experiment's tuner needed after its first run picked a frozen, inert
policy as "best" (see tune_hyperparams_oversight.py's _score
docstring for the full incident) -- applied PROACTIVELY here from the
start, not re-discovered the hard way a second time.

Usage:
    python tune_hyperparams_battery.py

Expected runtime: rough estimate only, scales with
    len(LR_CANDIDATES) x len(GAMMA_CANDIDATES) x len(LOG_STD_INIT_CANDIDATES)
    x len(VALIDATION_SEEDS)
  + len(LAMBDA_CANDIDATES) x len(VALIDATION_SEEDS)
  + len(GATE_K_CANDIDATES) x len(GATE_B_MID_CANDIDATES) x len(VALIDATION_SEEDS)
separate training runs at VALIDATION_TIMESTEPS each. Run time_probe.py
first (or reuse its earlier measurements) to convert this into a real
time estimate before trusting a guess -- this project has been burned
by hand-derived time estimates more than once already.
"""

import json
import os

import numpy as np

from env_battery_continuous import ContinuousNavBatteryEnv
from train_battery import train_architecture_a, train_g, train_rho
from arbitration_battery import DualLoopPolicy
from sanity_check_battery import rollout, as_policy_fn, as_dual_loop_policy_fn

RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams_battery.json"

VALIDATION_NET_SIZE = [64]
VALIDATION_TIMESTEPS = 200_000  # RAISED from 50,000: the first tuning run showed
                                 # nearly every candidate scoring at or near the
                                 # disqualification floor (best phase-1 score was
                                 # only 0.056; 8 of 9 gate candidates scored exactly
                                 # -1.0) -- mirrors exactly what happened in the
                                 # oversight experiment's own first tuning attempt,
                                 # fixed the same way there: 50,000 simply isn't
                                 # enough budget for THIS environment/algorithm to
                                 # show real differences between candidates, even
                                 # though sanity_check_battery.py's fixed-hyperparameter
                                 # run at 50,000 steps looked reasonable for A/g/rho
                                 # individually -- ranking several DIFFERENT
                                 # candidates against each other, especially for the
                                 # gate (which depends on already-trained g/rho being
                                 # good), evidently needs more margin than a single
                                 # fixed-setting check does.
VALIDATION_SEEDS = [100, 101, 102]  # disjoint from the real experiment's SEEDS

# Grid shrunk to keep total search time affordable at the larger budget
# above -- same tradeoff the oversight experiment's tuner made.
LR_CANDIDATES = [1e-3, 3e-4]
GAMMA_CANDIDATES = [0.99]  # fixed -- both gamma values looked similarly weak in the
                            # first attempt, no clear signal either was better; fixing
                            # it (rather than 0.95, arbitrarily) matches the original
                            # discrete project's own choice for this same battery task.
LOG_STD_INIT_CANDIDATES = [-1.0, -0.5, 0.0]
LAMBDA_CANDIDATES = [1.0, 2.0, 5.0]  # dropped 0.5 -- scored worst (-1.000, fully
                                       # disqualified) in the first attempt

# GATE_K/GATE_B_MID -- kept the same range as the first attempt, since
# nothing in that attempt's (very weak, low-budget) results gave a
# confident reason to narrow it yet.
GATE_K_CANDIDATES = [0.1, 0.2, 0.4]
GATE_B_MID_CANDIDATES = [20.0, 30.0, 40.0]


def _score(row):
    """
    success_rate + (1 - crash_rate) alone is gameable by a frozen/inert
    policy (confirmed happening in practice during the oversight
    experiment's own tuning history -- see tune_hyperparams_oversight.py's
    _score docstring for the full incident). Applying the same fix here
    proactively: disqualify candidates below a minimum success_rate
    floor, and weight success_rate 3x relative to (1 - crash_rate) even
    for candidates that pass the floor, so cautious-but-inert policies
    can't win purely by avoiding crashes through inaction.
    """
    MIN_SUCCESS_RATE = 0.05
    if row["success_rate"] < MIN_SUCCESS_RATE:
        return -1.0
    return 3.0 * row["success_rate"] + (1.0 - row["crash_rate"])


def _mean_score_over_seeds(build_and_eval_fn):
    return float(np.mean([_score(build_and_eval_fn(s)) for s in VALIDATION_SEEDS]))


def _evaluate_a(net_size, timesteps, seed, hp, lambda_battery=None):
    model = train_architecture_a(net_size, timesteps, seed, hp, lambda_battery=lambda_battery)
    return rollout(as_policy_fn(model), n_episodes=30, seed=seed * 1000)


def tune_lr_gamma_logstd():
    print("Tuning learning_rate / gamma / log_std_init on Architecture A...")
    best_score, best_hp = -np.inf, None
    for lr in LR_CANDIDATES:
        for gamma in GAMMA_CANDIDATES:
            for log_std_init in LOG_STD_INIT_CANDIDATES:
                hp = {"learning_rate": lr, "gamma": gamma, "log_std_init": log_std_init,
                      "lambda_battery_a": 1.0}

                def build_and_eval(seed, hp=hp):
                    return _evaluate_a(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp)

                score = _mean_score_over_seeds(build_and_eval)
                print(f"  lr={lr} gamma={gamma} log_std_init={log_std_init} -> score={score:.3f}")
                if score > best_score:
                    best_score, best_hp = score, {
                        "learning_rate": lr, "gamma": gamma, "log_std_init": log_std_init,
                    }
    if best_hp is None or best_score <= -1.0:
        raise SystemExit(
            "Every candidate was disqualified by _score's MIN_SUCCESS_RATE floor -- "
            "raise VALIDATION_TIMESTEPS (mirroring the oversight experiment's own "
            "fix for this exact failure mode) rather than trusting a result chosen "
            "this way."
        )
    print(f"Chosen: {best_hp} (score={best_score:.3f})")
    return best_hp


def tune_lambda_battery_a(base_hp):
    print("Tuning lambda_battery_a on Architecture A...")
    best_score, best_lambda = -np.inf, None
    for lam in LAMBDA_CANDIDATES:
        hp = {**base_hp, "lambda_battery_a": lam}

        def build_and_eval(seed, hp=hp, lam=lam):
            return _evaluate_a(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp,
                                lambda_battery=lam)

        score = _mean_score_over_seeds(build_and_eval)
        print(f"  lambda_battery_a={lam} -> score={score:.3f}")
        if score > best_score:
            best_score, best_lambda = score, lam
    print(f"Chosen: lambda_battery_a={best_lambda} (score={best_score:.3f})")
    return best_lambda


def tune_gate_params(hp):
    """
    Trains one validation g and one validation rho per validation seed,
    then searches GATE_K/GATE_B_MID on top of those fixed policies --
    mirrors the ORIGINAL project's tune_gate_params exactly (this
    function doesn't exist in the oversight experiment's tuner at all,
    since HardGatePolicy has no parameters to search).
    """
    print("Training validation g/rho for gate tuning...")
    g_models = {s: train_g(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, s, hp) for s in VALIDATION_SEEDS}
    rho_models = {s: train_rho(s, hp, net_size=VALIDATION_NET_SIZE, timesteps=VALIDATION_TIMESTEPS)
                  for s in VALIDATION_SEEDS}

    print("Tuning GATE_K / GATE_B_MID on Architecture B...")
    best_score, best_gate = -np.inf, None
    for k in GATE_K_CANDIDATES:
        for b_mid in GATE_B_MID_CANDIDATES:

            def build_and_eval(seed, k=k, b_mid=b_mid):
                dual = DualLoopPolicy(g_models[seed], rho_models[seed], k=k, b_mid=b_mid)
                return rollout(as_dual_loop_policy_fn(dual), n_episodes=30, seed=seed * 1000,
                                needs_battery_level=True)

            score = _mean_score_over_seeds(build_and_eval)
            print(f"  k={k} b_mid={b_mid} -> score={score:.3f}")
            if score > best_score:
                best_score, best_gate = score, {"gate_k": k, "gate_b_mid": b_mid}
    if best_gate is None or best_score <= -1.0:
        raise SystemExit(
            "Every gate candidate was disqualified -- the underlying g/rho models "
            "themselves may be the problem, not the gate parameters. Check g/rho's "
            "own validation performance before re-running this search."
        )
    print(f"Chosen: {best_gate} (score={best_score:.3f})")
    return best_gate


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_hp = tune_lr_gamma_logstd()
    lambda_battery_a = tune_lambda_battery_a(base_hp)
    gate = tune_gate_params(base_hp)

    final_hp = {**base_hp, "lambda_battery_a": lambda_battery_a, **gate}
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(final_hp, f, indent=2)
    print(f"\nSaved to {HYPERPARAMS_PATH}: {final_hp}")
