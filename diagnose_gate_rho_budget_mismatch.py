"""
diagnose_gate_rho_budget_mismatch.py
--------------------------------------
Focused diagnostic, NOT part of the final pipeline -- isolates ONE
question before Architecture B's weakness is written up as an
architectural finding: does B's poor showing come from a genuine
sigmoid-direction/architectural issue, or partly from a mismatch
between the rho budget the gate (gate_k/gate_b_mid) was TUNED against
and the rho budget the REAL sweep actually uses?

The mismatch, found during code review:
  - tune_hyperparams_battery.py's tune_gate_params() trains its
    validation rho at VALIDATION_TIMESTEPS = 200,000 steps, and picks
    gate_k/gate_b_mid to work well against THAT rho.
  - config_battery.py's RHO_TIMESTEPS = 50,000 -- the real sweep's rho
    is trained at 1/4 of the budget the gate's k/b_mid were chosen
    for.

If Architecture B depends on rho being reasonably converged (plausible,
since the gate's whole job is deciding how much to trust rho), then
gate_k/gate_b_mid tuned against a better rho could genuinely
underperform when paired with a weaker rho at 50k -- which would mean
part of B's measured weakness is a budget-matching artifact, not (or
not only) the counter-intuitive sigmoid direction inherited from the
original project.

This script trains rho at BOTH 50k and 200k (several seeds each, using
the SAME validation seeds as tune_hyperparams_battery.py, disjoint from
the real experiment's SEEDS 0-9), evaluates Architecture B (using the
ALREADY-TUNED gate_k/gate_b_mid from results/hyperparams_battery.json)
against each, and reports whether crash_rate/success_rate differ
meaningfully between the two.

Usage:
    python diagnose_gate_rho_budget_mismatch.py

Expected runtime: 3 seeds x 2 rho budgets (50k + 200k) x ~1 g training
(100k, matching --mini's cell) -- roughly the same order of magnitude
as diagnose_rho_instability.py. Run time_probe.py-derived estimates
first if in doubt.
"""

import json
import os

import numpy as np

from config_battery import HYPERPARAMS_PATH, RHO_NETWORK_SIZE
from env_battery_continuous import ContinuousNavBatteryEnv
from train_battery import train_g, train_rho
from arbitration_battery import DualLoopPolicy
from sanity_check_battery import rollout, as_dual_loop_policy_fn

# Disjoint from the real sweep's SEEDS (0-9), matches
# tune_hyperparams_battery.py's VALIDATION_SEEDS.
DIAG_SEEDS = [100, 101, 102]

# The two rho budgets under comparison.
RHO_BUDGET_REAL_SWEEP = 50_000   # what config_battery.RHO_TIMESTEPS actually is
RHO_BUDGET_GATE_TUNED = 200_000  # what tune_hyperparams_battery.VALIDATION_TIMESTEPS
                                  # actually trained rho at, when gate_k/gate_b_mid
                                  # were chosen

# A representative g -- net size / budget matching --mini's validation
# cell, not the full sweep, since this is a targeted diagnostic, not a
# re-run of the whole grid.
G_NET_SIZE = [64]
G_BUDGET = 100_000

N_EVAL_EPISODES = 100


def load_hp_and_gate():
    if not os.path.exists(HYPERPARAMS_PATH):
        raise SystemExit(
            f"{HYPERPARAMS_PATH} not found -- run tune_hyperparams_battery.py first. "
            f"This diagnostic evaluates the ALREADY-TUNED gate_k/gate_b_mid against "
            f"two different rho budgets, so it needs that file to exist."
        )
    with open(HYPERPARAMS_PATH) as f:
        hp = json.load(f)
    gate_k = hp.get("gate_k")
    gate_b_mid = hp.get("gate_b_mid")
    if gate_k is None or gate_b_mid is None:
        raise SystemExit(
            f"{HYPERPARAMS_PATH} has no gate_k/gate_b_mid -- make sure "
            f"tune_hyperparams_battery.py's tune_gate_params() actually ran and "
            f"was saved, not an older/partial hyperparams file."
        )
    return hp, gate_k, gate_b_mid


def run_one_seed(seed, hp, gate_k, gate_b_mid):
    print(f"\n--- seed {seed} ---")
    print(f"  training g (net={G_NET_SIZE}, budget={G_BUDGET})...")
    g_model = train_g(G_NET_SIZE, G_BUDGET, seed, hp)

    results = {}
    for label, rho_budget in (
        ("real_sweep_50k", RHO_BUDGET_REAL_SWEEP),
        ("gate_tuned_200k", RHO_BUDGET_GATE_TUNED),
    ):
        print(f"  training rho at {rho_budget} steps ({label})...")
        rho_model = train_rho(seed, hp, net_size=RHO_NETWORK_SIZE, timesteps=rho_budget)

        dual = DualLoopPolicy(g_model, rho_model, k=gate_k, b_mid=gate_b_mid)
        metrics = rollout(as_dual_loop_policy_fn(dual), n_episodes=N_EVAL_EPISODES,
                           seed=seed * 1000, needs_battery_level=True)
        print(f"    B: success_rate={metrics['success_rate']:.3f} "
              f"crash_rate={metrics['crash_rate']:.3f}")
        results[label] = metrics

    return results


if __name__ == "__main__":
    hp, gate_k, gate_b_mid = load_hp_and_gate()
    print(f"Using TUNED gate_k={gate_k}, gate_b_mid={gate_b_mid} from {HYPERPARAMS_PATH}")
    print(f"Comparing Architecture B's performance with rho trained at "
          f"{RHO_BUDGET_REAL_SWEEP} steps (what the real sweep actually uses) vs. "
          f"{RHO_BUDGET_GATE_TUNED} steps (what the gate's k/b_mid were tuned against).")
    print(f"g held fixed per seed: net_size={G_NET_SIZE}, budget={G_BUDGET}.")
    print(f"Seeds: {DIAG_SEEDS} (disjoint from the real sweep's SEEDS 0-9).\n")

    all_results = {s: run_one_seed(s, hp, gate_k, gate_b_mid) for s in DIAG_SEEDS}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    real_success = [all_results[s]["real_sweep_50k"]["success_rate"] for s in DIAG_SEEDS]
    real_crash = [all_results[s]["real_sweep_50k"]["crash_rate"] for s in DIAG_SEEDS]
    tuned_success = [all_results[s]["gate_tuned_200k"]["success_rate"] for s in DIAG_SEEDS]
    tuned_crash = [all_results[s]["gate_tuned_200k"]["crash_rate"] for s in DIAG_SEEDS]

    print(f"\n{'':20} {'success_rate':>15} {'crash_rate':>15}")
    print(f"{'rho @ 50k (real)':20} {np.mean(real_success):>15.3f} {np.mean(real_crash):>15.3f}")
    print(f"{'rho @ 200k (tuned)':20} {np.mean(tuned_success):>15.3f} {np.mean(tuned_crash):>15.3f}")

    d_success = np.mean(tuned_success) - np.mean(real_success)
    d_crash = np.mean(real_crash) - np.mean(tuned_crash)  # positive = crash improves with more rho budget

    print(f"\nDelta (200k minus 50k): success_rate {d_success:+.3f}, "
          f"crash_rate improvement {d_crash:+.3f}")

    if abs(d_success) < 0.05 and abs(d_crash) < 0.05:
        print(
            "\n-> NO meaningful difference between rho budgets. The gate's "
            "gate_k/gate_b_mid tuned against a 200k rho perform about the same "
            "paired with a 50k rho. This means the budget mismatch is NOT "
            "driving Architecture B's weakness -- the counter-intuitive sigmoid "
            "direction (inherited unchanged from the original project) remains "
            "the more likely explanation, and B's weakness can be written up "
            "as a real architectural finding without this caveat."
        )
    else:
        print(
            "\n-> MEANINGFUL difference between rho budgets. Architecture B's "
            "measured weakness is at least PARTLY a consequence of gate_k/gate_b_mid "
            "being tuned against a better-trained rho (200k) than the real sweep "
            "actually uses (50k) -- not purely an architectural property of the "
            "sigmoid direction. Before writing B's weakness up as a clean "
            "architectural finding, consider: (a) re-running "
            "tune_hyperparams_battery.py's tune_gate_params() with rho trained at "
            "the REAL 50k budget instead of 200k, so the gate is tuned against "
            "what it will actually be paired with, or (b) raising "
            "config_battery.RHO_TIMESTEPS to 200k for the real sweep to match what "
            "the gate was tuned against. Either fixes the mismatch; which one is "
            "more faithful to the experiment's intent is a design call, not a "
            "coding one -- flagging rather than deciding here."
        )
