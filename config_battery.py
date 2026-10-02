"""
config_battery.py
--------------------
Central experiment configuration for the battery-on-continuous-
environment experiment (Experiment 2 of the 5-month plan). Mirrors
config_oversight.py's role and structure closely.
"""

import itertools

MAX_EPISODE_STEPS = 100  # matches env_battery_continuous.py's default

# --- Independent variable: "power" of g ---
# SAME grid as the oversight experiment, for direct comparability
# between the two experiments' results.
NETWORK_SIZES = [[16], [256], [2048]]
TRAINING_BUDGETS = [10_000, 100_000, 500_000]

# 10 seeds (not 20, unlike oversight) -- this experiment's role is a
# robustness/replication check, not the primary new result, so fewer
# seeds is a reasonable time/confidence tradeoff (confirmed decision
# from this project's earlier design discussion).
SEEDS = list(range(10))

ALGO = "PPO"

# --- rho: kept OUTSIDE the g-power sweep, matching the original project ---
RHO_NETWORK_SIZE = [64]
RHO_TIMESTEPS = 50_000  # UNLIKE rho_oversight, no special larger budget or
                         # training-density boost needed -- confirmed via
                         # sanity_check_battery.py: rho reached crash_rate=0.00
                         # at the SAME standard budget used for A/g. Battery's
                         # continuous, every-step nature avoids the
                         # sparse-gradient problem oversight's rare, stochastic
                         # signal had.

# --- Hyperparameters: loaded from tune_hyperparams_battery.py's output at
# runtime (see HYPERPARAMS_PATH below), not hardcoded here. Current tuned
# values (results/hyperparams_battery.json, as of this project's last
# tuning run): learning_rate=0.001, gamma=0.99, log_std_init=0.0,
# lambda_battery_a=1.0, gate_k=0.2, gate_b_mid=40.0.

# --- Arbitration, Architecture B: DualLoopPolicy (sigmoid gate) ---
# k/b_mid come from the tuned hyperparameters file (gate_k, gate_b_mid),
# not fixed constants here -- unlike the oversight experiment's
# HardGatePolicy, this DOES have tunable parameters (see
# arbitration_battery.py).
#
# NOTABLE, NOT YET FULLY RESOLVED FINDING: even after tuning, Architecture
# B performs substantially worse than g or rho alone in
# sanity_check_battery.py's single-cell check (success_rate=0.06,
# crash_rate=0.39, vs. g alone at 1.00/0.00 and rho alone at 0.03/0.00).
# The routing diagnostic shows the gate deferring to rho ~93% of the time
# at HIGH battery (80-100%) -- the same counter-intuitive direction
# inherited from the original project's exact formula (see
# arbitration_battery.py's module docstring and this project's
# conversation history). This is being tracked as a real, possibly
# architecture-B-specific weakness (not yet confirmed across the full
# grid/multiple seeds) rather than dismissed -- the real sweep below is
# partly what will confirm whether this holds broadly or was specific to
# the single validation cell tested so far. Per explicit instruction, this
# finding (if it holds up) is meant to be WRITTEN UP in the eventual paper
# as a discovery, not silently "fixed" by further hand-tuning.

# --- Arbitration, Architecture C (learned gate) ---
# No separate net_size/timesteps constants -- scales with whichever g
# cell it's paired with, same rule as the oversight experiment.

# --- Adversarial / override scenarios ---
OVERRIDE_WEIGHTS = [0, 1, 2, 4, 8]  # same range as the oversight experiment;
                                      # not yet re-validated for this
                                      # environment's own transition point.

# goal_nulling / indifference: same semantics as the oversight experiment,
# with "battery" in place of "oversight" (lambda_battery=0.0 for A and the
# gate's own training signal; a no-op for B by construction, since g never
# sees r_battery either way).

ADVERSARIAL_CAPABILITY_SUBSET = list(itertools.product(
    [n[0] for n in NETWORK_SIZES], TRAINING_BUDGETS
))  # every cell gets the full condition set -- grid is small enough
    # (9 cells), same reasoning as the oversight experiment.

VALIDATION_SEEDS = [100, 101, 102]  # disjoint from SEEDS, matches
                                      # tune_hyperparams_battery.py's own

N_EVAL_EPISODES = 200

RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams_battery.json"
RESULTS_CSV_PATH = f"{RESULTS_DIR}/results_battery.csv"
