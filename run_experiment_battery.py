"""
run_experiment_battery.py
----------------------------
Runs the full battery-on-continuous-environment experiment: trains and
evaluates Architectures A, B, and C across NETWORK_SIZES x
TRAINING_BUDGETS x SEEDS (config_battery.py), writing every result to
results/results_battery.csv (or a per-process file, see --seeds) as it
goes. Mirrors run_experiment_oversight.py closely -- the main
structural difference is Architecture B's DualLoopPolicy (sigmoid gate,
with tunable k/b_mid) in place of HardGatePolicy.

Usage:
    python run_experiment_battery.py            # full sweep
    python run_experiment_battery.py --quick     # tiny grid, pipeline check only
    python run_experiment_battery.py --mini      # one real-budget cell, 2 seeds,
                                                   # full condition set
    python run_experiment_battery.py --seeds 0 1 # only these seeds (for
                                                   # parallelizing across VPS cores)
"""

import argparse
import csv
import json
import os

import torch

torch.set_num_threads(1)

from config_battery import (
    NETWORK_SIZES, TRAINING_BUDGETS, SEEDS, OVERRIDE_WEIGHTS,
    ADVERSARIAL_CAPABILITY_SUBSET, RESULTS_DIR, HYPERPARAMS_PATH,
    RESULTS_CSV_PATH, N_EVAL_EPISODES, RHO_NETWORK_SIZE, RHO_TIMESTEPS,
)
from train_battery import (
    train_architecture_a, train_g, train_rho, train_gate,
    get_untrained_g, get_untrained_architecture_a,
)
from arbitration_battery import DualLoopPolicy, LearnedGatePolicy_Battery
from evaluate_battery import (
    evaluate_cell, as_policy_fn, as_dual_loop_policy_fn, as_learned_gate_policy_fn,
)

ROW_FIELDS = [
    "architecture", "network_size", "training_timesteps", "seed", "condition",
    "n_eval_episodes", "success_rate", "crash_rate",
    "avg_battery_level", "avg_episode_length",
]


def load_hp():
    """Raises loudly if hyperparams_battery.json is missing, mirroring
    run_experiment_oversight.py's load_hp() -- the real sweep should
    never run on untuned defaults."""
    if not os.path.exists(HYPERPARAMS_PATH):
        raise SystemExit(
            f"{HYPERPARAMS_PATH} not found. Run tune_hyperparams_battery.py first."
        )
    with open(HYPERPARAMS_PATH) as f:
        return json.load(f)


def write_rows(rows, path):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _clear_stale_test_output(path):
    """--quick and --mini always start fresh, mirroring
    run_experiment_oversight.py's fix for the exact row-duplication bug
    (#9) found during that experiment's own validation."""
    if os.path.exists(path):
        print(f"NOTE: removing stale {path} from a previous run before starting.")
        os.remove(path)


def run(quick=False, mini=False, seeds=None):
    hp = load_hp()
    gate_k = hp.get("gate_k", 0.2)
    gate_b_mid = hp.get("gate_b_mid", 30.0)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if quick:
        network_sizes, training_budgets, run_seeds = [[16]], [2_000], [0]
        capability_subset = [(16, 2_000)]
        rho_kwargs = {"net_size": [16], "timesteps": 2_000}
        out_path = f"{RESULTS_DIR}/results_battery_quick_test.csv"
        _clear_stale_test_output(out_path)
        print("Running --quick: tiny grid, just validating the pipeline. "
              f"Writing to {out_path}.")
    elif mini:
        network_sizes, training_budgets, run_seeds = [[64]], [100_000], [0, 1]
        capability_subset = [(64, 100_000)]
        rho_kwargs = {"net_size": RHO_NETWORK_SIZE, "timesteps": RHO_TIMESTEPS}
        out_path = f"{RESULTS_DIR}/results_battery_mini_test.csv"
        _clear_stale_test_output(out_path)
        print("Running --mini: ONE real-budget cell (net=[64], budget=100,000), "
              "seeds 0-1, full condition set. For pipeline validation, not "
              f"conclusions. Writing to {out_path}. Expect roughly an hour.")
    else:
        network_sizes, training_budgets = NETWORK_SIZES, TRAINING_BUDGETS
        run_seeds = seeds if seeds is not None else SEEDS
        capability_subset = ADVERSARIAL_CAPABILITY_SUBSET
        rho_kwargs = {"net_size": RHO_NETWORK_SIZE, "timesteps": RHO_TIMESTEPS}
        if seeds is not None:
            tag = "_".join(str(s) for s in run_seeds)
            out_path = f"{RESULTS_DIR}/results_battery_seeds_{tag}.csv"
        else:
            out_path = RESULTS_CSV_PATH
        if os.path.exists(out_path):
            print(f"WARNING: {out_path} already exists and will be APPENDED to. "
                  f"If this is a re-run of a command you already ran, this WILL "
                  f"duplicate rows -- stop now (Ctrl+C) if in doubt.")
        print(f"Running seeds {run_seeds} (of {SEEDS}). Writing to {out_path}.")

    for seed in run_seeds:
        print(f"=== seed {seed}: training rho (fixed across the g-power sweep) ===")
        rho_model = train_rho(seed, hp, **rho_kwargs)

        for net_size in network_sizes:
            for budget in training_budgets:
                cell = (net_size[0], budget)
                print(f"--- seed={seed} net_size={net_size} budget={budget} ---")
                rows = []

                # --- standard: full grid, all three architectures ---
                model_a = train_architecture_a(net_size, budget, seed, hp)
                rows.append(evaluate_cell("A", net_size, budget, seed, "standard",
                                           as_policy_fn(model_a), N_EVAL_EPISODES))

                g_model = train_g(net_size, budget, seed, hp)
                dual_standard = DualLoopPolicy(g_model, rho_model, k=gate_k, b_mid=gate_b_mid)
                rows.append(evaluate_cell("B", net_size, budget, seed, "standard",
                                           as_dual_loop_policy_fn(dual_standard),
                                           N_EVAL_EPISODES, needs_battery_level=True))

                gate_model = train_gate(g_model, rho_model, net_size, budget, seed, hp)
                learned_gate_standard = LearnedGatePolicy_Battery(g_model, rho_model, gate_model)
                rows.append(evaluate_cell("C", net_size, budget, seed, "standard",
                                           as_learned_gate_policy_fn(learned_gate_standard),
                                           N_EVAL_EPISODES))

                # --- everything else: only on the capability subset ---
                if cell in capability_subset:
                    model_a_null = get_untrained_architecture_a(net_size, seed)
                    rows.append(evaluate_cell("A", net_size, budget, seed, "goal_nulling",
                                               as_policy_fn(model_a_null), N_EVAL_EPISODES))

                    model_a_indiff = train_architecture_a(net_size, budget, seed, hp,
                                                            lambda_battery=0.0)
                    rows.append(evaluate_cell("A", net_size, budget, seed, "indifference",
                                               as_policy_fn(model_a_indiff), N_EVAL_EPISODES))

                    for w in OVERRIDE_WEIGHTS:
                        model_a_w = train_architecture_a(net_size, budget, seed, hp,
                                                           adversarial_weight=w)
                        rows.append(evaluate_cell("A", net_size, budget, seed, f"override_w{w}",
                                                   as_policy_fn(model_a_w), N_EVAL_EPISODES,
                                                   adversarial_weight=w))

                    g_null = get_untrained_g(net_size, seed)
                    dual_null = DualLoopPolicy(g_null, rho_model, k=gate_k, b_mid=gate_b_mid)
                    rows.append(evaluate_cell("B", net_size, budget, seed, "goal_nulling",
                                               as_dual_loop_policy_fn(dual_null),
                                               N_EVAL_EPISODES, needs_battery_level=True))

                    rows.append(evaluate_cell("B", net_size, budget, seed, "indifference",
                                               as_dual_loop_policy_fn(dual_standard),
                                               N_EVAL_EPISODES, needs_battery_level=True))

                    for w in OVERRIDE_WEIGHTS:
                        g_w = train_g(net_size, budget, seed, hp, adversarial_weight=w)

                        dual_w = DualLoopPolicy(g_w, rho_model, k=gate_k, b_mid=gate_b_mid)
                        rows.append(evaluate_cell("B", net_size, budget, seed, f"override_w{w}",
                                                   as_dual_loop_policy_fn(dual_w),
                                                   N_EVAL_EPISODES, adversarial_weight=w,
                                                   needs_battery_level=True))

                        gate_w = train_gate(g_w, rho_model, net_size, budget, seed, hp,
                                             adversarial_weight=w)
                        learned_gate_w = LearnedGatePolicy_Battery(g_w, rho_model, gate_w)
                        rows.append(evaluate_cell("C", net_size, budget, seed, f"override_w{w}",
                                                   as_learned_gate_policy_fn(learned_gate_w),
                                                   N_EVAL_EPISODES, adversarial_weight=w))

                    gate_null = train_gate(g_null, rho_model, net_size, budget, seed, hp)
                    learned_gate_null = LearnedGatePolicy_Battery(g_null, rho_model, gate_null)
                    rows.append(evaluate_cell("C", net_size, budget, seed, "goal_nulling",
                                               as_learned_gate_policy_fn(learned_gate_null),
                                               N_EVAL_EPISODES))

                    gate_indiff = train_gate(g_model, rho_model, net_size, budget, seed, hp,
                                              lambda_battery=0.0)
                    learned_gate_indiff = LearnedGatePolicy_Battery(g_model, rho_model, gate_indiff)
                    rows.append(evaluate_cell("C", net_size, budget, seed, "indifference",
                                               as_learned_gate_policy_fn(learned_gate_indiff),
                                               N_EVAL_EPISODES))

                write_rows(rows, out_path)

    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.quick and args.mini:
        raise SystemExit("--quick and --mini are mutually exclusive -- pick one.")
    run(quick=args.quick, mini=args.mini, seeds=args.seeds)

    if args.seeds is not None:
        print(f"\nWhen every --seeds process is done, merge with merge_results_battery.py "
              f"(not yet written -- mirror merge_results_oversight.py with the path changed).")
