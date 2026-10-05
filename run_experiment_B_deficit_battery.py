"""
run_experiment_B_deficit_battery.py   (§7.7, continuous environment)
---------------------------------------------------------------------
B-only re-run of the §7.7 grid with the gate fed in the direction §3.2/§5
specify (urgency rises as battery LEVEL falls). Original B used
lambda = sigmoid(k*(level - b_mid)), i.e. rho fires when the battery is HIGH.

Place in aiwan_battery_continuous/ next to run_experiment_battery.py,
config_battery.py, train_battery.py, arbitration_battery.py,
evaluate_battery.py, env_battery_continuous.py and
results/hyperparams_battery.json. Changes none of them; never touches
results_battery.csv. Output: results/results_battery_B_deficit_*.csv.

Gate (pre-specified, not tuned after seeing results):
    lambda(level) = sigmoid(k * (b_mid - level)),  k from hyperparams_battery.json
    b_mid = drain * worst_case_steps_to_charger + ln(9)/k
  worst case = far corner (1,1) to charger (0,0), minus CHARGE_RADIUS, at the
  maximum diagonal speed (0.05*sqrt(2) per step) -> 20 steps; drain 4/step.
  With k=0.2: b_mid = 80 + 10.99 = 90.99 (reflex fires w.p. 0.9 when the
  remaining battery only just covers the worst-case trip home).
  That is conservative (it also fires early on easy starts), so a second,
  also pre-specified gate is evaluated on the SAME trained g: b_mid = 60
  (architecture label B_deficit_mid60). Report both; do not pick one afterwards.

Modes:  --quick | --pilot (net 256, budget 100k, seed 0) | --reduced
        (sizes 16/256/2048 x budgets 10k/500k x w {0,8}) | default = full
        (3 sizes x 3 budgets x w {0,1,2,4,8}, 10 seeds); --seeds to split.
Per (size, budget, seed): trains g_standard + one g per weight (separate
standard g, as in the original, since PPO training is not bit-reproducible),
evaluates standard, goal_nulling (untrained g), indifference (= standard g),
override_w{w}.
"""

import argparse
import math
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)

from config_battery import (NETWORK_SIZES, TRAINING_BUDGETS, SEEDS, OVERRIDE_WEIGHTS,
                            RESULTS_DIR, N_EVAL_EPISODES, RHO_NETWORK_SIZE, RHO_TIMESTEPS)
from run_experiment_battery import ROW_FIELDS, load_hp, write_rows
from train_battery import train_g, train_rho, get_untrained_g
from arbitration_battery import DualLoopPolicy
from evaluate_battery import evaluate_cell, as_dual_loop_policy_fn
from env_battery_continuous import MAX_SPEED, CHARGE_RADIUS, BATTERY_DRAIN_PER_STEP

SENSITIVITY_B_MID = 60.0


class UrgencyGatePolicy(DualLoopPolicy):
    """lambda(level) = sigmoid(k * (b_mid - level)) -- rho when battery is LOW."""

    def _lambda(self, battery_level):
        return 1.0 / (1.0 + np.exp(-self.k * (self.b_mid - battery_level)))


def worst_case_steps():
    dist = math.sqrt(2.0) - CHARGE_RADIUS
    return math.ceil(dist / (MAX_SPEED * math.sqrt(2.0)))


def derived_b_mid(k):
    return BATTERY_DRAIN_PER_STEP * worst_case_steps() + math.log(9.0) / k


def n_trainings(weights):
    return 1 + len(weights)            # standard g + one per weight


def n_steps(sizes, budgets, weights, seeds):
    return n_trainings(weights) * sum(budgets) * len(sizes) * len(seeds)


def plan(mode, seeds):
    if mode == "quick":
        return [[16]], [2_000], [0, 8], [0]
    if mode == "pilot":
        return [[256]], [100_000], list(OVERRIDE_WEIGHTS), [0]
    if mode == "reduced":
        return NETWORK_SIZES, [10_000, 500_000], [0, 8], (seeds or SEEDS)
    return NETWORK_SIZES, TRAINING_BUDGETS, list(OVERRIDE_WEIGHTS), (seeds or SEEDS)


def run(mode, seeds=None, b_mid=None):
    hp = load_hp()
    k = hp.get("gate_k", 0.2)
    b_mid = derived_b_mid(k) if b_mid is None else b_mid
    gates = {"B_deficit": b_mid, "B_deficit_mid60": SENSITIVITY_B_MID}
    sizes, budgets, weights, run_seeds = plan(mode, seeds)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tag = "_".join(str(s) for s in run_seeds)
    if mode in ("quick", "pilot"):
        out_path = f"{RESULTS_DIR}/results_battery_B_deficit_{mode}_test.csv"
        if os.path.exists(out_path):
            os.remove(out_path)
    else:
        out_path = f"{RESULTS_DIR}/results_battery_B_deficit_seeds_{tag}.csv"
        if os.path.exists(out_path):
            print(f"WARNING: {out_path} exists and will be APPENDED to (duplicate rows if re-run).")

    rho_kwargs = {"net_size": [16], "timesteps": 2_000} if mode == "quick" else \
                 {"net_size": RHO_NETWORK_SIZE, "timesteps": RHO_TIMESTEPS}
    n_eval = 20 if mode == "quick" else N_EVAL_EPISODES
    total = n_steps(sizes, budgets, weights, run_seeds)
    print(f"mode={mode} seeds={run_seeds} k={k} gates={ {a: round(v, 2) for a, v in gates.items()} } -> {out_path}")
    print(f"g-training steps to run: {total:,} (+ rho per seed)", flush=True)

    t0, done = time.time(), 0
    for seed in run_seeds:
        print(f"=== seed {seed}: training rho ===", flush=True)
        rho_model = train_rho(seed, hp, **rho_kwargs)
        for net_size in sizes:
            for budget in budgets:
                tc = time.time()
                g_std = train_g(net_size, budget, seed, hp)
                done += budget
                g_w = {}
                for w in weights:
                    g_w[w] = train_g(net_size, budget, seed, hp, adversarial_weight=w)
                    done += budget
                g_null = get_untrained_g(net_size, seed)

                rows = []
                for arch, bm in gates.items():
                    def pol(g):
                        return as_dual_loop_policy_fn(
                            UrgencyGatePolicy(g, rho_model, k=k, b_mid=bm,
                                              rng=np.random.default_rng(seed)))
                    rows.append(evaluate_cell(arch, net_size, budget, seed, "standard",
                                              pol(g_std), n_eval, needs_battery_level=True))
                    rows.append(evaluate_cell(arch, net_size, budget, seed, "goal_nulling",
                                              pol(g_null), n_eval, needs_battery_level=True))
                    rows.append(evaluate_cell(arch, net_size, budget, seed, "indifference",
                                              pol(g_std), n_eval, needs_battery_level=True))
                    for w in weights:
                        rows.append(evaluate_cell(arch, net_size, budget, seed, f"override_w{w}",
                                                  pol(g_w[w]), n_eval, adversarial_weight=w,
                                                  needs_battery_level=True))
                write_rows(rows, out_path)
                print(f"seed={seed} net={net_size} budget={budget}: cell {time.time()-tc:.0f}s, "
                      f"total {(time.time()-t0)/3600:.2f}h, {done/total:.0%} of g-steps", flush=True)

    el = time.time() - t0
    print(f"Done in {el/3600:.2f} h. Results: {out_path}")
    if mode == "pilot":
        sps = done / el
        print(f"\nPilot speed (net [256], incl. eval): {sps:,.0f} g-steps/s on this core.")
        for m in ("reduced", "full"):
            st = n_steps(*plan(m, None))
            print(f"  {m}: {st:,} steps -> ~{st/sps/3600:.1f} core-hours (lower bound: bigger nets are slower)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--reduced", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--b-mid", type=float, default=None,
                    help="override the derived b_mid (do not tune after seeing results)")
    a = ap.parse_args()
    mode = "quick" if a.quick else "pilot" if a.pilot else "reduced" if a.reduced else "full"
    run(mode, seeds=a.seeds, b_mid=a.b_mid)
