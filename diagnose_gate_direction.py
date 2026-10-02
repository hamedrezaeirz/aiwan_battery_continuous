"""
diagnose_gate_direction.py
--------------------------
Standalone diagnostic (does NOT touch any sweep code or results).

Question: is Architecture B's weakness caused by the direction in which
battery information enters the sigmoid gate?

    lambda(b) = sigmoid(k * (b - b_mid))

  - "level"   : b = battery LEVEL (0..100, high = full). This is what
                evaluate_battery.py passes today (env.battery). With this
                input the gate defers to rho when the battery is HIGH.
  - "deficit" : b = BATTERY_MAX - battery (distance from full charge).
                This is how the paper's Section 3.2 defines b_t
                ("distance from full battery charge"). With this input the
                gate defers to rho when the battery is LOW.

Both use the same tuned k, b_mid (and a small b_mid check for "deficit"),
the same g and rho per seed, the same evaluation episodes, and the
validation seeds (100-102), disjoint from the sweep's seeds 0-9.

Usage (from the battery project folder, venv active):
    python3 diagnose_gate_direction.py
"""

import json
import numpy as np

from train_battery import train_g, train_rho
from arbitration_battery import DualLoopPolicy
from evaluate_battery import rollout

from env_battery_continuous import BATTERY_MAX

HP_PATH = "results/hyperparams_battery.json"
SEEDS = [100, 101, 102]
NET = [64]
G_BUDGET = 100_000
RHO_BUDGET = 50_000          # what the real sweep uses
N_EPISODES = 100
DEFICIT_B_MIDS = [20.0, 40.0, 60.0]


def lam(b, k, b_mid):
    return 1.0 / (1.0 + np.exp(-k * (b - b_mid)))


def make_policy_fn(dual, mode):
    if mode == "level":
        return lambda obs, battery_level: dual.predict(obs, battery_level, deterministic=True)
    return lambda obs, battery_level: dual.predict(obs, BATTERY_MAX - battery_level, deterministic=True)


if __name__ == "__main__":
    with open(HP_PATH) as f:
        hp = json.load(f)
    k, b_mid = hp["gate_k"], hp["gate_b_mid"]
    print(f"Tuned gate_k={k}, gate_b_mid={b_mid}\n")

    print("Probability of routing to rho, by battery level (formula only):")
    print("  battery level :  " + "  ".join(f"{b:>5.0f}" for b in (10, 30, 50, 70, 90)))
    print("  'level' input :  " + "  ".join(f"{lam(b, k, b_mid):>5.2f}" for b in (10, 30, 50, 70, 90)))
    print("  'deficit' input: " + "  ".join(f"{lam(BATTERY_MAX - b, k, b_mid):>5.2f}" for b in (10, 30, 50, 70, 90)))
    print()

    variants = [("level", b_mid)] + [("deficit", bm) for bm in DEFICIT_B_MIDS]
    results = {v: [] for v in variants}

    for seed in SEEDS:
        print(f"--- seed {seed}: training g and rho ---")
        g = train_g(NET, G_BUDGET, seed, hp)
        rho = train_rho(seed, hp, net_size=NET, timesteps=RHO_BUDGET)
        for mode, bm in variants:
            dual = DualLoopPolicy(g, rho, k=k, b_mid=bm, rng=np.random.default_rng(seed))
            m = rollout(make_policy_fn(dual, mode), N_EPISODES, seed=6000, needs_battery_level=True)
            results[(mode, bm)].append(m)
            print(f"  {mode:<8} b_mid={bm:<5} success={m['success_rate']:.3f} crash={m['crash_rate']:.3f}")

    print("\n" + "=" * 60)
    print(f"SUMMARY (mean over seeds {SEEDS}, net={NET}, g budget={G_BUDGET:,}, rho budget={RHO_BUDGET:,})")
    print("=" * 60)
    for (mode, bm), ms in results.items():
        s = np.mean([m["success_rate"] for m in ms])
        c = np.mean([m["crash_rate"] for m in ms])
        print(f"  {mode:<8} b_mid={bm:<5} success={s:.3f} crash={c:.3f}")
    print("\nRead: if 'deficit' gives a much lower crash rate than 'level', B's weakness\n"
          "comes from the input direction (the Section 3.2 definition was not followed\n"
          "in the port), not from an intrinsic property of the gate.")
