"""
plot_results_battery.py  (corrected)
------------------------------------
Analysis of the battery-on-continuous-environment sweep (A/B/C).

Fixes vs. the first version:
  * NO statistic is pooled across cells or conditions. Every number is per
    (network_size, budget) cell, and every paired test holds the other
    axis fixed (the pooling bug recorded in HANDOFF v10).
  * crash_rate is always shown NEXT TO success_rate (a policy that never
    moves also never crashes).
  * B's conflict-insensitivity is reported as a range across w, not as
    "p>0.05 therefore equal".
  * Figures carry +-1 SEM error bars.

Usage:
    python3 plot_results_battery.py results/results_battery.csv
"""

import sys
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY = ["architecture", "network_size", "training_timesteps", "seed", "condition"]
NETS = [16, 256, 2048]
BUDGETS = [10_000, 100_000, 500_000]
W = 8


def load(path):
    d = pd.read_csv(path)
    dup = d.groupby(KEY).size()
    if (dup > 1).any():
        raise SystemExit("Duplicate keys -- run check_completeness_battery.py first.")
    d["net"] = d["network_size"].astype(str).str.strip("[]").astype(int)
    n = d["seed"].nunique()
    print(f"Loaded {len(d)} rows, {n} seeds.")
    return d


def cell_table(d, cond):
    s = d[d.condition == cond]
    g = s.groupby(["net", "training_timesteps", "architecture"])[["crash_rate", "success_rate"]].mean()
    t = g.unstack("architecture").round(3)
    print(f"\n=== per-cell means, condition = {cond} (crash_rate | success_rate; A/B/C) ===")
    print(t.to_string())


def seed_series(d, cond, net, bud, arch, metric):
    s = d[(d.condition == cond) & (d.net == net) & (d.training_timesteps == bud) & (d.architecture == arch)]
    return s.set_index("seed")[metric].sort_index()


def gap(d, net, bud, metric="crash_rate", x="A", y="C", cond=f"override_w{W}"):
    return seed_series(d, cond, net, bud, x, metric) - seed_series(d, cond, net, bud, y, metric)


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def fmt_test(a, b=None):
    if b is None:
        if np.std(a, ddof=1) == 0:
            return "t undefined (zero variance)"
        t, p = stats.ttest_1samp(a, 0)
    else:
        if np.allclose(a.values, b.values):
            return "t undefined (identical)"
        t, p = stats.ttest_rel(a, b)
    return f"t({len(a)-1})={t:.2f}, p={p:.3g}"


def gap_tests(d):
    print(f"\n=== A-C crash gap at override_w{W}, per cell (mean +- SEM; vs 0) ===")
    for net in NETS:
        for bud in BUDGETS:
            g = gap(d, net, bud)
            print(f"  net={net:<5} budget={bud:<7} gap={g.mean():+.3f} +- {sem(g):.3f}   {fmt_test(g)}")

    print("\n=== Budget axis: gap at 10k vs 500k, paired, network held fixed ===")
    for net in NETS:
        g1, g2 = gap(d, net, BUDGETS[0]), gap(d, net, BUDGETS[-1])
        print(f"  net={net:<5} {g1.mean():.3f} -> {g2.mean():.3f}   {fmt_test(g1, g2)}")

    print("\n=== Network axis: gap at net=16 vs 2048, paired, budget held fixed ===")
    for bud in BUDGETS:
        g1, g2 = gap(d, NETS[0], bud), gap(d, NETS[-1], bud)
        print(f"  budget={bud:<7} {g1.mean():.3f} -> {g2.mean():.3f}   {fmt_test(g1, g2)}")

    print(f"\n=== C vs A success_rate at override_w{W} (C-A, paired) ===")
    for net in NETS:
        for bud in BUDGETS[1:]:
            ds = gap(d, net, bud, metric="success_rate", x="C", y="A")
            print(f"  net={net:<5} budget={bud:<7} C-A={ds.mean():+.3f} +- {sem(ds):.3f}   {fmt_test(ds)}")


def b_sensitivity(d):
    s = d[d.architecture == "B"]
    g = s[s.condition.str.startswith("override_w")].copy()
    g["w"] = g.condition.str.replace("override_w", "").astype(int)
    per_w = g.groupby("w")["crash_rate"].mean()
    print("\n=== Architecture B: mean crash_rate by w (each mean is over 9 cells x 10 seeds) ===")
    print(per_w.round(3).to_string())
    print(f"  range across w: {per_w.max() - per_w.min():.3f}")
    cells = s.groupby(["condition", "net", "training_timesteps"])["crash_rate"].mean()
    print(f"  B crash_rate, min/max over ALL cells and conditions: {cells.min():.3f} / {cells.max():.3f}")


def determinism_check(d):
    k = ["architecture", "net", "training_timesteps", "seed"]
    a = d[d.condition == "standard"].set_index(k).sort_index()
    b = d[d.condition == "override_w0"].set_index(k).sort_index()
    diff = (a["crash_rate"] - b["crash_rate"]).abs()
    print(f"\n=== standard vs override_w0 (same config): max |crash diff| per seed = {diff.max():.3f}, "
          f"mean = {diff.mean():.4f} ===")


def figures(d):
    # Figure B1: crash and success vs w, largest cell
    net, bud = NETS[-1], BUDGETS[-1]
    styles = {"A": ("--", "#d64545", "A (single-loop)"),
              "B": ("-.", "#2f6fb0", "B (hardcoded sigmoid gate)"),
              "C": ("-", "#3ba55d", "C (learned gate)")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric in zip(axes, ["crash_rate", "success_rate"]):
        for arch, (ls, c, lab) in styles.items():
            xs, ms, es = [], [], []
            for w in (0, 1, 2, 4, 8):
                v = seed_series(d, f"override_w{w}", net, bud, arch, metric)
                xs.append(w); ms.append(v.mean()); es.append(sem(v))
            ax.errorbar(xs, ms, yerr=es, fmt=ls, marker="o", color=c, label=lab, capsize=2)
        ax.set_xlabel("override / conflict weight (w)")
        ax.set_ylabel(metric)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Battery, continuous env: net_size=[{net}], budget={bud:,} (mean +- SEM, 10 seeds)")
    fig.tight_layout()
    fig.savefig("results/battery_vs_w_crash_and_success.png", dpi=150)
    plt.close(fig)
    print("Saved results/battery_vs_w_crash_and_success.png")

    # Figure B2: gap vs budget and vs network, SEM, w8
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for n in NETS:
        ms = [gap(d, n, b).mean() for b in BUDGETS]
        es = [sem(gap(d, n, b)) for b in BUDGETS]
        axes[0].errorbar(BUDGETS, ms, yerr=es, marker="o", capsize=2, label=f"net=[{n}]")
    axes[0].set_xscale("log"); axes[0].set_xlabel("training budget"); axes[0].set_ylabel("crash-rate gap (A-C)")
    axes[0].set_title(f"Gap vs budget (override_w{W})")
    for b in BUDGETS:
        ms = [gap(d, n, b).mean() for n in NETS]
        es = [sem(gap(d, n, b)) for n in NETS]
        axes[1].errorbar(NETS, ms, yerr=es, marker="o", capsize=2, label=f"budget={b:,}")
    axes[1].set_xscale("log"); axes[1].set_xlabel("network size"); axes[1].set_ylabel("crash-rate gap (A-C)")
    axes[1].set_title(f"Gap vs network size (override_w{W})")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/battery_gap_AC_vs_power.png", dpi=150)
    plt.close(fig)
    print("Saved results/battery_gap_AC_vs_power.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/results_battery.csv"
    d = load(path)
    os.makedirs("results", exist_ok=True)
    for cond in ("standard", f"override_w{W}", "goal_nulling", "indifference"):
        cell_table(d, cond)
    gap_tests(d)
    b_sensitivity(d)
    determinism_check(d)
    figures(d)
    print("\nDone. Report crash_rate together with success_rate; do not quote any pooled mean.")
