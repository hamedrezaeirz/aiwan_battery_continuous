"""
plot_results_B_deficit_battery.py   (§7.7)
Regenerates Figure 12 (crash and success vs override weight, largest g) with all
versions of B, +-1 SEM over seeds, in the style of plot_results_battery.py:
  B               hardcoded gate as originally run (gate fed the battery level)
  B_deficit       gate direction corrected, b_mid derived from the geometry (90.99)
  B_deficit_mid60 gate direction corrected, second pre-specified b_mid = 60
Also writes a second figure: A - gate crash gap vs training budget (w=8) for the
corrected gates and for C, per network size, to show how the gap behaves.
Run from aiwan_battery_continuous/:
    python3 plot_results_B_deficit_battery.py
Reads results/results_battery.csv and results/results_battery_B_deficit.csv.
Writes results/battery_vs_w_crash_and_success_B_deficit.png and
results/battery_gap_vs_budget_B_deficit.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

old = pd.read_csv("results/results_battery.csv")
new = pd.read_csv("results/results_battery_B_deficit.csv")
d = pd.concat([old, new], ignore_index=True)
d["net"] = d["network_size"].astype(str).str.strip("[]").astype(int)
NETS, BUDGETS = [16, 256, 2048], [10_000, 100_000, 500_000]
W = (0, 1, 2, 4, 8)

def sem(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))

def series(arch, cond, net, bud, metric):
    s = d[(d.architecture == arch) & (d.condition == cond) & (d.net == net) & (d.training_timesteps == bud)]
    return s.set_index("seed")[metric].sort_index()

def fig12():
    net, bud = NETS[-1], BUDGETS[-1]
    styles = {"A": ("--", "#d64545", "A (single-loop)"),
              "B": ("-.", "#2f6fb0", "B as originally run (gate fed battery level)"),
              "B_deficit": (":", "#8e44ad", "B, direction corrected (b_mid=90.99)"),
              "B_deficit_mid60": (":", "#e67e22", "B, direction corrected (b_mid=60)"),
              "C": ("-", "#3ba55d", "C (learned gate)")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric in zip(axes, ["crash_rate", "success_rate"]):
        for arch, (ls, c, lab) in styles.items():
            xs, ms, es = [], [], []
            for w in W:
                v = series(arch, f"override_w{w}", net, bud, metric)
                xs.append(w); ms.append(v.mean()); es.append(sem(v))
            ax.errorbar(xs, ms, yerr=es, fmt=ls, marker="o", color=c, label=lab, capsize=2)
        ax.set_xlabel("override / conflict weight (w)"); ax.set_ylabel(metric)
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Battery, continuous env: net_size=[{net}], budget={bud:,} (mean +- SEM, 10 seeds)")
    fig.tight_layout()
    fig.savefig("results/battery_vs_w_crash_and_success_B_deficit.png", dpi=150)
    print("saved results/battery_vs_w_crash_and_success_B_deficit.png")

def gap_vs_budget():
    archs = [("B_deficit", "#8e44ad", "A - B (corrected, b_mid=90.99)"),
             ("B_deficit_mid60", "#e67e22", "A - B (corrected, b_mid=60)"),
             ("C", "#3ba55d", "A - C (learned gate)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, net in zip(axes, NETS):
        for arch, c, lab in archs:
            ms, es = [], []
            for b in BUDGETS:
                g = series("A", "override_w8", net, b, "crash_rate") - series(arch, "override_w8", net, b, "crash_rate")
                ms.append(g.mean()); es.append(sem(g))
            ax.errorbar(BUDGETS, ms, yerr=es, marker="o", color=c, label=lab, capsize=2)
        ax.set_xscale("log"); ax.axhline(0, color="gray", lw=1); ax.grid(alpha=0.3)
        ax.set_title(f"net_size=[{net}]"); ax.set_xlabel("training budget (timesteps)")
    axes[0].set_ylabel("crash-rate gap at w=8"); axes[0].legend(fontsize=7)
    fig.suptitle("Gap vs training budget (mean +- SEM, 10 seeds)")
    fig.tight_layout()
    fig.savefig("results/battery_gap_vs_budget_B_deficit.png", dpi=150)
    print("saved results/battery_gap_vs_budget_B_deficit.png")

if __name__ == "__main__":
    fig12(); gap_vs_budget()
