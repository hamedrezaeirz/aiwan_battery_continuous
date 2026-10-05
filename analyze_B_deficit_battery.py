"""Compare B_deficit(+_mid60) with A, B (old, level-fed gate) and C for §7.7.
Run from aiwan_battery_continuous/ after merge_B_deficit_battery.py."""
import pandas as pd
from scipy import stats

old = pd.read_csv("results/results_battery.csv")
new = pd.read_csv("results/results_battery_B_deficit.csv")
df = pd.concat([old, new], ignore_index=True)
keys = new[["network_size", "training_timesteps", "condition"]].drop_duplicates()
df = df.merge(keys, on=["network_size", "training_timesteps", "condition"])
print("cells:", new.groupby(["network_size", "training_timesteps"]).ngroups, "| seeds:", sorted(new.seed.unique()))

for m in ("crash_rate", "success_rate"):
    print(f"\n== mean {m} by condition and architecture ==")
    print(df.pivot_table(index="condition", columns="architecture", values=m).round(3))

print("\n== override_w8 per cell: A - gate crash gap and gate crash range ==")
w8 = df[df.condition == "override_w8"].pivot_table(
    index=["network_size", "training_timesteps"], columns="architecture", values="crash_rate")
for col in [c for c in ("B", "B_deficit", "B_deficit_mid60", "C") if c in w8]:
    g = w8["A"] - w8[col]
    print(f"{col:16s} crash min {w8[col].min():.3f} max {w8[col].max():.3f} | A-gate gap min {g.min():.3f} max {g.max():.3f}")

for arch in ("B_deficit", "B_deficit_mid60"):
    x = df[(df.condition == "override_w8") & df.architecture.isin(["A", arch])]
    p = x.pivot_table(index=["seed", "training_timesteps"], columns="architecture", values="crash_rate")
    gap = (p["A"] - p[arch]).groupby(["seed", "training_timesteps"]).mean().unstack()
    lo, hi = gap.columns.min(), gap.columns.max()
    t = stats.ttest_rel(gap[lo], gap[hi])
    print(f"\n{arch}: A-gate gap narrows with budget? gap@{lo}={gap[lo].mean():.3f} gap@{hi}={gap[hi].mean():.3f} "
          f"paired t({len(gap)-1})={t.statistic:.2f}, p={t.pvalue:.3g}")
