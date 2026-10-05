# aiwan_battery_continuous

Code, data, and figures for the **battery signal in a continuous environment** experiment (§7.7 of the AIWAN paper: *AIWAN: Artificial Intelligence With Artificial Need*).

## What this tests

Architectures A (single-loop reward shaping), B (hardcoded sigmoid gate) and C (learned gate) on a continuous 2-D navigation task with a depleting battery (drains 4 units/step, recharges 20 units within 0.05 of a fixed charger). Grid: network size {16, 256, 2048} × training budget {10k, 100k, 500k} steps × 10 seeds × 8 conditions (standard, goal_nulling, indifference, override w = 0, 1, 2, 4, 8). Algorithm: PPO (Stable-Baselines3). Total: 2,160 rows in `results/results_battery.csv` (A, B as originally run, C).

### Two versions of B (important)

The gate in AIWAN §3.2 and §5 is an *urgency* gate: ρ takes over when the battery level is **low**. The original sweep implemented the formula with the opposite sign, so it was re-run for B alone:

| Label in the data | Gate | Parameters | Where |
|---|---|---|---|
| `B` (as originally run) | `λ(b) = σ(k·(b − b_mid))` on the battery **level** — ρ is chosen when the battery is **high** | `k=0.2`, `b_mid=40` (`results/hyperparams_battery.json`) | `results_battery.csv` |
| `B_deficit` | `λ(b) = σ(k·(b_mid − b))` — ρ is chosen when the battery is **low** | `k=0.2`, `b_mid=90.99` = `4 (drain/step) × 20 (worst-case steps to the charger: far corner at full diagonal speed) + ln 9 / k` (the §5 derivation) | `results_battery_B_deficit.csv` |
| `B_deficit_mid60` | same direction | `k=0.2`, `b_mid=60`, a second, less conservative value | `results_battery_B_deficit.csv` |

Both corrected `b_mid` values were fixed **in advance**, not tuned on the results, and both are reported; neither should be picked after the fact. The corrected rows have g and ρ retrained on the same seeds and the full 3×3 grid, every condition, 10 seeds (1,440 rows = 9 cells × 10 seeds × 8 conditions × 2 gates). The original A, B and C rows were **not** re-run.

## Files

- `env_battery_continuous.py`, `arbitration_battery.py`, `train_battery.py`, `evaluate_battery.py`, `config_battery.py`: environment, architectures, training, evaluation, configuration.
- `tune_hyperparams_battery.py`: hyperparameter tuning on validation seeds 100–102 (disjoint from the sweep's seeds 0–9).
- `run_experiment_battery.py`, `launch_parallel_battery.sh`, `merge_results_battery.py`, `check_completeness_battery.py`: original sweep (A, B, C), parallel launch, merge, completeness check.
- `sanity_check_battery.py`, `diagnose_gate_rho_budget_mismatch.py`, `diagnose_gate_direction.py`: sanity and diagnostic scripts (see notes below).
- `plot_results_battery.py`: analysis of the original sweep; every statistic is per cell, nothing is pooled across cells or conditions.
- **`run_experiment_B_deficit_battery.py`**: B-only re-run with the corrected gate (`--quick`, `--pilot`, `--reduced`, default = full grid); evaluates both `b_mid` values on the same trained g.
- **`launch_B_deficit_battery.sh`**, **`merge_B_deficit_battery.py`**, **`analyze_B_deficit_battery.py`**, **`plot_results_B_deficit_battery.py`**: parallel launch, merge into `results_battery_B_deficit.csv`, comparison with A/B/C, and the figures below.
- `results/`: merged results, tuned hyperparameters, figures, and analysis outputs (`analysis_output.txt` for the original sweep, `analysis_B_deficit_battery.txt` for the re-run).

## Reproduce

```bash
pip install stable-baselines3 gymnasium torch pandas scipy matplotlib numpy
python3 tune_hyperparams_battery.py
bash launch_parallel_battery.sh          # two processes, seeds 0-4 and 5-9
python3 merge_results_battery.py
python3 check_completeness_battery.py results/results_battery.csv
python3 plot_results_battery.py results/results_battery.csv

# B re-run with the corrected gate direction
python3 run_experiment_B_deficit_battery.py --quick     # pipeline check
python3 run_experiment_B_deficit_battery.py --pilot     # one cell; prints a runtime projection
bash launch_B_deficit_battery.sh                        # full grid, 4 processes (about 13 hours on 4 cores)
python3 merge_B_deficit_battery.py && python3 analyze_B_deficit_battery.py
python3 plot_results_B_deficit_battery.py
```

## Results

At the highest override weight (w=8), largest cell (network size 2048, 500,000 steps), mean over 10 seeds:

| | crash | success |
|---|---|---|
| A (single-loop) | 0.116 | 0.416 |
| B as originally run (level-fed gate) | 0.320 | 0.168 |
| B_deficit (`b_mid=90.99`) | 0.002 | 0.100 |
| B_deficit_mid60 (`b_mid=60`) | 0.160 | 0.576 |
| C (learned gate) | 0.008 | 0.399 |

Mean over the 9 cells at w=8: crash A 0.355, B 0.359, B_deficit 0.001, B_deficit_mid60 0.190, C 0.014; success A 0.313, B 0.108, B_deficit 0.070, B_deficit_mid60 0.376, C 0.443. Under goal-nulling (untrained networks), crash: A 0.995, B 0.375, B_deficit 0.002, B_deficit_mid60 0.436, C 0.020.

## Notes on interpretation

- At the highest override weight, the A−C crash-rate gap falls with training budget because the single-loop agent improves; C's crash rate stays low throughout (C is safe at the smallest budget by being inert). The network-size axis does not show a significant narrowing.
- **The same narrowing appears for the corrected hardcoded gate**, which cannot be eroded by a more capable g: the A−B_deficit gap falls from 0.971, 0.904 and 0.927 to 0.000, 0.060 and 0.114 at 16, 256 and 2048 units (paired t(9)=138.9, 24.8, 10.8), and the A−B_deficit_mid60 gap from 0.73, 0.67 and 0.68 to −0.15, −0.07 and −0.04. So the narrowing in this environment reflects the baseline improving, not fold-back (`battery_gap_vs_budget_B_deficit.png`).
- **Always read `crash_rate` together with `success_rate`**: an inert policy never crashes. `B_deficit` (`b_mid=90.99`) almost never crashes (≤0.003 in every cell and condition) but rarely reaches the goal (success 0.04–0.12); `B_deficit_mid60` trades safety for success (crash 0.14–0.25 and success up to 0.64 at w=8). `b_mid` moves the corrected gate along this trade-off; C, at low override weights, is both safe and successful (success 1.00).
- The gate direction was the main cause of B's crashes in this environment (as originally run, B crashes 0.31–0.42 at w=8). `diagnose_gate_direction.py` is the earlier single-cell diagnostic (3 validation seeds); its output was not archived and the full re-run above supersedes it.
- Engineered indifference does not reproduce the single-loop collapse seen in the tabular task: here the goal is reachable before the battery depletes.
- Training is not bit-for-bit deterministic (standard and override_w0 are the same configuration but differ per seed).

Figures in `results/`:

- `battery_gap_AC_vs_power.png`, `battery_vs_w_crash_and_success.png` — original (B as originally run)
- `battery_vs_w_crash_and_success_B_deficit.png` — crash and success vs w in the largest cell, all versions of B (Figure 12 of the paper)
- `battery_gap_vs_budget_B_deficit.png` — A − gate crash gap vs budget at w=8, per network size
