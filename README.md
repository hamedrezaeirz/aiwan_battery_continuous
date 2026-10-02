# aiwan_battery_continuous

Code, data, and figures for the **battery signal in a continuous environment** experiment
(§7.7 of the AIWAN paper: *AIWAN: Artificial Intelligence With Artificial Need*).

## What this tests
Architectures **A** (single-loop reward shaping), **B** (hardcoded sigmoid gate), and **C** (learned gate) on a
continuous 2-D navigation task with a depleting battery (drains 4 units/step, recharges 20 units within 0.05 of a fixed
charger). Grid: network size {16, 256, 2048} x training budget {10k, 100k, 500k} steps x 10 seeds x 8 conditions
(standard, goal_nulling, indifference, override w = 0, 1, 2, 4, 8). Algorithm: PPO (Stable-Baselines3).
Total: 2,160 rows in `results/results_battery.csv`.

## Files
- `env_battery_continuous.py`, `arbitration_battery.py`, `train_battery.py`, `evaluate_battery.py`, `config_battery.py`: environment, architectures, training, evaluation, configuration.
- `tune_hyperparams_battery.py`: hyperparameter tuning on validation seeds 100-102 (disjoint from the sweep's seeds 0-9).
- `run_experiment_battery.py`, `launch_parallel_battery.sh`, `merge_results_battery.py`, `check_completeness_battery.py`: sweep, parallel launch, merge, completeness check.
- `sanity_check_battery.py`, `diagnose_gate_rho_budget_mismatch.py`, `diagnose_gate_direction.py`: sanity and diagnostic scripts (see notes below).
- `plot_results_battery.py`: analysis; every statistic is per cell, nothing is pooled across cells or conditions.
- `results/`: merged results, tuned hyperparameters, figures, and the analysis output (`analysis_output.txt`).

## Reproduce
```
pip install stable-baselines3 gymnasium torch pandas scipy matplotlib numpy
python3 tune_hyperparams_battery.py
bash launch_parallel_battery.sh          # two processes, seeds 0-4 and 5-9
python3 merge_results_battery.py
python3 check_completeness_battery.py results/results_battery.csv
python3 plot_results_battery.py results/results_battery.csv
```

## Notes on interpretation
- At the highest override weight, the A-C crash-rate gap falls with training budget because the single-loop agent improves; C's crash rate stays low throughout (C is safe at the smallest budget by being inert). The network-size axis does not show a significant narrowing.
- Always read `crash_rate` together with `success_rate`: an inert policy never crashes.
- **Architecture B** uses the gate lambda(b) = sigmoid(k(b - b_mid)) evaluated on the battery *level*, as in the original implementation. `diagnose_gate_direction.py` shows (3 validation seeds, one cell) that feeding the battery *deficit* instead lowers B's crash rate substantially. The sweep itself was not rerun with the reversed input.
- Engineered indifference does not reproduce the single-loop collapse seen in the tabular task: here the goal is reachable before the battery depletes.
- Training is not bit-for-bit deterministic (`standard` and `override_w0` are the same configuration but differ per seed).
