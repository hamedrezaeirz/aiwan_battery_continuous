#!/bin/bash
# launch_parallel_battery.sh
# ----------------------------
# Splits config_battery.SEEDS = [0..9] across 2 processes (5 seeds
# each), matching this project's confirmed 2-of-4-cores decision (same
# as launch_parallel_oversight.sh). Each process writes its own
# results/results_battery_seeds_<...>.csv -- merge afterwards with
# merge_results_battery.py.
#
# Run this AFTER the oversight sweep has finished (per the user's
# explicit instruction this round) -- both experiments together would
# otherwise compete for the same 2 cores, slowing both down
# unpredictably.
#
#   tmux new -s aiwan_battery
#   bash launch_parallel_battery.sh
#   [Ctrl+b, d to detach]
#
# or (preferred on this VPS, given tmux's unreliable behavior observed
# earlier in this project -- see HANDOFF_STATUS's tmux/nohup notes):
#
#   nohup bash launch_parallel_battery.sh > launch_battery_main.log 2>&1 &
#   disown
#
# Check progress any time with:
#   tail -f logs/battery_seeds_0-4.log
#   tail -f logs/battery_seeds_5-9.log
#   wc -l results/results_battery_seeds_*.csv
#
# IMPORTANT: after both processes finish, run merge_results_battery.py
# then check_completeness_battery.py -- never trust a raw row count
# alone (this project's own repeated lesson).

set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p results logs

echo "Starting 2 parallel processes across seeds 0-9 (5 seeds each)..."

python3 -u run_experiment_battery.py --seeds 0 1 2 3 4 \
    > logs/battery_seeds_0-4.log 2>&1 &
PID1=$!
python3 -u run_experiment_battery.py --seeds 5 6 7 8 9 \
    > logs/battery_seeds_5-9.log 2>&1 &
PID2=$!

echo "Launched PIDs: $PID1 $PID2"
echo "Waiting for both to finish..."

wait $PID1 $PID2

echo "Both processes finished. Run: python3 merge_results_battery.py"
echo "Then verify completeness: python3 check_completeness_battery.py results/results_battery.csv"
