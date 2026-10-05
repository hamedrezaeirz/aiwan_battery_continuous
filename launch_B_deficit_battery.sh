#!/bin/bash
# Usage (from aiwan_battery_continuous/):  bash launch_B_deficit_battery.sh [--reduced]
# 4 processes over seeds 0-9. Start it only when cores are free (the §7.3 rerun uses all 4).
# Then: python3 merge_B_deficit_battery.py && python3 analyze_B_deficit_battery.py
set -e
cd "$(dirname "$0")"; mkdir -p results logs
MODE="$1"
python3 run_experiment_B_deficit_battery.py $MODE --seeds 0 1 2 > logs/Bdefbat_0_1_2.log 2>&1 &
python3 run_experiment_B_deficit_battery.py $MODE --seeds 3 4 5 > logs/Bdefbat_3_4_5.log 2>&1 &
python3 run_experiment_B_deficit_battery.py $MODE --seeds 6 7   > logs/Bdefbat_6_7.log   2>&1 &
python3 run_experiment_B_deficit_battery.py $MODE --seeds 8 9   > logs/Bdefbat_8_9.log   2>&1 &
wait
echo "done -- now: python3 merge_B_deficit_battery.py && python3 analyze_B_deficit_battery.py"
