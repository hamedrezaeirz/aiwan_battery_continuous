"""
merge_results_battery.py
---------------------------
Merges every results/results_battery_seeds_*.csv produced by a
parallel `--seeds` run (see launch_parallel_battery.sh) into the
canonical results/results_battery.csv. Mirrors merge_results_oversight.py.

Usage (after both launch_parallel_battery.sh processes have finished):
    python3 merge_results_battery.py

IMPORTANT: does NOT deduplicate or verify completeness -- always run
check_completeness_battery.py on the merged file afterward.
"""

import glob
import pandas as pd

from config_battery import RESULTS_DIR, RESULTS_CSV_PATH

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RESULTS_DIR}/results_battery_seeds_*.csv"))
    if not files:
        raise SystemExit(
            f"No results_battery_seeds_*.csv files found in {RESULTS_DIR}/ -- "
            f"nothing to merge yet."
        )
    print(f"Merging {len(files)} files:")
    for f in files:
        print(f"  {f}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nWrote {len(df)} total rows to {RESULTS_CSV_PATH}")
    print(f"\nNow run: python3 check_completeness_battery.py {RESULTS_CSV_PATH}")
