'''
Analyzes run time vs n for all experiments
'''

import logging
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import SimulationStudy
from scenarios import make_scenarios_for_all_n_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import sys

# Full label and number of cores come from the Makefile
if len(sys.argv) > 1:
    label = sys.argv[1]
else:
    label = "baseline"

if len(sys.argv) > 2:
    N_CORES = int(sys.argv[2])
else:
    N_CORES = 1

COMPLEXITY_DIR = Path("results/complexity")

### Make scenarios for complexity analysis
# Initial unit 2 study did not take long to run, add larger sample sizes
#   to test how run time changes
n_samples = [100, 500, 1000, 5000, 10000, 25000, 50000]
SCENARIOS = {}
make_scenarios_for_all_n_samples(n_samples, SCENARIOS)

# One panel per experiment and proportion treated
CONDITIONS = [
    {"label": "E1: baseline (Table 1)", "experiment": "e1_table1", "pi_treat": 0.75},
    {"label": "E2: 75% treated", "experiment": "e2_pct_treated", "pi_treat": 0.75},
    {"label": "E2: 50% treated", "experiment": "e2_pct_treated", "pi_treat": 0.50},
    {"label": "E2: 25% treated", "experiment": "e2_pct_treated", "pi_treat": 0.25},
    {"label": "E3: cut offs 0.95, 1.5", "experiment": "e3_cutoffs", "pi_treat": 0.75},
    {"label": "E4: Z = U + V", "experiment": "e4_covariate", "pi_treat": 0.75},
    {"label": "E5: (U, V) ~ Beta(2, 5)", "experiment": "e5_joint_dist", "pi_treat": 0.75},
]

#
def time_all_scenarios():
    COMPLEXITY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_id, settings in SCENARIOS.items():
        # Use scratch directory to have blank start and throw away results
        with tempfile.TemporaryDirectory() as scratch_dir:
            study = SimulationStudy({scenario_id: settings}, label=label, n_cores=N_CORES, out_dir=scratch_dir)

            start = perf_counter()
            study.run()
            seconds = perf_counter() - start

        logging.info("Timed %s: %.1f seconds", scenario_id, seconds)
        rows.append({
            "scenario_id": scenario_id,
            "experiment": settings["experiment"],
            "pi_treat": settings["pi_treat"],
            "n_sample": settings["n_sample"],
            "seconds": seconds,
        })

    # Save runtimes
    runtimes = pd.DataFrame(rows)
    runtimes_path = COMPLEXITY_DIR / f"{label}_runtimes.csv"
    runtimes.to_csv(runtimes_path, index=False)
    logging.info("Runtimes written to %s", runtimes_path)

    return runtimes

if __name__ == "__main__":
    time_all_scenarios()