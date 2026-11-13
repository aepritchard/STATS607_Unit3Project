'''
Analyzes run time vs n for all experiments
'''

import logging
import tempfile
from pathlib import Path
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt

from simulation import SimulationStudy
from scenarios import make_scenarios_for_all_n_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

LABEL = "baseline"
COMPLEXITY_DIR = Path("results/complexity")

### Make scenarios for complexity analysis
# Initial unit 2 study did not take long to run, add larger sample sizes
#   to test how run time changes
n_samples = [100, 500, 1000, 5000, 10000, 25000, 50000]
SCENARIOS = {}
make_scenarios_for_all_n_samples(n_samples, SCENARIOS)

# 
def time_all_scenarios():
    COMPLEXITY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_id, settings in SCENARIOS.items():
        # Use scratch directory to have blank start and throw away results
        with tempfile.TemporaryDirectory() as scratch_dir:
            study = SimulationStudy({scenario_id: settings}, out_dir=scratch_dir)

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
    runtimes_path = COMPLEXITY_DIR / f"{LABEL}_runtimes.csv"
    runtimes.to_csv(runtimes_path, index=False)
    logging.info("Runtimes written to %s", runtimes_path)

if __name__ == "__main__":
    time_all_scenarios()