'''
Runs the simulation study and saves raw results to results/raw/
'''
import logging

from scenarios import SCENARIOS
from simulation import SimulationStudy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import sys

# Full label and number of workers come from the Makefile
if len(sys.argv) > 1:
    label = sys.argv[1]
else:
    label = "baseline"

if len(sys.argv) > 2:
    N_WORKERS = int(sys.argv[2])
else:
    N_WORKERS = 1


if __name__ == "__main__":
    study = SimulationStudy(SCENARIOS, label=label, n_workers=N_WORKERS)
    study.run()