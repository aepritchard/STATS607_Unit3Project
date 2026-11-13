'''
Runs the simulation study and saves raw results to results/raw/
'''
import logging

from scenarios import SCENARIOS
from simulation import SimulationStudy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import sys

# Label comes from the Makefile (make <target> LABEL=...); defaults to baseline
if len(sys.argv) > 1:
    LABEL = sys.argv[1]
else:
    LABEL = "baseline"

study = SimulationStudy(SCENARIOS, label=LABEL)
study.run()