'''
Times the entire simulation study
Profiles one experiment in depth (experiment 1 w/ n=5000)
'''
import logging
import tempfile
from pathlib import Path
from pyinstrument import Profiler
from time import perf_counter

from scenarios import SCENARIOS
from simulation import SimulationStudy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

LABEL = "baseline"
PROFILE_DIR = Path("results/profiling")

# Profile experiment 1 w/ n 5000
def profile_experiment_1():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_ID = "e1_n5000"
    scenarios_to_profile = {SCENARIO_ID: SCENARIOS[SCENARIO_ID]}

    # Use scratch directory to have blank start and throw away results
    with tempfile.TemporaryDirectory() as scratch_dir:
        study = SimulationStudy(scenarios_to_profile, out_dir = scratch_dir)

        with Profiler() as profiler:
            study.run()

    file_stem = f"{LABEL}_{SCENARIO_ID}_profile"

    # Create interactive flame graph
    html_path = PROFILE_DIR / f"{file_stem}.html"
    profiler.write_html(html_path)

    # Output profiler call tree text
    text_path = PROFILE_DIR / f"{file_stem}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(profiler.output_text(unicode=True, color=False))

    # Save profiling session information
    session_path = PROFILE_DIR / f"{file_stem}.pyisession"
    profiler.last_session.save(session_path)

    logging.info("Profiled %s (%s) in %.1f seconds",
                 SCENARIO_ID, LABEL, profiler.last_session.duration)
    logging.info("Profile written to %s", PROFILE_DIR)

# Time the entire simulation study
def time_simulation_study():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Use scratch directory to have blank start and throw away results
    with tempfile.TemporaryDirectory() as scratch_dir:
        study = SimulationStudy(SCENARIOS, out_dir=scratch_dir)

        start = perf_counter()
        study.run()
        total_seconds = perf_counter() - start

    # Save total runtime
    runtime_path = PROFILE_DIR / f"{LABEL}_total_runtime.txt"
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"label: {LABEL}\n")
        f.write(f"n_scenarios: {len(SCENARIOS)}\n")
        f.write(f"n_sim: {study.n_sim}\n")
        f.write(f"total_seconds: {total_seconds:.1f}\n")
        f.write(f"total_minutes: {total_seconds / 60:.2f}\n")

    logging.info("Entire study (%s) took %.1f seconds", LABEL, total_seconds)
    logging.info("Total runtime written to %s", runtime_path)

if __name__ == "__main__":
    profile_experiment_1()
    time_simulation_study()