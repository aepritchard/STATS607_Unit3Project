.PHONY: all simulate analyze figures test clean

LABEL ?= optimized
WORKERS ?= 4
# Full label used for every output folder/file: plain for serial, _Nworkers for parallel
ifeq ($(WORKERS),1)
RUN_LABEL = $(LABEL)
else
RUN_LABEL = $(LABEL)_$(WORKERS)workers
endif

all: setup simulate analyze figures

# Set-up
setup: .setup_complete

.setup_complete: 
	@echo "Setting up environment"
	python3 -m pip install -r requirements.txt
	touch .setup_complete

# Run all scenarios and save raw results to results/raw/
simulate:
	PYTHONPATH=src python3 simulate.py $(LABEL) $(WORKERS)

# Compute performance measures from results/raw/runs.csv
analyze:
	PYTHONPATH=src python3 -c "from metrics import run_all_metrics; run_all_metrics()" $(LABEL) $(WORKERS)

# Create all visualizations
figures:
	PYTHONPATH=src python3 -c "from publication_figures import run_publication_figures; run_publication_figures()" $(LABEL) $(WORKERS)
	PYTHONPATH=src python3 -c "from diagnostic_figures import run_diagnostic_figures; run_diagnostic_figures()" $(LABEL) $(WORKERS)

# Run test suite
test:
	PYTHONPATH=src python3 -m pytest tests/

# Profile one experiment, time entire study
profile:
	PYTHONPATH=src python3 profile_study.py $(LABEL) $(WORKERS)

# Runtime vs sample size
complexity:
	PYTHONPATH=src python3 complexity_study.py $(LABEL) $(WORKERS)

# Numerical stability (warnings and errors for numerical/convergence problems)
stability:
	PYTHONPATH=src python3 stability_study.py $(LABEL) $(WORKERS)

# Remove generated files
# clean_baseline:
#	rm -rf results/raw/baseline results/tables/baseline results/figures/baseline
#	rm -f .*complete

clean_optimized:
	rm -rf results/raw/optimized results/tables/optimized results/figures/optimized
	rm -f .*complete