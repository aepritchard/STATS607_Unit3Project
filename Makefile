.PHONY: all setup simulate parallel analyze figures test profile complexity stability-check clean_optimized regression_test benchmark

LABEL ?= optimized
WORKERS ?= 1
PARALLEL_WORKERS ?= 4

# Full label used for every output folder/file: plain for serial, _Nworkers for parallel
ifeq ($(WORKERS),1)
RUN_LABEL = $(LABEL)
else
RUN_LABEL = $(LABEL)_$(WORKERS)workers
endif

all: setup simulate analyze figures profile complexity stability-check

# Set-up
setup: .setup_complete

.setup_complete:
	@echo "Setting up environment"
	python3 -m pip install -r requirements.txt
	touch .setup_complete

# Run all scenarios and save raw results to results/raw/<RUN_LABEL>/
simulate:
	PYTHONPATH=src python3 simulate.py $(RUN_LABEL) $(WORKERS)

# Run all scenarios in parallel
parallel:
	$(MAKE) simulate WORKERS=$(PARALLEL_WORKERS)

# Compute performance measures from results/raw/<RUN_LABEL>/runs.csv
analyze:
	PYTHONPATH=src python3 -c "from metrics import run_all_metrics; run_all_metrics(label='$(RUN_LABEL)')"

# Create all visualizations
figures:
	PYTHONPATH=src python3 -c "from publication_figures import run_publication_figures; run_publication_figures(label='$(RUN_LABEL)')"
	PYTHONPATH=src python3 -c "from diagnostic_figures import run_diagnostic_figures; run_diagnostic_figures(label='$(RUN_LABEL)')"

# Profile one experiment, time entire study
profile:
	PYTHONPATH=src python3 profile_study.py $(RUN_LABEL) $(WORKERS)

# Runtime vs sample size
complexity:
	PYTHONPATH=src python3 complexity_study.py $(RUN_LABEL) $(WORKERS)

# Numerical stability (failure reasons across conditions)
stability-check:
	PYTHONPATH=src python3 stability_study.py $(RUN_LABEL)

# Compare runtimes across all saved versions using saved results
benchmark:
	PYTHONPATH=src python3 benchmark_study.py

# Regression test (compare baseline and optimized output)
regression_test:
	PYTHONPATH=src python3 -m pytest test_regression.py -rs

# Remove generated files
clean_optimized:
	rm -rf results/raw/optimized* results/tables/optimized* results/figures/optimized*
	rm -f .*complete