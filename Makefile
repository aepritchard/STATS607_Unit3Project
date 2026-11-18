.PHONY: all setup simulate parallel analyze figures test profile complexity stability-check clean_optimized regression_test benchmark

RUN_LABEL ?= optimized
CORES ?= 1
PARALLEL_CORES ?= 2

# Full label used for every output folder/file: plain for serial, _Nworkers for parallel
ifeq ($(PARALLEL_CORES),1)
PARALLEL_RUN_LABEL = $(RUN_LABEL)
else
PARALLEL_RUN_LABEL = $(RUN_LABEL)_$(PARALLEL_CORES)cores
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
	PYTHONPATH=src python3 simulate.py $(RUN_LABEL) $(CORES)

# Run all scenarios in parallel, run profiling and complexity
parallel:
	$(MAKE) simulate RUN_LABEL=$(PARALLEL_RUN_LABEL) CORES=$(PARALLEL_CORES)
	$(MAKE) analyze RUN_LABEL=$(PARALLEL_RUN_LABEL) CORES=$(PARALLEL_CORES)
	$(MAKE) profile RUN_LABEL=$(PARALLEL_RUN_LABEL) CORES=$(PARALLEL_CORES)
	$(MAKE) complexity RUN_LABEL=$(PARALLEL_RUN_LABEL) CORES=$(PARALLEL_CORES)
	$(MAKE) stability-check RUN_LABEL=$(PARALLEL_RUN_LABEL) CORES=$(PARALLEL_CORES)

# Compute performance measures from results/raw/<RUN_LABEL>/runs.csv
analyze:
	PYTHONPATH=src python3 -c "from metrics import run_all_metrics; run_all_metrics(label='$(RUN_LABEL)')"

# Create all visualizations
figures:
	PYTHONPATH=src python3 -c "from publication_figures import run_publication_figures; run_publication_figures(label='$(RUN_LABEL)')"
	PYTHONPATH=src python3 -c "from diagnostic_figures import run_diagnostic_figures; run_diagnostic_figures(label='$(RUN_LABEL)')"

# Profile one experiment, time entire study
profile:
	PYTHONPATH=src python3 profile_study.py $(RUN_LABEL) $(CORES)

# Runtime vs sample size
complexity:
	PYTHONPATH=src python3 complexity_study.py $(RUN_LABEL) $(CORES)

# Numerical stability (failure reasons across conditions)
stability-check:
	PYTHONPATH=src python3 stability_study.py $(RUN_LABEL) $(CORES)

# Compare runtimes across all saved versions using saved results
benchmark:
	PYTHONPATH=src python3 benchmark_study.py

# Regression test (compare baseline and optimized output)
regression-test:
	PYTHONPATH=src python3 -m pytest test_regression.py -rs

# Remove generated files
clean_optimized:
	rm -rf results/raw/optimized* results/tables/optimized* results/figures/optimized*
	rm -rf results/profiling/optimized* results/complexity/optimized*
	rm -rf results/warnings/optimized*
	rm -f .*complete