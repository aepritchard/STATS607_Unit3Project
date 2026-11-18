# STATS 607 Unit Project 

## Project Description:
The unit 2 project reproduced the analysis of the paper "Randomization Does Not Justify Logistic Regression" (Freedman 2008). The simulation examines the use of the logit model under a non-logit data generating mechanism to examine the performance of estimators of the differential log odds of success. Experiments are used to compare the results under different sample sizes, number of treated, data generating distributions, outcome definitions, and under a probit model.

The unit 3 project builds on the unit 2 simulation study by optimizing it. This includes adding vectorization and parallization to improve runtime.

## Setup Instructions
Python 3.13+ must be installed on the user's system. All required python packages are stored in `requirements.txt` and are installed using the command `make setup`. This setup process also runs when the `make all` command is used.

## How to Run the Analysis
In the main project directory, the following commands are defined in the Makefile:
* `make all`: installs required python packages, runs simulation and analysis, generates all figures for the serial optimized version
* `make setup`: installs required python packages
* `make simulate`: runs simulations and saves raw results
* `make analyze`: processes raw results and generates summary statistics
* `make figures`: generates all figures
* `make profile`: records the entire simulation runtime of the serial optimized version, profiles the runtime of experiment 1 with n=5000 for the serial optimized version
* `make complexity`: records the simulation run time by sample size using the serial optimized version
* `make stability-check`: summarizes all errors recieved when running the simulation study with the serial optimized version
* `make parallel`: runs the simulation, analysis, figures, profile, complexity, and stability-check procedures using the parallelized version
* `make benchmark`: creates log-log plots of runtime versus sample size using all versions, summarizes run time for all versions
* `make regression-test`: runs all tests to verify that results are consistent between the baseline (unit 2) version and the optimized verions
* `make clean_optimized`: removes all generated files for optimized runs

The number of cores used for parallelization are defined as `PARALLEL_CORES` in the Makefile. Changing this number will change the number of cores used when running `make parallel`.

## Summary of Key Findings
Across all scenarios and variations, the coefficient estimator $\hat{\beta}_2$ is biased for $\Delta$ while the plug-in and ITT estimators $\tilde{\Delta}$ and $\hat{\Delta}$ are consistently unbiased for $\Delta$. As described in the paper, using a probit model resulted in coefficients that were roughly 5/8 of that of the logit model, especially for larger sample sizes.