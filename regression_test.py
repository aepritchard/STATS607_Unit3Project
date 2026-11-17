import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import pytest

# Baseline and optimized use the same random streams to sample from the population
#       They only differ in treatment assignment (not the same random streams)
#       So, results should be similar, but not identical

# Import data
versions = ["baseline", "optimized", "optimized_2workers", "optimized_4workers"]
all_version_tables = {}
for version in versions:
    # Import all tables
    version_dir = Path("results/tables") / version

    performance = pd.read_csv(version_dir / "performance.csv")
    probit_logit_paired = pd.read_csv(version_dir / "probit_logit_paired.csv")
    probit_logit_ratio = pd.read_csv(version_dir / "probit_logit_ratio.csv")
    table1 = pd.read_csv(version_dir / "table1_reproduction.csv")

    # Save
    version_tables = {'performance': performance,
                      'probit_logit_paired': probit_logit_paired,
                      'probit_logit_ratio': probit_logit_ratio,
                      'table1': table1}
    all_version_tables[version] = version_tables

################################################################## 
#  Check table output from baseline to new optimized versions
################################################################## 

TABLE1_ESTIMATORS = ['beta1_hat', 'beta2_hat', 'beta3_hat', 'plug_in']

def get_n_valid(version, model="logit", experiment = None, scenario_id_contains = None):
    '''
    Returns number of valid repetitions per n_sample
    If experiment is specified, only returns the specified experiment
    If scenario_id_contains is specified, only returns scenario_ids that contain the provided string
    '''
    performance = all_version_tables[version]['performance']

    # Only rows for the requested model (n_valid can differ between logit and probit)
    rows = performance[performance['model'] == model]

    if experiment is not None:
        rows = rows[rows['experiment'] == experiment]

    if scenario_id_contains is not None:
        rows = rows[rows['scenario_id'].str.contains(scenario_id_contains, na=False)]

    # n_valid is the same for every estimator within a scenario and model, so max just picks it
    n_valid_by_scenario = (rows
                           .groupby('scenario_id')[['n_valid', 'n_sample']]
                           .max()
                           .reset_index()
                           .sort_values('n_sample')
                           )
    return n_valid_by_scenario

@pytest.mark.parametrize("optimized_version", versions)
def test_table1_truth_matches_baseline(optimized_version):
    # Truth should match exactly between baseline and optimized versions
    optimized_table1 = all_version_tables[optimized_version]['table1']
    baseline_table1 = all_version_tables['baseline']['table1']

    np.testing.assert_allclose(optimized_table1['truth'], baseline_table1['truth'])

@pytest.mark.parametrize("optimized_version", versions)
def test_table1_estimators_are_close_to_baseline(optimized_version):
    # Optimized version differs in treatment assignment, so estimates should be close but not identical
    optimized_table1 = all_version_tables[optimized_version]['table1']
    baseline_table1 = all_version_tables['baseline']['table1']

    # Number of valid logit repetitions for experiment 1, one value per n_sample
    baseline_n_valid = get_n_valid('baseline', experiment="e1_table1")['n_valid'].to_numpy()
    optimized_n_valid = get_n_valid(optimized_version, experiment="e1_table1")['n_valid'].to_numpy()

    for estimator in TABLE1_ESTIMATORS:
        baseline_means = baseline_table1[f"{estimator}_mean"].to_numpy()
        baseline_sds = baseline_table1[f"{estimator}_sd"].to_numpy()
        optimized_means = optimized_table1[f"{estimator}_mean"].to_numpy()
        optimized_sds = optimized_table1[f"{estimator}_sd"].to_numpy()

        # Test similarity of means w/ Z-scores
        numerators = baseline_means - optimized_means
        denominators = np.sqrt(baseline_sds**2/baseline_n_valid + optimized_sds**2/optimized_n_valid)
        Z_scores = numerators / denominators
        Z_p_values = 2 * stats.norm.sf(np.abs(Z_scores))
        assert np.all(Z_p_values > 0.0005), f"{estimator}: Z = {Z_scores}, p = {Z_p_values}"

        # Test similarity of SDs w/ F-test
        F_scores = baseline_sds**2 / optimized_sds**2
        numerator_dfs = baseline_n_valid - 1
        denominator_dfs = optimized_n_valid - 1
        lower_tail = stats.f.cdf(F_scores, numerator_dfs, denominator_dfs)
        upper_tail = stats.f.sf(F_scores, numerator_dfs, denominator_dfs)
        p_values = 2 * np.minimum(lower_tail, upper_tail)

        assert np.all(p_values > 0.0005), f"{estimator}: F = {F_scores}, p = {p_values}"
        