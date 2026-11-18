import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import pytest
import warnings
import statsmodels.api as sm

from scenarios import SCENARIOS
from simulation import rng_for, DGP_CLASSES, METHOD_CLASSES
from metrics import get_valid_reps


# Baseline and optimized use the same random streams to sample from the population
#       They only differ in treatment assignment (not the same random streams)
#       So, results should be similar, but not identical

# Import data
versions = ["baseline", "optimized", "optimized_2cores", "optimized_4cores"]
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
ALPHA = 0.0005  

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

@pytest.mark.parametrize("version", versions)
def test_table1_truth_matches_baseline(version):
    # Truth should match exactly between baseline and optimized versions
    optimized_table1 = all_version_tables[version]['table1']
    baseline_table1 = all_version_tables['baseline']['table1']

    np.testing.assert_allclose(optimized_table1['truth'], baseline_table1['truth'])

@pytest.mark.parametrize("version", versions)
def test_table1_estimators_are_close_to_baseline(version):
    # Optimized version differs in treatment assignment, so estimates should be close but not identical
    optimized_table1 = all_version_tables[version]['table1']
    baseline_table1 = all_version_tables['baseline']['table1']

    # Number of valid logit repetitions for experiment 1, one value per n_sample
    baseline_n_valid = get_n_valid('baseline', experiment="e1_table1")['n_valid'].to_numpy()
    optimized_n_valid = get_n_valid(version, experiment="e1_table1")['n_valid'].to_numpy()

    for estimator in TABLE1_ESTIMATORS:
        baseline_means = baseline_table1[f"{estimator}_mean"].to_numpy()
        baseline_sds = baseline_table1[f"{estimator}_sd"].to_numpy()
        optimized_means = optimized_table1[f"{estimator}_mean"].to_numpy()
        optimized_sds = optimized_table1[f"{estimator}_sd"].to_numpy()

        # Test similarity of means w/ Z-tests
        numerators = baseline_means - optimized_means
        denominators = np.sqrt(baseline_sds**2/baseline_n_valid + optimized_sds**2/optimized_n_valid)
        Z_scores = numerators / denominators
        Z_p_values = 2 * stats.norm.sf(np.abs(Z_scores))
        assert np.all(Z_p_values > ALPHA), f"{estimator}: Z = {Z_scores}, p = {Z_p_values}"

        # Test similarity of SDs w/ F-test
        F_scores = baseline_sds**2 / optimized_sds**2
        numerator_dfs = baseline_n_valid - 1
        denominator_dfs = optimized_n_valid - 1
        lower_tail = stats.f.cdf(F_scores, numerator_dfs, denominator_dfs)
        upper_tail = stats.f.sf(F_scores, numerator_dfs, denominator_dfs)
        p_values = 2 * np.minimum(lower_tail, upper_tail)

        assert np.all(p_values > ALPHA), f"{estimator}: F = {F_scores}, p = {p_values}"

################################################################## 
#  Check that vectorized Newton-Rhapson methods match 
#       the pre-built model fitting functions used in the baseline
##################################################################
N_SIM = 30
SEED = 607
def generate_test_data(scenario_id, n_sim):
    '''
    Generate trt, Y_observed, and Z for one scenario
    '''
    settings = SCENARIOS[scenario_id]
    rng = rng_for(SEED, scenario_id, 0, "dgp")
    dgp_class = DGP_CLASSES[settings["covariate_distribution"]]

    dgp_settings = {
        "experiment": settings["experiment"],
        "n_sample": settings["n_sample"],
        "rng": rng,
        "n_sim": n_sim,
        "pi_treat": settings["pi_treat"],
        "cut_control": settings["cut_control"],
        "cut_treat": settings["cut_treat"],
        "observed_covariate": settings["observed_covariate"],
    }
    if settings["covariate_distribution"] == "beta":
        dgp_settings["alpha"] = settings["beta_dist_alpha"]
        dgp_settings["beta"] = settings["beta_dist_beta"]

    dgp = dgp_class(**dgp_settings)
    repetition_data = dgp.generate_repetitions()
    return repetition_data["trt"], repetition_data["Y_observed"], dgp.Z

def fit_statsmodels(model, trt, Y_observed, Z):
    '''
    Fits each repetition individually with statsmodels (like used in baseline version)
    '''
    n_sim, n_sample = trt.shape
    beta = np.full((n_sim, 3), np.nan)
    converged = np.zeros(n_sim, dtype=bool)

    for rep in range(n_sim):
        design_matrix = np.column_stack((np.ones(n_sample), trt[rep], Z))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if model == "logit":
                    fit = sm.Logit(Y_observed[rep], design_matrix).fit(disp=0)
                else:
                    fit = sm.Probit(Y_observed[rep], design_matrix).fit(disp=0)
                beta[rep] = fit.params
                converged[rep] = fit.mle_retvals["converged"]
            except Exception:
                pass  

    return beta, converged

@pytest.mark.parametrize("model", ["logit", "probit"])
@pytest.mark.parametrize("scenario_id", ["e1_n500", "e3_n100_ctrlcut0.95_trtcut1.5"])
def test_vectorized_matches_statsmodels(model, scenario_id):
    # Generate the same data used to fit both methods
    trt, Y_observed, Z = generate_test_data(scenario_id, N_SIM)

    # Fit vectorized and statsmodels iterated versions
    vector_beta, vector_converged = METHOD_CLASSES[model](trt, Y_observed, Z).fit_model()
    statsmodels_beta, statsmodels_converged = fit_statsmodels(model, trt, Y_observed, Z)

    # Only compare repetitions where both methods converged
    both_converged = vector_converged & statsmodels_converged
    assert np.any(both_converged), "No repetitions converged for both methods"

    np.testing.assert_allclose(vector_beta[both_converged], statsmodels_beta[both_converged],
                               rtol=1e-6, atol=1e-6)

################################################################## 
#  Check that vectorized Newton-Rhapson methods match 
#       the pre-built model fitting functions used in the baseline
##################################################################
def get_performance_data(version, scenario_id, model, estimator):
    '''
    Returns the row in performance.csv for specific scenario_id, model, and estimator
    '''
    performance = all_version_tables[version]['performance']
    is_scenario = performance['scenario_id'] == scenario_id
    is_model = performance['model'] == model
    is_estimator = performance['estimator'] == estimator
    return performance[is_scenario & is_model & is_estimator].iloc[0]

# Extract baseline performance
baseline_performance = all_version_tables['baseline']['performance']
performance_cases = []
for version in versions:
    for index in range(len(baseline_performance)):
        row = baseline_performance.iloc[index]
        performance_cases.append((version, row['scenario_id'], row['model'], row['estimator']))

# Test that estimator means match baseline
@pytest.mark.parametrize("version, scenario_id, model, estimator", performance_cases)
def test_performance_mean_matches_baseline(version, scenario_id, model, estimator):
    # Test similarity of means w/ Z-tests
    baseline_row = get_performance_data('baseline', scenario_id, model, estimator)
    optimized_row = get_performance_data(version, scenario_id, model, estimator)

    numerator = baseline_row['mean'] - optimized_row['mean']
    denominator = np.sqrt(baseline_row['sd']**2 / baseline_row['n_valid']
                          + optimized_row['sd']**2 / optimized_row['n_valid'])
    Z_score = numerator / denominator
    p_value = 2 * stats.norm.sf(np.abs(Z_score))

    assert p_value > ALPHA, f"Z = {Z_score:.3f}, p = {p_value:.4f}"


@pytest.mark.parametrize("version, scenario_id, model, estimator", performance_cases)
def test_performance_sd_matches_baseline(version, scenario_id, model, estimator):
    if scenario_id.startswith("e4_"):
        pytest.skip("E4 SDs tested separately")

    # Test similarity of SDs w/ F-test
    baseline_row = get_performance_data('baseline', scenario_id, model, estimator)
    optimized_row = get_performance_data(version, scenario_id, model, estimator)

    F_score = baseline_row['sd']**2 / optimized_row['sd']**2
    numerator_df = baseline_row['n_valid'] - 1
    denominator_df = optimized_row['n_valid'] - 1

    lower_tail = stats.f.cdf(F_score, numerator_df, denominator_df)
    upper_tail = stats.f.sf(F_score, numerator_df, denominator_df)
    p_value = 2 * np.minimum(lower_tail, upper_tail)

    assert p_value > ALPHA, f"F = {F_score:.3f}, p = {p_value:.4f}"

################################################################## 
#  Experiment 4 fails the above test for matching SD, examine this
#       by considering a KS test to compare distributions
##################################################################
RAW_DIR = Path("results/raw")
E4_ESTIMATORS = ["beta1_hat", "beta2_hat", "beta3_hat", "plug_in", "itt"]

# Get raw per-rep results for all versions
all_version_runs = {}
for version in versions:
    runs = pd.read_csv(RAW_DIR / version / "runs.csv")
    runs["failure_reason"] = runs["failure_reason"].fillna("")
    all_version_runs[version] = runs

# Get scenarios for experiment 4
E4_SCENARIOS = sorted(baseline_performance.loc[baseline_performance['experiment'] == 'e4_covariate',
                                               'scenario_id'].unique())

def get_valid_estimates(version, scenario_id, estimator):
    '''
    Valid per-repetition estimates for one scenario (logit model), using the same rule as metrics.py
    '''
    runs = all_version_runs[version]
    is_scenario = runs['scenario_id'] == scenario_id
    is_logit = runs['model'] == 'logit'
    valid, counts = get_valid_reps(runs[is_scenario & is_logit])
    return valid[estimator].to_numpy()

E4_DISTRIBUTION_CASES = []
for version in versions:
    for scenario_id in E4_SCENARIOS:
        for estimator in E4_ESTIMATORS:
            E4_DISTRIBUTION_CASES.append((version, scenario_id, estimator))

@pytest.mark.parametrize("version, scenario_id, estimator", E4_DISTRIBUTION_CASES)
def test_e4_distribution_ks(version, scenario_id, estimator):
    # Kolmogorov-Smirnov: compares the whole distributions
    baseline_estimates = get_valid_estimates('baseline', scenario_id, estimator)
    optimized_estimates = get_valid_estimates(version, scenario_id, estimator)

    result = stats.ks_2samp(baseline_estimates, optimized_estimates)
    assert result.pvalue > ALPHA, f"KS statistic = {result.statistic:.3f}, p = {result.pvalue:.4f}"

################################################################## 
#  Check that parallel and serial results match exactly
##################################################################
PARALLEL_VERSIONS = [version for version in versions if version.endswith("cores")]

@pytest.mark.parametrize("parallel_version", PARALLEL_VERSIONS)
def test_parallel_matches_serial(parallel_version):
    # Remove runtime since this will vary b/w versions
    serial_runs = all_version_runs['optimized'].drop(columns=['seconds'])
    parallel_runs = all_version_runs[parallel_version].drop(columns=['seconds'])

    pd.testing.assert_frame_equal(serial_runs, parallel_runs,
                                  check_exact=False, rtol=1e-10, atol=1e-12)
