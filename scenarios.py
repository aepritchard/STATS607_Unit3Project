'''
Scenario definitions for the simulation study
Each key is a unique, filename-safe scenario ID (also used to derive its random stream)
Each value holds the settings run_scenario reads
'''

make_scenarios_for_all_n_samples = [100, 500, 1000, 5000]

# Define baseline scenarios
BASELINE = {
    "covariate_distribution": "uniform",
    "models": ["logit"],
    "pi_treat": 0.75,
    "cut_control": 0.5,
    "cut_treat": 0.75,
    "observed_covariate": "V",
}

SCENARIOS = {}

def make_scenario_for_n_sample(n_sample, SCENARIOS):

    #### Experiment 1: Reproduce Table 1
    # 25% of subjects assigned to control, 75% assigned to trt
    # Control cut off = 1/2, treatment cut off = 3/4
    # Observed covariate Z = V
    # Both logit and probit models are fit (probit: Section 6)

    SCENARIOS[f"e1_n{n_sample}"] = {
        **BASELINE,
        "experiment": "e1_table1",
        "n_sample": n_sample,
        "models": ["logit", "probit"],
    }

    #### Experiment 2: Change % treated
    # Use same set-up as experiment 1, BUT change the % assigned to trt/ctrl
    # Specific values were not specified in the paper
    # Use 0.75, 0.5, 0.25
    for pi_treat in [0.75, 0.50, 0.25]:
        SCENARIOS[f"e2_pi{int(pi_treat * 100)}_n{n_sample}"] = {
            **BASELINE,
            "experiment": "e2_pct_treated",
            "n_sample": n_sample,
            "pi_treat": pi_treat,
        }

    #### Experiment 3: Change cut offs
    # Use same set-up as experiment 1, BUT change control and treatment cut offs
    # Control cut off = 0.95, treatment cut off = 1.5
    SCENARIOS[f"e3_n{n_sample}_ctrlcut0.95_trtcut1.5"] = {
                    **BASELINE,
                    "experiment": "e3_cutoffs",
                    "n_sample": n_sample,
                    "cut_control": 0.95,
                    "cut_treat": 1.5,
                }
            

    #### Experiment 4: Change observed covariate
    # Use same set-up as experiment 1, BUT change observed covariate
    # Observed covariate Z = U + V
    SCENARIOS[f"e4_n{n_sample}"] = {
        **BASELINE,
        "experiment": "e4_covariate",
        "n_sample": n_sample,
        "observed_covariate": "U+V",
    }

    #### Experiment 5: Change joint distribution
    # Use same set-up as experiment 1, BUT change joint distribution of (U_i, V_i)
    # Specific distribution used was not specified in the paper
    # U_i, V_i IID Beta(2, 5) (same support as Uniform(0, 1))
    SCENARIOS[f"e5_n{n_sample}"] = {
        **BASELINE,
        "experiment": "e5_joint_dist",
        "n_sample": n_sample,
        "covariate_distribution": "beta",
        "beta_dist_alpha": 2.0,
        "beta_dist_beta": 5.0,
    }

def make_scenarios_for_all_n_samples(n_samples, SCENARIOS):
    for n_sample in n_samples:
        make_scenario_for_n_sample(n_sample, SCENARIOS)

make_scenarios_for_all_n_samples(n_samples, SCENARIOS)