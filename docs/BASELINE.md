# Overall Runtime
Overall, the entire simulation study took 25.5 seconds to run.

*see results/profiling/baseline_total_runtime.txt*

# Profiling Results

The largest bottle necks are the model fitting. Both the probit and logit model fitting in the model take the most amount of time to run, and they are called multiple times by all repetitions.

*see results/profiling*

# Computational Complexity

While the overall simulation study took very little time to run, this is partially due to the small sample sizes used.
Adding larger values of n increased the run time for all experiments, and the run time by n did not vary strongly between experiments.

![Runtime vs n for baseline version](../results/complexity/baseline_runtime_vs_n.png)

# Numerical Warnings

There were many numerical and convergence warnings resulting from the simulation study.

Experiments 3, 4, and 5 all recieved convergence warnings from the maximum likelihood optimization failing to converge (529, 4, and 15 such warnings for experiments 3, 4, and 5, respecitvely). Experiment 4 also recieved runtime errors from dividing by 0, takig the log 0, and exponential overflow (2, 4, and 70 such warnings, respectively).

Experiments 3 and 5 recieved value errors from recieveing a probability of 0 (529 and 15 such errors respectively). Experiment 4 recieved 1 error from a singular matrix.

*see results/warnings/baseline for full tables of error and warnign counts*