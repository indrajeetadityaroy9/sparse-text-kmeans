import numpy as np
from scipy import stats


class SignificanceResult:
    def __init__(self, metric, mean_difference, ci_low, ci_high, p_value):
        self.metric = metric
        self.mean_difference = mean_difference
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.p_value = p_value


def bootstrap_confidence_interval(
    sample,
    alpha=0.05,
    n_bootstrap=2000,
    random_state=None,
):
    sample = np.asarray(list(sample))
    if sample.size == 0:
        raise ValueError("Sample must contain at least one element")

    rng = np.random.RandomState(random_state)
    bootstrap_samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        resample = rng.choice(sample, size=sample.size, replace=True)
        bootstrap_samples[i] = np.mean(resample)
    lower = float(np.percentile(bootstrap_samples, 100 * (alpha / 2)))
    upper = float(np.percentile(bootstrap_samples, 100 * (1 - alpha / 2)))
    return lower, upper


def paired_significance_tests(
    metric_name,
    custom_values,
    baseline_values,
    alpha=0.05,
):
    custom_arr = np.asarray(list(custom_values), dtype=float)
    baseline_arr = np.asarray(list(baseline_values), dtype=float)
    if custom_arr.shape != baseline_arr.shape:
        raise ValueError("Custom and baseline arrays must share the same shape for paired tests")
    if custom_arr.size < 2:
        raise ValueError("At least two paired samples are required for significance testing")

    diff = custom_arr - baseline_arr
    mean_diff = float(np.mean(diff))
    ci_low, ci_high = bootstrap_confidence_interval(diff, alpha=alpha)
    t_stat, p_value = stats.ttest_rel(custom_arr, baseline_arr)

    return SignificanceResult(metric_name, mean_diff, ci_low, ci_high, float(p_value))
