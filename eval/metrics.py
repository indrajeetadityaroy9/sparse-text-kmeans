import numpy as np
from sklearn import metrics

class MetricSummary:
    def __init__(self, values):
        self.values = values


def compute_clustering_metrics(
    X,
    predicted_labels,
    true_labels,
    silhouette_sample_size=2000,
    random_state=None,
):
    scores = {
        "homogeneity": metrics.homogeneity_score(true_labels, predicted_labels),
        "completeness": metrics.completeness_score(true_labels, predicted_labels),
        "v_measure": metrics.v_measure_score(true_labels, predicted_labels),
        "adjusted_rand": metrics.adjusted_rand_score(true_labels, predicted_labels),
    }

    if silhouette_sample_size is not None:
        sample_size = min(len(predicted_labels), silhouette_sample_size)
        scores["silhouette"] = metrics.silhouette_score(
            X, predicted_labels, sample_size=sample_size, random_state=random_state
        )

    return MetricSummary(values=scores)


def aggregate_metrics(metric_summaries):
    collected = {}
    for summary in metric_summaries:
        for key, value in summary.values.items():
            collected.setdefault(key, []).append(value)

    aggregated = {}
    for metric_name, metric_values in collected.items():
        arr = np.asarray(metric_values)
        aggregated[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return aggregated
