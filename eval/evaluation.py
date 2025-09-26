import json
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import MiniBatchKMeans
from custom_kmeans import CustomKMeans
from .data import prepare_dataset
from .metrics import aggregate_metrics, compute_clustering_metrics
from .significance import paired_significance_tests


class VectorizerSetting:
    def __init__(self, label, vectorizer_config, dimensionality_reduction=None):
        self.label = label
        self.vectorizer_config = vectorizer_config
        self.dimensionality_reduction = dimensionality_reduction


class ModelSpec:
    def __init__(self, label, family, params, n_clusters=None):
        self.label = label
        self.family = family
        self.params = params
        self.n_clusters = n_clusters


class ExperimentConfig:
    def __init__(
        self,
        subset="all",
        remove_metadata=True,
        categories=None,
        vectorizer_settings=None,
        models=None,
        seeds=None,
        silhouette_sample_size=2000,
        significance_alpha=0.05,
        output_dir=None,
    ):
        self.subset = subset
        self.remove_metadata = remove_metadata
        self.categories = categories
        self.vectorizer_settings = list(vectorizer_settings) if vectorizer_settings is not None else []
        self.models = list(models) if models is not None else []
        self.seeds = list(seeds) if seeds is not None else list(range(5))
        self.silhouette_sample_size = silhouette_sample_size
        self.significance_alpha = significance_alpha
        self.output_dir = output_dir


class SeedResult:
    def __init__(self, seed, metrics, inertia, runtime_sec, iterations=None, converged=None):
        self.seed = seed
        self.metrics = metrics
        self.inertia = inertia
        self.runtime_sec = runtime_sec
        self.iterations = iterations
        self.converged = converged


class ModelEvaluation:
    def __init__(self, model_label, seed_results, aggregated_metrics):
        self.model_label = model_label
        self.seed_results = seed_results
        self.aggregated_metrics = aggregated_metrics


class VectorizerEvaluation:
    def __init__(self, vectorizer_label, dataset, model_evaluations, significance):
        self.vectorizer_label = vectorizer_label
        self.dataset = dataset
        self.model_evaluations = model_evaluations
        self.significance = significance


class EvaluationRunner:
    def __init__(self, config):
        if not config.vectorizer_settings:
            raise ValueError("At least one vectorizer setting must be specified")
        if not config.models:
            raise ValueError("At least one model specification is required")
        self.config = config

    def run(self):
        results = []
        for vectorizer_setting in self.config.vectorizer_settings:
            dataset_bundle = prepare_dataset(
                vectorizer_config=vectorizer_setting.vectorizer_config,
                subset=self.config.subset,
                remove_metadata=self.config.remove_metadata,
                categories=self.config.categories,
                random_state=self.config.seeds[0],
                dimensionality_reduction=vectorizer_setting.dimensionality_reduction,
            )
            vectorizer_result = self._evaluate_vectorizer_setting(
                vectorizer_setting=vectorizer_setting,
                dataset_bundle=dataset_bundle,
            )
            results.append(vectorizer_result)
            self._maybe_persist(vectorizer_setting.label, vectorizer_result)
        return results

    def _evaluate_vectorizer_setting(self, vectorizer_setting, dataset_bundle):
        model_evaluations = {}

        for model_spec in self.config.models:
            seed_results = self._run_model_across_seeds(
                model_spec=model_spec,
                dataset_bundle=dataset_bundle,
            )
            aggregated = aggregate_metrics([sr.metrics for sr in seed_results])
            model_evaluations[model_spec.label] = ModelEvaluation(
                model_label=model_spec.label,
                seed_results=seed_results,
                aggregated_metrics=aggregated,
            )

        significance = self._run_significance_tests(model_evaluations)
        return VectorizerEvaluation(
            vectorizer_label=vectorizer_setting.label,
            dataset=dataset_bundle,
            model_evaluations=model_evaluations,
            significance=significance,
        )

    def _run_model_across_seeds(self, model_spec, dataset_bundle):
        seed_results = []
        for seed in self.config.seeds:
            model = self._build_model(model_spec, dataset_bundle, seed)
            start = time.perf_counter()
            model.fit(dataset_bundle.data)
            runtime = time.perf_counter() - start
            metrics = compute_clustering_metrics(
                dataset_bundle.data,
                model.labels_,
                dataset_bundle.labels,
                silhouette_sample_size=self.config.silhouette_sample_size,
                random_state=seed,
            )
            inertia = getattr(model, "inertia_", float("nan"))
            fit_result = getattr(model, "fit_result_", None)
            seed_results.append(
                SeedResult(
                    seed=seed,
                    metrics=metrics,
                    inertia=float(inertia) if inertia is not None else float("nan"),
                    runtime_sec=runtime,
                    iterations=getattr(fit_result, "iterations", None),
                    converged=getattr(fit_result, "converged", None),
                )
            )
        return seed_results

    def _build_model(self, model_spec, dataset_bundle, seed):
        params = dict(model_spec.params)
        params.setdefault("n_clusters", model_spec.n_clusters or int(np.unique(dataset_bundle.labels).size))
        params.setdefault("random_state", seed)

        family = model_spec.family.lower()
        if family == "custom":
            return CustomKMeans(**params)
        if family == "sklearn":
            return SKLearnKMeans(**params)
        if family == "minibatch":
            params.setdefault("batch_size", 1024)
            return MiniBatchKMeans(**params)
        raise ValueError(f"Unsupported model family: {model_spec.family}")

    def _run_significance_tests(self, model_evaluations):
        if "custom" not in {spec.family for spec in self.config.models}:
            return {}

        custom_keys = [spec.label for spec in self.config.models if spec.family == "custom"]
        if not custom_keys:
            return {}
        custom_label = custom_keys[0]
        custom_eval = model_evaluations[custom_label]

        significance = {}
        for label, evaluation in model_evaluations.items():
            if label == custom_label:
                continue
            metric_names = custom_eval.aggregated_metrics.keys()
            metric_results = {}
            for metric_name in metric_names:
                custom_values = [sr.metrics.values[metric_name] for sr in custom_eval.seed_results]
                candidate_values = [sr.metrics.values[metric_name] for sr in evaluation.seed_results]
                result = paired_significance_tests(
                    metric_name=metric_name,
                    custom_values=custom_values,
                    baseline_values=candidate_values,
                    alpha=self.config.significance_alpha,
                )
                metric_results[metric_name] = result
            significance[label] = metric_results
        return significance

    def _maybe_persist(self, label, evaluation):
        if self.config.output_dir is None:
            return
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{label}_results.json"
        payload = self._serialize_evaluation(evaluation)
        file_path.write_text(json.dumps(payload, indent=2))

    def _serialize_evaluation(self, evaluation):
        serialized = {
            "vectorizer": evaluation.vectorizer_label,
            "model_results": {},
            "significance": {},
        }
        for label, model_eval in evaluation.model_evaluations.items():
            serialized["model_results"][label] = {
                "aggregated": model_eval.aggregated_metrics,
                "seed_results": [
                    {
                        "seed": sr.seed,
                        "metrics": sr.metrics.values,
                        "inertia": sr.inertia,
                        "runtime_sec": sr.runtime_sec,
                        "iterations": sr.iterations,
                        "converged": sr.converged,
                    }
                    for sr in model_eval.seed_results
                ],
            }
        for label, metric_results in evaluation.significance.items():
            serialized["significance"][label] = {
                metric: {
                    "mean_difference": result.mean_difference,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "p_value": result.p_value,
                }
                for metric, result in metric_results.items()
            }
        return serialized
