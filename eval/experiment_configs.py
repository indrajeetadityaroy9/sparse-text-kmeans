from pathlib import Path
from .evaluation import ExperimentConfig, ModelSpec, VectorizerSetting

def build_default_experiment(output_dir=Path("research_results")):
    vectorizers = [
        VectorizerSetting(
            label="tfidf_default",
            vectorizer_config={
                "type": "tfidf",
                "max_df": 0.5,
                "min_df": 5,
                "stop_words": "english",
            },
        ),
        VectorizerSetting(
            label="tfidf_bigram",
            vectorizer_config={
                "type": "tfidf",
                "max_df": 0.5,
                "min_df": 3,
                "stop_words": "english",
                "ngram_range": (1, 2),
                "max_features": 50000,
            },
            dimensionality_reduction={
                "method": "svd",
                "n_components": 300,
            },
        ),
        VectorizerSetting(
            label="count_binary",
            vectorizer_config={
                "type": "count",
                "max_df": 0.4,
                "min_df": 5,
                "stop_words": "english",
                "binary": True,
            },
        ),
    ]

    models = [
        ModelSpec(
            label="custom_kmeans",
            family="custom",
            params={
                "max_iterations": 500,
                "tolerance": 5e-5,
                "n_init": 10,
            },
        ),
        ModelSpec(
            label="sklearn_kmeans",
            family="sklearn",
            params={
                "max_iter": 500,
                "tol": 5e-5,
                "n_init": 10,
            },
        ),
        ModelSpec(
            label="minibatch_kmeans",
            family="minibatch",
            params={
                "max_iter": 500,
                "batch_size": 2048,
                "n_init": 10,
            },
        ),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return ExperimentConfig(
        vectorizer_settings=vectorizers,
        models=models,
        seeds=seeds,
        silhouette_sample_size=4000,
        significance_alpha=0.01,
        output_dir=output_dir,
    )
