import argparse
import json
from pathlib import Path
from .evaluation import EvaluationRunner
from .experiment_configs import build_default_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_results"),
        help="Directory to store detailed JSON outputs",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices={"all", "train", "test"},
        help="20 Newsgroups subset to load",
    )
    parser.add_argument(
        "--categories-file",
        type=Path,
        help="Optional path to a newline-delimited list of category names to filter",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Retain headers, footers, and quotes during preprocessing",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        help="Custom list of random seeds to use",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to export the aggregated summary as JSON",
    )
    return parser.parse_args()


def load_categories(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def render_summary(evaluations, output_path=None):
    summary = []
    for evaluation in evaluations:
        vectorizer_entry = {
            "vectorizer": evaluation.vectorizer_label,
            "models": {},
            "significance": {},
        }
        for label, model_eval in evaluation.model_evaluations.items():
            vectorizer_entry["models"][label] = model_eval.aggregated_metrics
        for label, tests in evaluation.significance.items():
            vectorizer_entry["significance"][label] = {
                metric: {
                    "mean_difference": result.mean_difference,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "p_value": result.p_value,
                }
                for metric, result in tests.items()
            }
        summary.append(vectorizer_entry)

    if output_path:
        output_path.write_text(json.dumps(summary, indent=2))

    for entry in summary:
        print(f"Vectorizer: {entry['vectorizer']}")
        for model, metrics in entry["models"].items():
            print(f"  Model: {model}")
            for metric_name, stats in metrics.items():
                mean = stats["mean"]
                std = stats["std"]
                print(f"    {metric_name}: mean={mean:.4f}, std={std:.4f}")
        if entry["significance"]:
            print("  Significance (custom vs baseline):")
            for label, tests in entry["significance"].items():
                print(f"    Baseline: {label}")
                for metric, stats in tests.items():
                    print(
                        "      {0}: mean_diff={1:.4f}, CI=({2:.4f}, {3:.4f}), p={4:.4g}".format(
                            metric,
                            stats["mean_difference"],
                            stats["ci_low"],
                            stats["ci_high"],
                            stats["p_value"],
                        )
                    )
        print()


def main():
    args = parse_args()
    config = build_default_experiment(output_dir=args.output_dir)
    config.subset = args.subset
    config.remove_metadata = not args.keep_metadata

    if args.categories_file:
        config.categories = load_categories(args.categories_file)
    if args.seeds:
        config.seeds = args.seeds
    if args.output_dir:
        config.output_dir = args.output_dir

    runner = EvaluationRunner(config)
    evaluations = runner.run()
    render_summary(evaluations, output_path=args.summary_json)


if __name__ == "__main__":
    main()
