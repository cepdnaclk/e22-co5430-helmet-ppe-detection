"""
Compare baseline and improved model metrics.

Expected files:

results/metrics/baseline_metrics.csv
results/metrics/improved_metrics.csv
"""

from pathlib import Path
import csv


def read_metrics(file_path):

    metrics = {}

    with open(file_path, "r", encoding="utf-8") as file:

        reader = csv.reader(file)

        next(reader)

        for row in reader:
            metrics[row[0]] = float(row[1])

    return metrics


def main():

    baseline_file = Path(
        "results/metrics/baseline_metrics.csv"
    )

    improved_file = Path(
        "results/metrics/improved_metrics.csv"
    )

    if not baseline_file.exists():
        raise FileNotFoundError(
            "Baseline metrics file not found."
        )

    if not improved_file.exists():
        raise FileNotFoundError(
            "Improved metrics file not found."
        )

    baseline = read_metrics(baseline_file)
    improved = read_metrics(improved_file)

    output_file = Path(
        "results/metrics/model_comparison.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Baseline",
            "Improved",
            "Change"
        ])

        for metric in baseline:

            base_value = baseline[metric]
            improved_value = improved[metric]

            change = improved_value - base_value

            writer.writerow([
                metric,
                f"{base_value:.4f}",
                f"{improved_value:.4f}",
                f"{change:+.4f}"
            ])

    print("\nModel comparison:")
    print("=" * 60)

    for metric in baseline:

        base = baseline[metric]
        improved_value = improved[metric]

        print(
            f"{metric:15s} "
            f"Baseline={base:.4f} "
            f"Improved={improved_value:.4f} "
            f"Change={improved_value - base:+.4f}"
        )

    print("\nSaved to:")
    print(output_file)


if __name__ == "__main__":
    main()