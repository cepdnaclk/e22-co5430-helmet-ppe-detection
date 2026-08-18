"""
Utility functions for extracting and saving YOLO object-detection metrics.

The actual values are generated from a trained model evaluation.
No metrics are hard-coded.
"""

from pathlib import Path
import csv


def extract_metrics(results):
    """
    Extract common object detection metrics from an Ultralytics
    validation result.

    Parameters
    ----------
    results:
        Ultralytics validation result object.

    Returns
    -------
    dict
        Dictionary containing precision, recall, mAP50 and mAP50-95.
    """

    metrics = results.box

    precision = float(metrics.mp)
    recall = float(metrics.mr)
    map50 = float(metrics.map50)
    map5095 = float(metrics.map)

    # F1 = 2PR / (P + R)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "mAP50-95": map5095,
    }


def save_metrics(metrics, output_file):
    """
    Save evaluation metrics to a CSV file.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Value"
        ])

        writer.writerow(["Precision", metrics["precision"]])
        writer.writerow(["Recall", metrics["recall"]])
        writer.writerow(["F1", metrics["f1"]])
        writer.writerow(["mAP@0.5", metrics["mAP50"]])
        writer.writerow(["mAP@0.5:0.95", metrics["mAP50-95"]])