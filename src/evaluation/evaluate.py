"""
Evaluate a YOLO object detection model.

Examples:

Baseline:
python src/evaluation/evaluate.py ^
    --model models/baseline.pt ^
    --data dataset/data.yaml ^
    --output baseline_metrics.csv

Improved:
python src/evaluation/evaluate.py ^
    --model models/best.pt ^
    --data dataset/data.yaml ^
    --output improved_metrics.csv
"""

from pathlib import Path
import argparse

from ultralytics import YOLO

from metrics import extract_metrics, save_metrics


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO PPE detection model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO model (.pt)"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml"
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--output",
        default="evaluation_metrics.csv",
        help="Output CSV filename"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}"
        )

    print("\nLoading model...")
    model = YOLO(str(model_path))

    print("\nRunning evaluation...")
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        plots=True,
        save_json=True
    )

    metrics = extract_metrics(results)

    output_path = (
        Path("results")
        / "metrics"
        / args.output
    )

    save_metrics(
        metrics,
        output_path
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1']:.4f}")
    print(f"mAP@0.5        : {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   : {metrics['mAP50-95']:.4f}")

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()