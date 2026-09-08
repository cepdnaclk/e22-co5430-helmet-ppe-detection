import argparse
from pathlib import Path
from ultralytics import YOLO


def run_demo(model_path, source, output_dir):
    model_path = Path(model_path)
    source = Path(source)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not source.exists():
        raise FileNotFoundError(
            f"Input source not found: {source}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HELMET & PPE DETECTION DEMO")
    print("=" * 60)

    print(f"Model : {model_path}")
    print(f"Source: {source}")
    print(f"Output: {output_dir}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(source),
        save=True,
        project=str(output_dir),
        name="predictions",
        conf=0.25,
        verbose=True
    )

    print("\nDetection completed.")
    print(f"Results saved inside: {output_dir}/predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Run PPE detection on an image, folder, or video."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO model (.pt)"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to image, folder, or video"
    )

    parser.add_argument(
        "--output",
        default="results/images",
        help="Directory for prediction results"
    )

    args = parser.parse_args()

    run_demo(
        args.model,
        args.source,
        args.output
    )


if __name__ == "__main__":
    main()