"""
Run YOLO inference on an image, directory, or video.

Examples:

python src/inference/predict.py \
    --model models/best.pt \
    --source sample.jpg

python src/inference/predict.py \
    --model models/best.pt \
    --source dataset/test/images
"""

from pathlib import Path
import argparse

from ultralytics import YOLO


def main():

    parser = argparse.ArgumentParser(
        description="Run PPE detection inference."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO model"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Image, directory, video, or webcam source"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images for directory inference"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    source = Path(args.source)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not source.exists():
        raise FileNotFoundError(
            f"Source not found: {source}"
        )

    model = YOLO(str(model_path))

    print("=" * 60)
    print("PPE DETECTION INFERENCE")
    print("=" * 60)

    print(f"Model : {model_path}")
    print(f"Source: {source}")

    # If source is a directory, limit the number of images.
    if source.is_dir():

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }

        image_files = [
            file for file in source.iterdir()
            if file.suffix.lower() in image_extensions
        ]

        image_files = image_files[:args.max_images]

        if not image_files:
            raise RuntimeError(
                f"No supported images found in {source}"
            )

        print(f"Images selected: {len(image_files)}")

        for image in image_files:

            model.predict(
                source=str(image),
                conf=args.conf,
                imgsz=args.imgsz,
                save=True,
                save_txt=True,
                project="results",
                name="predictions",
                exist_ok=True
            )

    else:

        model.predict(
            source=str(source),
            conf=args.conf,
            imgsz=args.imgsz,
            save=True,
            save_txt=True,
            project="results",
            name="predictions",
            exist_ok=True
        )

    print("\nPredictions saved to:")
    print("results/predictions/")


if __name__ == "__main__":
    main()