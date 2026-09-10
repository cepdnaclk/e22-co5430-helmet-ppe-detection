"""
Improved YOLOv8n training for Construction Site PPE Detection.

Improvement over baseline:
- Longer fine-tuning
- Explicit augmentation
- Same model size and image size for fair comparison
"""

from pathlib import Path
from ultralytics import YOLO


DATASET = "dataset/data.yaml"
MODEL = "yolov8n.pt"

EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 16


def main():

    print("=" * 60)
    print("YOLOv8n IMPROVED PPE DETECTION TRAINING")
    print("=" * 60)

    if not Path(DATASET).exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET}"
        )

    print(f"Dataset    : {DATASET}")
    print(f"Model      : {MODEL}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")

    model = YOLO(MODEL)

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Improved augmentation configuration
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,

        flipud=0.0,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        # Save run separately
        project="runs",
        name="improved",

        patience=10,
        verbose=True
    )

    print()
    print("=" * 60)
    print("IMPROVED TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()