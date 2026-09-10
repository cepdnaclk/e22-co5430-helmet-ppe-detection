"""
Baseline YOLOv8n training for Construction Site PPE Detection.

This creates the baseline model used for comparison
with the improved/fine-tuned model.
"""

from pathlib import Path
from ultralytics import YOLO


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET = "dataset/data.yaml"

MODEL = "yolov8n.pt"

EPOCHS = 20
IMAGE_SIZE = 640
BATCH_SIZE = 16

PROJECT = "runs"
RUN_NAME = "baseline"


def main():

    print("=" * 60)
    print("CONSTRUCTION SITE PPE DETECTION")
    print("YOLOv8n BASELINE TRAINING")
    print("=" * 60)

    # Check that dataset configuration exists
    if not Path(DATASET).exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET}"
        )

    print(f"Dataset       : {DATASET}")
    print(f"Model         : {MODEL}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Image size    : {IMAGE_SIZE}")
    print(f"Batch size    : {BATCH_SIZE}")
    print()

    # Load pretrained YOLOv8 nano model
    model = YOLO(MODEL)

    # Train baseline
    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=RUN_NAME,
        verbose=True
    )

    print()
    print("=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)

    print("Expected best model:")
    print("runs/baseline/weights/best.pt")


if __name__ == "__main__":
    main()