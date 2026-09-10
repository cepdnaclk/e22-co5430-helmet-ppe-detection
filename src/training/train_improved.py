"""
Improved YOLOv8n training for Construction Site PPE Detection.

This is the final improved model configuration, selected after an
ablation study comparing baseline, no-augmentation, and multiple
augmentation strengths (see results/metrics/model_comparison.csv
and results/failure_cases/analysis.md for full details).

Key findings from the ablation:
- Training longer (50 epochs vs. baseline's 20) alone improves
  results substantially, since the baseline had not converged.
- Aggressive augmentation (hsv_s=0.9, degrees=10, scale=0.7) hurt
  performance relative to baseline when using a short epoch budget.
- This configuration (gentler, tuned augmentation + 50 epochs)
  outperforms baseline on every metric: precision, recall, mAP50,
  and mAP50-95.
"""

from pathlib import Path
from ultralytics import YOLO


DATASET = "dataset/data.yaml"
MODEL = "yolov8n.pt"

EPOCHS = 50
PATIENCE = 15
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
    print(f"Epochs     : {EPOCHS} (patience={PATIENCE})")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")

    model = YOLO(MODEL)

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Tuned augmentation configuration (gentler than an earlier,
        # rejected configuration that used hsv_s=0.9/degrees=10/scale=0.7
        # and underperformed baseline — see analysis.md)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        degrees=5.0,
        translate=0.1,
        scale=0.3,

        fliplr=0.5,
        mosaic=1.0,

        project="runs",
        name="improved",

        verbose=True
    )

    print()
    print("=" * 60)
    print("IMPROVED TRAINING COMPLETE")
    print("=" * 60)
    print("Expected best model:")
    print("runs/improved/weights/best.pt")


if __name__ == "__main__":
    main()