"""
Generates per-class and overall evaluation metrics for a trained
YOLO model: precision, recall, F1, mAP50, mAP50-95, and inference
speed (ms/image, FPS).

Correctly maps Ultralytics' internal per-class result arrays to
class names using metrics.box.ap_class_index, since not every class
necessarily appears in a given validation pass (naive index-based
mapping would misalign results for missing classes).

Outputs a CSV with an "all" summary row followed by per-class rows.
"""
from ultralytics import YOLO
import pandas as pd
import time
import glob

def compute_fps(model, test_images_dir="C:/ppe/dataset/ppe-data/test/images", n_samples=20):
    """Measure average inference time and FPS over a sample of test images."""
    images = glob.glob(f"{test_images_dir}/*.jpg")[:n_samples]
    if not images:
        return None, None
    # Warm-up run (first inference is always slower)
    model.predict(images[0], verbose=False)
    start = time.time()
    for img in images:
        model.predict(img, verbose=False)
    elapsed = time.time() - start
    avg_time_ms = (elapsed / len(images)) * 1000
    fps = 1000 / avg_time_ms
    return round(avg_time_ms, 2), round(fps, 2)

def export_metrics(model_path, output_csv, data_yaml="C:/ppe/dataset/ppe-data/data.yaml"):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    p, r, ap50, ap = metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap
    class_indices = metrics.box.ap_class_index
    names = metrics.names

    rows = []
    for pos, cls_id in enumerate(class_indices):
        precision = float(p[pos])
        recall = float(r[pos])
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        rows.append({
            "class": names[cls_id],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mAP50": round(float(ap50[pos]), 4),
            "mAP50-95": round(float(ap[pos]), 4),
        })

    df = pd.DataFrame(rows)

    overall_p = float(metrics.box.mp)
    overall_r = float(metrics.box.mr)
    overall_f1 = 0.0 if (overall_p + overall_r) == 0 else 2 * overall_p * overall_r / (overall_p + overall_r)

    avg_ms, fps = compute_fps(model)

    overall_row = pd.DataFrame([{
        "class": "all",
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }])

    final = pd.concat([overall_row, df], ignore_index=True)
    final.to_csv(output_csv, index=False)

    # Also save inference speed as a separate small summary line appended to the file
    with open(output_csv, "a") as f:
        f.write(f"\n# Inference speed (CPU, averaged over 20 test images): {avg_ms} ms/image, {fps} FPS\n")

    print(f"Saved: {output_csv}")
    print(final)
    print(f"Inference speed: {avg_ms} ms/image ({fps} FPS)")


# NOTE: model paths below point to this project's actual training run
# locations. If you re-run src/training/train_baseline.py or
# train_improved.py fresh, outputs will instead land in
# runs/baseline/weights/best.pt and runs/improved/weights/best.pt
# respectively — update the paths below to match if reproducing from scratch.
if __name__ == "__main__":
    export_metrics(
        "C:/ppe/runs/detect/train/weights/best.pt",
        "C:/ppe/results/metrics/baseline_metrics.csv"
    )
    export_metrics(
        "C:/ppe/runs/no_augmentation/weights/best.pt",
        "C:/ppe/results/metrics/no_augmentation_metrics.csv"
    )
    export_metrics(
        "C:/ppe/runs/improved_v2/weights/best.pt",
        "C:/ppe/results/metrics/improved_metrics.csv"
    )