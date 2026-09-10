from ultralytics import YOLO

def run_predictions(model_path, run_name, source="C:/ppe/dataset/ppe-data/test/images"):
    model = YOLO(model_path)
    model.predict(
        source=source,
        save=True,
        project="C:/ppe/runs",
        name=run_name,
        exist_ok=True
    )

if __name__ == "__main__":
    # Baseline predictions already exist at runs/detect/predict — skip re-running unless needed
    run_predictions(
        "C:/ppe/runs/improved_v2/weights/best.pt",
        "no_augmentation_predict"
    )