from ultralytics import YOLO

model = YOLO("C:/ppe/runs/improved-2/weights/best.pt")
model.predict(
    source="C:/ppe/dataset/ppe-data/test/images",
    save=True,
    conf=0.25,
    project="C:/ppe/results",
    name="improved_predictions"
)