from ultralytics import YOLO

model = YOLO("C:/ppe/runs/detect/train/weights/best.pt")
model.predict(
    source="C:/ppe/dataset/ppe-data/test/images",
    save=True,
    conf=0.25
)