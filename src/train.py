from ultralytics import YOLO
import os

checkpoint = "C:/ppe/runs/detect/train/weights/last.pt"

if os.path.exists(checkpoint):
    # Resume interrupted training from last saved epoch
    model = YOLO(checkpoint)
    model.train(resume=True)
else:
    # Start baseline training from pretrained YOLOv8n
    model = YOLO("yolov8n.pt")
    model.train(
        data="C:/ppe/dataset/ppe-data/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16
    )