from ultralytics import YOLO

model = YOLO("C:/ppe/runs/detect/train/weights/best.pt")
model.val()