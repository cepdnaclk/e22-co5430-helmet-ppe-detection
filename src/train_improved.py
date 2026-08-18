from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/ppe/dataset/ppe-data/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.7,
        project="C:/ppe/runs",
        name="improved"
    )

if __name__ == "__main__":
    main()