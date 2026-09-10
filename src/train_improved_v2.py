from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/ppe/dataset/ppe-data/data.yaml",
        epochs=50,
        patience=15,          # stop early if no improvement for 15 epochs, saves time if it plateaus
        imgsz=640,
        batch=16,
        hsv_h=0.015,           # default-strength, mild hue jitter
        hsv_s=0.4,             # much gentler than 0.9
        hsv_v=0.3,
        degrees=5.0,            # gentler rotation
        translate=0.1,          # gentler translation
        scale=0.3,              # gentler scale range
        fliplr=0.5,             # standard horizontal flip
        mosaic=1.0,              # standard mosaic augmentation
        project="C:/ppe/runs",
        name="improved_v2"
    )

if __name__ == "__main__":
    main()