from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/ppe/dataset/ppe-data/data.yaml",
        epochs=50,
        patience=15,
        imgsz=640,
        batch=16,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
        project="C:/ppe/runs",
        name="no_augmentation"
    )

if __name__ == "__main__":
    main()