from ultralytics import YOLO

def main():
    model = YOLO("C:/ppe/runs/improved/weights/last.pt")
    model.train(resume=True)

if __name__ == "__main__":
    main()