from ultralytics import YOLO

def validate():
    model = YOLO("runs/detect/train2/weights/best.pt")
    results = model.predict("test.jpg", imgsz=640, conf=0.01, save=True)  # 낮은 confidence로 검출
    print(results)

if __name__ == "__main__":
    validate()
