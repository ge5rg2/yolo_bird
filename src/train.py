from ultralytics import YOLO
import os

# Path to the data config file
DATA_CONFIG = os.path.abspath("datasets/bird_v2/bird_v2.yaml")

def main():
    model = YOLO("yolo11n.pt")  #YOLO("configs/yolo11n.yaml")  # YOLOv11n 아키텍처 사용
    model.train(
        data=DATA_CONFIG,
        epochs=50, # 50~100
        imgsz=640,
        batch=8,  # 기존 16 → 8로 줄여 GPU 메모리 절약
        # lr0=0.01,  # 학습률 증가
        weight_decay=0.0001,
        # amp=True,
        name='train_yolov8s',
        device="mps" # M1/M2 a single GPU (device=0), multiple GPUs (device=0,1), CPU (device=cpu),
    )

if __name__ == "__main__":
    main()