from ultralytics import YOLO
import os

# Path to the data config file
DATA_CONFIG = os.path.abspath("datasets/bird_v1/bird_v1.yaml")

def main():
    model = YOLO("configs/yolo11n.yaml")  # YOLOv11n 아키텍처 사용
    model.train(
        data=DATA_CONFIG,
        epochs=100, 
        imgsz=640,
        batch=8,  # 기존 16 → 8로 줄여 GPU 메모리 절약
        lr0=0.005,  # 학습률 증가
        weight_decay=0.0001,
        amp=True,
        device="mps" # M1/M2 
    )

if __name__ == "__main__":
    main()