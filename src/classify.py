from ultralytics import YOLO
import cv2

def validate():
    # 모델 로드 (detect 또는 classification 모델인지 확인)
    model = YOLO("runs/detect/train_yolov11n2/weights/best.pt")
    
    # 이미지 로드 및 확인
    image_path = "test.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    # 다양한 confidence 값으로 테스트
    confidence_levels = [0.01, 0.1, 0.25, 0.5]
    
    for conf in confidence_levels:
        print(f"\nConfidence 임계값: {conf}")
        
        # 모델 예측
        results = model.predict(
            image_path, 
            imgsz=640, 
            conf=conf,  # 낮은 confidence로 시도
            save=True  # 결과 이미지 저장
        )
        
        # 결과 분석
        for result in results:
            # 박스 정보 확인
            boxes = result.boxes
            
            if len(boxes) > 0:
                print(f"탐지된 객체 수: {len(boxes)}")
                for box in boxes:
                    # 클래스 이름과 confidence 출력
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    name = model.names[cls]
                    print(f"클래스: {name}, 신뢰도: {conf.item()*100:.2f}%")
            else:
                print("탐지된 객체 없음")

if __name__ == "__main__":
    validate()