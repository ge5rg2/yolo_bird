import os
import shutil
import numpy as np
from pathlib import Path

# 경로 설정 (실제 경로로 수정하세요)
CUB_DATASET_PATH = '/Users/kyuhyeonkim/test_local/service/CUB_200_2011'  # CUB-200-2011 데이터셋 경로
YOLO_OUTPUT_PATH = '/Users/kyuhyeonkim/test_local/service/yolo_bird/datasets/bird_v2'  # YOLO 포맷으로 변환된 데이터가 저장될 경로

# 디렉토리 생성
os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'val', 'labels'), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'test', 'images'), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'test', 'labels'), exist_ok=True)

# 파일 로드
def load_file(file_path, delimiter=' '):
    with open(file_path, 'r') as f:
        return [line.strip().split(delimiter) for line in f.readlines()]

# 이미지 정보 로드
images_info = load_file(os.path.join(CUB_DATASET_PATH, 'images.txt'))
image_id_to_path = {int(item[0]): item[1] for item in images_info}

# 클래스 정보 로드
classes_info = load_file(os.path.join(CUB_DATASET_PATH, 'classes.txt'))
class_id_to_name = {int(item[0]): item[1].split('.')[1] for item in classes_info}  # 넘버링과 '.' 제거
# class_id_to_name = {int(item[0]): item[1] for item in classes_info}
class_names = [class_id_to_name[i] for i in range(1, len(class_id_to_name) + 1)]

# 이미지 클래스 라벨 로드
image_labels = load_file(os.path.join(CUB_DATASET_PATH, 'image_class_labels.txt'))
image_id_to_class = {int(item[0]): int(item[1]) - 1 for item in image_labels}  # YOLO는 0부터 시작하므로 -1 해줍니다

# 바운딩 박스 정보 로드
bbox_info = load_file(os.path.join(CUB_DATASET_PATH, 'bounding_boxes.txt'))
image_id_to_bbox = {int(item[0]): (float(item[1]), float(item[2]), float(item[3]), float(item[4])) for item in bbox_info}

# 학습/테스트 분할 정보 로드
split_info = load_file(os.path.join(CUB_DATASET_PATH, 'train_test_split.txt'))
image_id_to_is_train = {int(item[0]): int(item[1]) == 1 for item in split_info}

# YOLO 형식으로 바운딩 박스 변환 (x_center, y_center, width, height를 정규화된 값으로)
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x, y, width, height = bbox
    
    # 정규화된 값으로 변환 (0~1 사이 값)
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width = width / img_width
    height = height / img_height
    
    # 값이 0~1 범위 내에 있도록 보장
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

# 검증(validation) 데이터 비율 설정
VAL_RATIO = 0.2

# 데이터 나누기: train, val, test
train_ids = []
val_ids = []
test_ids = []

for image_id, is_train in image_id_to_is_train.items():
    if is_train:
        if np.random.rand() < VAL_RATIO:
            val_ids.append(image_id)
        else:
            train_ids.append(image_id)
    else:
        test_ids.append(image_id)

print(f"Train samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Test samples: {len(test_ids)}")

# 이미지 파일 이동 및 라벨 생성 함수
def process_images(image_ids, subset):
    for image_id in image_ids:
        # 이미지 정보 가져오기
        image_path = os.path.join(CUB_DATASET_PATH, 'images', image_id_to_path[image_id])
        class_id = image_id_to_class[image_id]
        bbox = image_id_to_bbox[image_id]
        
        # 이미지 크기 가져오기 (라이브러리 PIL을 사용)
        from PIL import Image
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # YOLO 형식으로 바운딩 박스 변환
        x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
        
        # 대상 경로 설정
        dest_img_dir = os.path.join(YOLO_OUTPUT_PATH, subset, 'images')
        dest_label_dir = os.path.join(YOLO_OUTPUT_PATH, subset, 'labels')
        
        # 파일명 추출
        img_filename = os.path.basename(image_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        
        # 이미지 파일 복사
        shutil.copy(image_path, os.path.join(dest_img_dir, img_filename))
        
        # 라벨 파일 생성
        with open(os.path.join(dest_label_dir, label_filename), 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 데이터 처리
print("Processing training images...")
process_images(train_ids, 'train')

print("Processing validation images...")
process_images(val_ids, 'val')

print("Processing test images...")
process_images(test_ids, 'test')

# YOLO 데이터셋 설정 파일 생성 (bird_v2.yaml)
with open(os.path.join(YOLO_OUTPUT_PATH, 'bird_v2.yaml'), 'w') as f:
    f.write(f"# YOLOv11n Bird Dataset\n")
    f.write(f"path: {YOLO_OUTPUT_PATH}\n")
    f.write(f"train: train/images\n")
    f.write(f"val: val/images\n")
    f.write(f"test: test/images\n\n")
    f.write(f"# Classes\n")
    f.write(f"names:\n")
    for i, name in enumerate(class_names):
        f.write(f"  {i}: {name}\n")  # name에는 넘버링이 제거된 이름이 들어갑니다.

print("Conversion completed successfully!")