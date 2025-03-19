from PIL import Image
import os
import shutil
import random

def resize_and_save(src_img, dest_dir, target_size=600):
    """이미지 리사이징 및 저장 (클래스명 보존)"""
    try:
        # 클래스명 추출 (raw_data/images/<클래스명>/...)
        class_name = src_img.split(os.sep)[3] 
        base_name = os.path.basename(src_img)
        new_name = f"{class_name}_{base_name.split('.')[0]}.jpg"
        dest_path = os.path.join(dest_dir, new_name)

        if os.path.exists(dest_path):
            return dest_path  # 이미 처리된 파일

        img = Image.open(src_img)
        img.thumbnail((target_size, target_size))
        img.convert("RGB").save(dest_path, "JPEG")
        return dest_path
    except Exception as e:
        print(f"❌ {src_img} 처리 실패: {str(e)}")
        return None

# 데이터 분할 비율
SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test

# 메인 처리 파이프라인
for entry in os.scandir("raw_data/images"):
    # 1. 디렉토리 및 숨김 파일 필터링
    if not entry.is_dir() or entry.name.startswith('.'):
        continue
        
    class_dir = entry.name
    class_images = []
    class_labels = []
    
    # 2. 클래스별 데이터 수집
    for img_entry in os.scandir(entry.path):
        # 숨김 파일 및 확장자 필터링
        if (img_entry.name.startswith('.') or 
            not img_entry.name.lower().endswith(('.jpg', '.jpeg'))):
            continue
            
        base_name = os.path.splitext(img_entry.name)[0]
        label_path = os.path.join("raw_data/labels", class_dir, f"{base_name}.txt")
        
        # 라벨 존재 여부 확인
        if os.path.exists(label_path):
            class_images.append(img_entry.path)
            class_labels.append(label_path)
        else:
            print(f"⚠️ 라벨 누락: {class_dir}/{img_entry.name}")

    # 3. 데이터 분할
    combined = list(zip(class_images, class_labels))
    random.shuffle(combined)
    train_idx = int(len(combined)*SPLIT_RATIO[0])
    val_idx = train_idx + int(len(combined)*SPLIT_RATIO[1])

    # 4. 파일 처리
    for idx, (img_path, label_path) in enumerate(combined):
        # 분할 결정
        if idx < train_idx:
            split = 'train'
        elif idx < val_idx:
            split = 'val'
        else:
            split = 'test'
        
        # 이미지 처리
        dest_img = resize_and_save(
            img_path, 
            f"datasets/bird_v1/{split}/images"
        )
        
        # 라벨 복사
        if dest_img:
            dest_label = dest_img.replace('images', 'labels').replace('.jpg', '.txt')
            os.makedirs(os.path.dirname(dest_label), exist_ok=True)
            shutil.copy(label_path, dest_label)
            print(f"✅ {split} 세트에 추가: {os.path.basename(dest_img)}")

print("🎉 모든 데이터 처리 완료!")