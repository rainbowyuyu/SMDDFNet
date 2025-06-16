import os
import random
import shutil
from pathlib import Path

# 设置随机种子保证可复现
random.seed(42)

# 原始数据路径
base_dir = Path("../../datasets/FullIJCNN2013_yolo")  # 修改为你images和labels所在的父目录名
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# 目标路径
for split in ['train', 'val', 'test']:
    (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# 获取所有图片文件名（假设为.jpg，YOLO格式）
image_files = list(images_dir.glob("*.jpg"))
random.shuffle(image_files)

# 划分比例
num_total = len(image_files)
num_train = int(0.7 * num_total)
num_val = int(0.2 * num_total)
num_test = num_total - num_train - num_val

splits = {
    "train": image_files[:num_train],
    "val": image_files[num_train:num_train + num_val],
    "test": image_files[num_train + num_val:]
}

# 执行移动
for split_name, files in splits.items():
    for img_path in files:
        label_path = labels_dir / (img_path.stem + ".txt")

        # 移动图片
        shutil.move(str(img_path), str(base_dir / "images" / split_name / img_path.name))

        # 移动标签（确保标签存在）
        if label_path.exists():
            shutil.move(str(label_path), str(base_dir / "labels" / split_name / label_path.name))

print("✅ 数据集划分完成：train/val/test")
