import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm  # 导入进度条模块

# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def run_model(model, image_path):
    # 使用YOLO模型进行检测
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取检测结果
    results = model(image_rgb)[0]

    # 转换为supervision库的Detections格式
    detections = sv.Detections.from_ultralytics(results)
    return detections, image


def zoom_in_on_detection(image, bbox, zoom_factor=2):
    # 根据检测框放大图像
    x, y, w, h = bbox
    # Cast to integers before slicing
    x, y, w, h = map(int, [x, y, w, h])  # Ensure these values are integers
    crop = image[y:y + h, x:x + w]
    zoomed_crop = cv2.resize(crop, (w * zoom_factor, h * zoom_factor))
    return zoomed_crop


def plot_detection_results(
        image_path,
        n,
        output_folder="output",
        model=None,
):
    # 获取文件夹中的所有图像
    images = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 随机选择 n 张图片
    random_images = random.sample(images, n)

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 使用 tqdm 显示进度条
    for img_name in tqdm(random_images, desc="处理图片", unit="张图片"):
        img_path = os.path.join(image_path, img_name)

        # 使用模型进行检测
        detections, model_image = run_model(model, img_path)

        # 使用supervision库的BoxAnnotator和LabelAnnotator进行注解
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # 创建子图
        fig, ax = plt.subplots(figsize=(10, 10))

        # 原图
        ax.imshow(cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"模型检测结果: {img_name}")  # 这里的中文应该能正确显示
        ax.axis("off")

        # 注解图像
        model_image = box_annotator.annotate(scene=model_image, detections=detections)
        model_image = label_annotator.annotate(scene=model_image, detections=detections,
                                               labels=[f"conf: {conf:.2f}" for conf in detections.confidence])

        # 放大展示自定义模型检测区域（右上角）
        zoomed_images = []
        for bbox in detections.xyxy:
            zoomed_images.append(zoom_in_on_detection(model_image, bbox))

        # 将放大图像放在右上角
        zoomed_width = 100  # 调整放大图像的大小
        zoomed_height = 100
        for zoomed in zoomed_images:
            # 调整放大图像的大小
            zoomed_resized = cv2.resize(zoomed, (zoomed_width, zoomed_height))
            # 放到右上角
            y_offset = 0
            x_offset = model_image.shape[1] - zoomed_resized.shape[1]
            model_image[y_offset:y_offset + zoomed_resized.shape[0], x_offset:x_offset + zoomed_resized.shape[1]] = zoomed_resized

        # 保存或显示比较图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"detection_{img_name}"))
        plt.close(fig)


if __name__ == "__main__":
    # 加载YOLO模型（只初始化一次）
    yolo_model_path = '../yolov8n.pt'  # 替换为你的YOLO模型路径
    yolo_model = YOLO(yolo_model_path)

    # 示例使用
    plot_detection_results(
        image_path="../datasets/tt100k_yolo/tt100k_yolo/images/",
        n=5,
        output_folder="output",
        model=yolo_model
    )
