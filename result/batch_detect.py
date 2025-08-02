import os
import cv2
import supervision as sv
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def yolo_detect(input_folder, output_folder, model_path, num_images=None):
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 加载YOLO模型
    model = YOLO(model_path)

    # 初始化supervision的注解器
    bounding_box_annotator = sv.BoxAnnotator()  # 使用 BoxAnnotator
    label_annotator = sv.LabelAnnotator()  # 使用 LabelAnnotator，它会自动显示置信度

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 如果指定了处理图片数量，则只处理前 num_images 张
    if num_images is not None:
        image_files = image_files[:num_images]

    # 循环处理每一张图片
    for image_file in tqdm(image_files, desc="Processing images"):
        # 读取图片
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)

        # YOLO进行检测
        results = model(frame)[0]

        # 检查是否有检测框
        if results.boxes.shape[0] == 0:  # 如果没有检测到任何物体，跳过此图片
            print(f"No detections for {image_file}, skipping...")
            continue

        # 使用supervision库处理检测结果
        detections = sv.Detections.from_ultralytics(results)

        # 绘制标注框
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)

        # 自动标注置信度
        # 自动标注置信度和标签
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=[f"{cls} {conf:.2f}" for cls, conf in
                    zip(detections.data['class_name'], detections.confidence)]
        )

        # 保存带标注的图片
        output_img_path = os.path.join(output_folder, f"{image_file}")
        cv2.imwrite(output_img_path, annotated_image)

        # 保存检测结果到txt文件
        output_txt_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(output_txt_path, 'w') as f:
            for i in range(results.boxes.shape[0]):  # 迭代所有检测框
                box = results.boxes.xywh[i].cpu().numpy()  # 获取检测框
                label = results.names[int(results.boxes.cls[i])]  # 获取标签
                confidence = results.boxes.conf[i].cpu().numpy()  # 获取置信度
                f.write(f"{label} {confidence:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f}\n")

    print(f"Detection completed. Results saved to {output_folder}")


if __name__ == '__main__':
    input_folder = './input_images'  # 输入文件夹，包含待检测的图片
    output_folder = './output_results'  # 输出文件夹，保存检测结果和标注图片
    model_path = r"./best.pt"  # YOLO模型路径

    yolo_detect(input_folder, output_folder, model_path)
