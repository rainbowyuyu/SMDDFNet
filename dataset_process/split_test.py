# rainbow_yu MDDFNet.split_test 🐋✨

import os
import shutil
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条


def copy_images_from_txt(txt_file, output_folder):
    # 获取txt文件的目录作为根目录
    base_dir = os.path.dirname(os.path.abspath(txt_file))

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取 txt 文件中的图片路径
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # 逐行处理每个图片路径
    for line in tqdm(lines, desc="正在复制图片", unit="图片"):
        # 获取图片路径并去掉多余的空白符
        img_path = line.strip()

        # 将相对路径转为绝对路径
        full_img_path = os.path.join(base_dir, img_path) if not os.path.isabs(img_path) else img_path

        # 检查图片路径是否有效
        if os.path.exists(full_img_path):
            # 获取图片文件名
            img_name = os.path.basename(full_img_path)

            # 构建新的文件路径
            new_img_path = os.path.join(output_folder, img_name)

            # 复制图片到新的文件夹
            shutil.copy(full_img_path, new_img_path)
        else:
            print(f"文件 {full_img_path} 不存在，跳过该文件。")

if __name__ == '__main__':
    # 示例用法
    txt_file = '../../datasets/tt100k_yolo/tt100k_yolo/test.txt'  # 你提供的 txt 文件路径
    output_folder = 'gstdb_test_images'  # 目标文件夹，存储复制的图片

    # 执行复制操作
    copy_images_from_txt(txt_file, output_folder)
