# rainbow_yu MDDFNet.re_filename 🐋✨

import os
from pathlib import Path


def remove_annotated_prefix(folder_path):
    """遍历文件夹中的所有文件，将文件名以 'annotated_' 开头的部分去掉"""
    folder = Path(folder_path)

    # 遍历文件夹中的所有文件
    for file in folder.glob("*"):
        # 检查文件名是否以 'annotated_' 开头
        if file.name.startswith('annotated_'):
            # 获取新的文件名，去掉 'annotated_' 部分
            new_name = file.name[len('annotated_'):]

            # 创建新的文件路径
            new_file_path = file.parent / new_name

            # 重命名文件
            file.rename(new_file_path)
            print(f"Renamed: {file.name} -> {new_name}")


# 输入文件夹路径
folder_path = r"E:\python_project\MDDFNet\result\compare_yolo"

# 运行函数
remove_annotated_prefix(folder_path)
