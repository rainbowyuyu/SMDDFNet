import os

input_train_path = "../../datasets/FullIJCNN2013/gt.txt"
output_train_path = "../../datasets/FullIJCNN2013_yolo"
orign_w = 1360
orign_h = 800


def get_label(label):
    prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
    danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    if label in prohibitory:
        return "0"
    elif label in mandatory:
        return "1"
    elif label in danger:
        return "2"
    else:
        return "3"


def gt2yolo():
    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)

    file_cache = {}

    with open(input_train_path, "r") as f:
        for line in f:
            words = line.strip().split(';')
            img_name = words[0].split(".")[0]
            label = (get_label(int(words[-1])))

            # 原始坐标
            x_min = float(words[1])
            y_min = float(words[2])
            x_max = float(words[3])
            y_max = float(words[4])

            # 中心点与宽高（归一化）
            x_center = ((x_min + x_max) / 2) / orign_w
            y_center = ((y_min + y_max) / 2) / orign_h
            w = (x_max - x_min) / orign_w
            h = (y_max - y_min) / orign_h

            yolo_line = f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
            label_path = os.path.join(output_train_path, f"{img_name}.txt")

            if label_path not in file_cache:
                file_cache[label_path] = []
            file_cache[label_path].append(yolo_line)

    # 写入所有文件
    for path, lines in file_cache.items():
        with open(path, 'w') as fw:
            fw.writelines(lines)


if __name__ == '__main__':
    gt2yolo()
    print("Transform finished ✅")
