from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
import os


def batch_convert_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.ppm') or filename.endswith('.pgm'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            save_path = os.path.join(output_dir, new_filename)
            img.save(save_path)

if __name__ == '__main__':
    # 使用示例
    input_dir = '../../datasets/FullIJCNN2013'  # 输入图片所在目录
    output_dir = '../../datasets/FullIJCNN2013_yolo'  # 输出图片所在目录
    batch_convert_images(input_dir, output_dir)