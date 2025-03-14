import os
import random
import shutil
from pathlib import Path

def select_and_copy_images(src_dir, dst_dir, num_samples=2000):
    # 确保目标目录存在
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    # 遍历源目录中的每个子目录（即每个类别）
    for category in os.listdir(src_dir):
        src_category_path = os.path.join(src_dir, category)
        
        # 只处理是目录的情况
        if not os.path.isdir(src_category_path):
            continue
        
        # 创建目标类别目录
        dst_category_path = os.path.join(dst_dir, category)
        Path(dst_category_path).mkdir(parents=True, exist_ok=True)
        
        # 获取该类别的所有图片文件路径列表
        image_files = [f for f in os.listdir(src_category_path) if os.path.isfile(os.path.join(src_category_path, f))]
        
        # 如果该类别下的图片数量少于需要的数量，则全部复制
        if len(image_files) <= num_samples:
            print(f"Category {category} has less than {num_samples} images. Copying all images.")
            selected_images = image_files
        else:
            # 从该类别中随机选择指定数量的图片
            selected_images = random.sample(image_files, num_samples)
        
        # 复制选中的图片到目标目录
        for img in selected_images:
            src_img_path = os.path.join(src_category_path, img)
            dst_img_path = os.path.join(dst_category_path, img)
            shutil.copy(src_img_path, dst_img_path)
        
        print(f"Copied {len(selected_images)} images from category {category}")

if __name__ == "__main__":
    # 定义源和目标目录路径
    source_dataset_dir = '/home/zl/openset/AUC/data/AID'  # 替换为你的源数据集路径
    destination_dataset_dir = '/home/zl/openset/AUC/data/AID_new'  # 替换为目标数据集路径
    
    # 调用函数执行操作
    select_and_copy_images(source_dataset_dir, destination_dataset_dir, num_samples=220)