import os
import shutil
import random
from math import floor

# 定义输入和输出路径
input_dir = '/home/zl/openset/AUC/data/AID'  # 替换为你的数据集路径
output_dir = '/home/zl/openset/AUC/data/AID_2'

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_ratio = 8/10  # 训练集占比
val_ratio = 1/10    # 验证集占比
test_ratio = 1/10   # 测试集占比

# 处理每个类别的文件夹
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # 创建输出分类目录
    for subset in ['train', 'val', 'test']:
        subset_output_dir = os.path.join(output_dir, subset, class_name)
        if not os.path.exists(subset_output_dir):
            os.makedirs(subset_output_dir)
    
    # 获取所有图片文件列表
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    num_images = len(images)
    
    # 检查是否有足够的图片进行划分
    if num_images < 7:
        print(f"Warning: Class '{class_name}' has less than 7 images and cannot be split into train/val/test sets.")
        continue
    
    # 随机打乱图片顺序
    random.shuffle(images)
    
    # 根据比例计算各部分的数量
    num_train = floor(num_images * train_ratio)
    num_val = floor(num_images * val_ratio)
    
    # 划分图片到不同的集合
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]
    
    # 将图片复制到相应的输出目录
    def copy_images(image_list, subset):
        subset_output_dir = os.path.join(output_dir, subset, class_name)
        for image in image_list:
            src = os.path.join(class_path, image)
            dst = os.path.join(subset_output_dir, image)
            shutil.copy(src, dst)
    
    copy_images(train_images, 'train')
    copy_images(val_images, 'val')
    copy_images(test_images, 'test')

print("数据集划分完成")