import os
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import random

# 定义输入和输出路径
input_dir = '/home/zl/openset/AUC/data/nwpu_au_all/val'  # 替换为你的数据集路径
output_dir = '/home/zl/openset/AUC/data/nwpu_au_all/val_au'

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义增强序列
seq = iaa.Sequential([
    iaa.SomeOf((1, None), [
        iaa.Fliplr(0.5),  # 水平翻转
        iaa.Affine(
            rotate=(-45, 45),  # 随机旋转
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 随机缩放
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}  # 随机平移
        ),
        iaa.Multiply((0.8, 1.2)),  # 改变亮度
        iaa.GaussianBlur(sigma=(0, 1.0)),  # 高斯模糊
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # 添加高斯噪声
        iaa.Crop(percent=(0, 0.1)),  # 随机裁剪
        iaa.LinearContrast((0.75, 1.5)),  # 线性对比度调整
    ]),
], random_order=True)  # 随机顺序应用增强

# 处理每个类别的文件夹
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    output_class_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    
    images = []
    original_images_paths = []

    # 加载原始图片并记录路径
    for image_file in os.listdir(class_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(class_path, image_file)
            original_images_paths.append(image_path)
            image = Image.open(image_path).convert('RGB')  # 确保所有图片都是RGB模式
            images.append(np.array(image))
    
    num_original_images = len(images)
    target_num_images = 1000
    batch_size = 32  # 可根据需要调整

    # 先复制原始图片到输出目录
    for idx, original_image_path in enumerate(original_images_paths):
        output_image_path = os.path.join(output_class_dir, f'{class_name}_{idx+1:06d}.jpg')
        os.system(f'cp "{original_image_path}" "{output_image_path}"')

    # 计算还需要多少张增强图片
    num_augmented_needed = target_num_images - num_original_images

    for i in range(0, num_augmented_needed, batch_size):
        # 计算当前批次大小
        current_batch_size = min(batch_size, num_augmented_needed - i)
        
        # 从原始图片中随机选取一批进行增强
        indices = [random.randint(0, num_original_images - 1) for _ in range(current_batch_size)]
        batch_images = [images[idx] for idx in indices]
        
        # 应用增强
        augmented_images = seq(images=batch_images)
        
        # 保存增强后的图片
        for j, aug_image in enumerate(augmented_images):
            output_image_path = os.path.join(output_class_dir, f'{class_name}_{num_original_images + i + j + 1:06d}.jpg')
            aug_image_pil = Image.fromarray(aug_image)
            aug_image_pil.save(output_image_path)

print("图像增强完成")