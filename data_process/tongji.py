import os

def count_images(directory, extensions=('.jpg', '.jpeg', '.png', '.gif')):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count

base_dir = '/home/zl/openset/AUC/data/AID'  # 将'.'替换为你要检查的目录路径
for subdir in next(os.walk(base_dir))[1]:
    print(f"{subdir}: {count_images(os.path.join(base_dir, subdir))}")