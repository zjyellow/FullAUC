import os

def get_folder_names_os(dir_path):
    try:
        # os.listdir() 返回指定路径下的所有文件和文件夹名称的列表
        # 然后我们遍历这个列表，使用 os.path.isdir 来检查每个条目是否为文件夹
        return [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    except FileNotFoundError:
        print(f"The directory {dir_path} does not exist.")
        return []

# 示例用法
directory_path = '/home/zl/openset/AUC/data/nwpu_ood/test'  # 替换为你的目标目录路径
folders_list = get_folder_names_os(directory_path)
print(folders_list)