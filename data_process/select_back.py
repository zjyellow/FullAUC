import random

def select_additional_numbers(selected, total_range=30, new_selection_size=10):
    """
    从剩余的数中随机选择指定数量的新数。
    
    参数:
    selected (list): 已经被选中的数列表。
    total_range (int): 总数范围（默认为45，即0-44）。
    new_selection_size (int): 需要新选择的数的数量（默认为15）。
    
    返回:
    list: 新选择的数的列表。
    """
    # 确保输入是有效的
    if len(selected) + new_selection_size > total_range:
        raise ValueError("Selected numbers and new selection size exceed the total range.")
    
    # 创建一个全集
    full_set = set(range(total_range))
    
    # 计算剩余未选中的数
    remaining_numbers = list(full_set - set(selected))
    
    # 从剩余的数中随机选择新的数
    new_selected = random.sample(remaining_numbers, new_selection_size)
    
    return new_selected

# 示例用法
if __name__ == "__main__":
    # 假设这是已经选出的15个数
    already_selected = [29, 18, 6, 17, 23, 0, 21, 12, 4, 15]
    
    # 调用函数并获取新选中的15个数
    newly_selected = select_additional_numbers(already_selected)
    
    print("Newly selected numbers:", newly_selected)