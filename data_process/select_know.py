import random

# 定义一个函数来生成带有重复项的随机列表
def generate_random_lists_with_replacement(times, num_range, list_length):
    for _ in range(times):
        # 从num_range范围内随机选取list_length个整数，允许重复
        random_list = [random.choice(range(num_range)) for _ in range(list_length)]
        print(random_list)

# 调用函数，生成5次，每次从0-29中选10个整数，允许重复
generate_random_lists_with_replacement(5, 30, 10)