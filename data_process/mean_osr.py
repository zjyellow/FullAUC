import pandas as pd

df = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MDL4OW/osr_nwpu_0.22.csv')  # OVRN | PTLLoss | MDL4OW
df2 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MDL4OW/osr_RSSCN72.csv')
df3 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MDL4OW/osr_EuroSAT2.csv')
df4 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MDL4OW/osr_siri2.csv')
df5 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MDL4OW/osr_AID2.csv')

# df6 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_PTLLoss/cifar100_10_osr.csv')
# df7 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_PTLLoss/cifar100_50_osr.csv')
# df8 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_PTLLoss/osr_mnist.csv')
# df9 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_PTLLoss/osr_tiny_imagenet.csv')
# df10 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_PTLLoss/osr_svhn.csv')
# df11 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_PTLLoss/osr_cifar10.csv')
# print(df)

def mean_std_format(values):
    """计算均值±标准差并格式化输出"""
    mean = values.mean()
    std = values.std()
    return f"{mean:.2f} ± {std:.2f}"

def print_metrics(df):
    # 计算 ACC
    acc_values = df.loc[5][1:].astype(float)
    print(df.loc[5][0], mean_std_format(acc_values))

    # 计算 AUROC
    auroc_values = df.loc[1][1:].astype(float)
    print(df.loc[1][0], mean_std_format(auroc_values))

    # 计算 OSCR
    oscr_values = df.loc[6][1:].astype(float)
    print(df.loc[6][0], mean_std_format(oscr_values))

print("nwpu:")
print_metrics(df)

print("RSSCN7:")
print_metrics(df2)

print("EuroSAT:")
print_metrics(df3)

print("siri:")
print_metrics(df4)

print("AID:")
print_metrics(df5)

print("AID:")
print_metrics(df5)
    
# print("CIFAR+10:")
# print_metrics(df6)

# print("CIFAR+50:")
# print_metrics(df7)
    
# print("MNIST")
# print_metrics(df8)
        
# print("TINY-IMAGENET")
# print_metrics(df9)
        
# print("SVHN")
# print_metrics(df10)
        
# print("CIFAR10")
# print_metrics(df11)