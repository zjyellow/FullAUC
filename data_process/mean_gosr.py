# import pandas as pd

# df = pd.read_csv('/home/zl/openset/AUC/log/results/classifier32_GCPLoss/g-osr_nwpu.csv')
# print(df)


# ##ACC_close
# print(df.loc[0][0], df.loc[0][1:].astype(float).mean())
# ##ACC_all
# print(df.loc[1][0], df.loc[1][1:].astype(float).mean())
# ##F1_score
# print(df.loc[2][0], df.loc[2][1:].astype(float).mean())
# ##AUROC
# print(df.loc[3][0], df.loc[3][1:].astype(float).mean())
# ##OSCR
# print(df.loc[4][0], df.loc[4][1:].astype(float).mean())

import pandas as pd

# df = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MDL4OW/g-osr_nwpu2.csv')
df = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss2/g-osr_nwpu8.csv')
df1 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss2/g-osr_RSSCN78.csv')
# df1 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_GPTLLoss/g-osr_RSSCN73.csv')
df2 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss2/g-osr_EuroSAT8.csv')
df3 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss2/g-osr_siri8.csv')
df4 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss2/g-osr_AID8.csv')

# df5 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new2/g-osr_svhn.csv')
# df6 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new2/new2_cifar100_10_g-osr.csv')

# df7 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss/g-osr_svhn.csv')
# df8 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_FAUCLoss/g-osr_cifar100_10_3.csv')

# df9 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_GPTLLoss/g-osr_svhn.csv')
# df10 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_GPTLLoss/g-osr_cifar100_10.csv')

# print(df)

def mean_std_format1(values):
    """计算均值 ± 标准差并格式化输出"""
    mean = values.mean()
    std = values.std()
    return f"{mean:.2f} ± {std:.2f}"

def mean_std_format2(values):
    """计算均值 ± 标准差并格式化输出"""
    mean = values.mean()
    std = values.std()
    return f"{mean:.4f} ± {std:.4f}"

def acc_close(df):
    # 计算 ACC_close
    acc_close_values = df.loc[0][1:].astype(float)
    print(df.loc[0][0], mean_std_format1(acc_close_values))

def acc_all(df):
    # 计算 ACC_all
    acc_all_values = df.loc[1][1:].astype(float)
    print(df.loc[1][0], mean_std_format1(acc_all_values))

def f1_score(df):
    # 计算 F1_score
    f1_values = df.loc[2][1:].astype(float)
    print(df.loc[2][0], mean_std_format2(f1_values))

def auroc(df):
    # 计算 AUROC
    auroc_values = df.loc[3][1:].astype(float)
    print(df.loc[3][0], mean_std_format1(auroc_values))

def oscr(df):
    # 计算 OSCR
    oscr_values = df.loc[4][1:].astype(float)
    print(df.loc[4][0], mean_std_format1(oscr_values))

def calculate_metric(df):
    print("-----------------------------------------")
    acc_close(df)
    acc_all(df)
    f1_score(df)
    auroc(df)
    oscr(df)
    print("-----------------------------------------")

print('------------------nwpu------------------')
calculate_metric(df)
print('------------------RSSCN7------------------')
calculate_metric(df1)
print('------------------EuroSAT------------------')
calculate_metric(df2)
print('------------------siri------------------')
calculate_metric(df3)
print('------------------AID------------------')
calculate_metric(df4)
    
# print('------------------svhn------------------')
# calculate_metric(df5)

# print('------------------cifar+10------------------')
# calculate_metric(df6)

# print('------------------svhn------------------')
# calculate_metric(df7)

# print('------------------cifar+10------------------')
# calculate_metric(df8)

# print('------------------svhn------------------')
# calculate_metric(df9)

# print('------------------cifar+10------------------')
# calculate_metric(df10)