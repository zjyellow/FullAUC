import pandas as pd

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



for dataset in ['nwpu','EuroSAT','RSSCN7','siri','AID']:
    df00 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.0'+ '2.csv')
    df005= pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.05'+ '2.csv')
    df01 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.1'+ '2.csv')
    df02 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.2'+ '2.csv')
    df03 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.3'+ '2.csv')
    df04 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.4'+ '2.csv')
    df05 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.5'+ '2.csv')
    df06 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.6'+ '2.csv')
    df07 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.7'+ '2.csv')
    df08 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.8'+ '2.csv')
    df09 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '0.9'+ '2.csv')
    df10 = pd.read_csv('/home/zijunhuang/AUC/log/results/classifier32_MyLoss_new/osr_' + dataset +'_' + '1.0'+ '2.csv')

    print('=========dataset: {}============'.format(dataset))
    print('===0.0===')
    print_metrics(df00)
    print('===0.05===')
    print_metrics(df005)
    print('===0.1===')
    print_metrics(df01)
    print('===0.2===')
    print_metrics(df02)
    print('===0.3===')
    print_metrics(df03)
    print('===0.4===')
    print_metrics(df04)
    print('===0.5===')
    print_metrics(df05)
    print('===0.6===')
    print_metrics(df06)
    print('===0.7===')
    print_metrics(df07)
    print('===0.8===')
    print_metrics(df08)
    print('===0.9===')
    print_metrics(df09)
    print('===1.0===')
    print_metrics(df10)
    
