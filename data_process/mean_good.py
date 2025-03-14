import pandas as pd

def print_metric(df):
    
    # TNR_max = df.loc[0, "TNR_max"]
    # AUROC_max = df.loc[0, "AUROC_max"]
    # DTACC_max = df.loc[0, "DTACC_max"]
    # AUIN_max = df.loc[0, "AUIN_max"]
    # AUOUT_max = df.loc[0, "AUOUT_max"]

    TNR_max = df.loc[0, "max_TNR"]
    AUROC_max = df.loc[0, "max_AUROC"]
    DTACC_max = df.loc[0, "max_DTACC"]
    AUIN_max = df.loc[0, "max_AUIN"]
    AUOUT_max = df.loc[0, "max_AUOUT"]

    # ACC_max = df.loc[0, "ACC_max"]
    # OSCR_max = df.loc[0, "OSCR_max"]
    print(f"TNR_max: {TNR_max:.2f}")
    print(f"AUROC_max: {AUROC_max:.2f}")
    print(f"DTACC_max: {DTACC_max:.2f}")
    print(f"AUIN_max: {AUIN_max:.2f}")
    print(f"AUOUT_max: {AUOUT_max:.2f}")
    # print(f"ACC_max: {ACC_max}")
    # print(f"OSCR_max: {OSCR_max}")

# losses = ['Softmax', 'GCPLoss', 'RPLoss',  'ARPLoss', 'GCACLoss',]
losses = ['OpenAUCLoss', 'MyLoss_new2',  'PTLLoss']
dir_path = '/home/zijunhuang/AUC/log/G_OOD/'
for loss in losses:
    df_svhn = dir_path + loss + '/' + 'g-ood-_cifar10_cifar100_svhn_svhn.csv'
    df_cifar100 = dir_path + loss + '/' + 'g-ood-_cifar10_svhn_cifar100_cifar100.csv'
    print('-------------------- loss: {} ------------------------'.format(loss))

    print('---in : cifar10 --- background: cifar100 --- out: svhn ---')
    df_svhn = pd.read_csv(df_svhn)
    print_metric(df_svhn)

    print('---in : cifar10 --- background: svhn --- out: cifar100 ---')
    df_cifar100 = pd.read_csv(df_cifar100)
    print_metric(df_cifar100)
    print('----------------------------------------------------------')