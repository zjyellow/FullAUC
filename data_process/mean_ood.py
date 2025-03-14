import pandas as pd

def print_metric(df):
    TNR_max = df.loc[0, "TNR_max"]
    AUROC_max = df.loc[0, "AUROC_max"]
    DTACC_max = df.loc[0, "DTACC_max"]
    AUIN_max = df.loc[0, "AUIN_max"]
    AUOUT_max = df.loc[0, "AUOUT_max"]
    # ACC_max = df.loc[0, "ACC_max"]
    # OSCR_max = df.loc[0, "OSCR_max"]
    print(f"TNR_max: {TNR_max:.2f}")
    print(f"AUROC_max: {AUROC_max:.2f}")
    print(f"DTACC_max: {DTACC_max:.2f}")
    print(f"AUIN_max: {AUIN_max:.2f}")
    print(f"AUOUT_max: {AUOUT_max:.2f}")
    # print(f"ACC_max: {ACC_max}")
    # print(f"OSCR_max: {OSCR_max}")

# losses = ['CACLoss', 'PTLLoss', 'OVRN', 'MDL4OW']
losses = ['PTLLoss', ]
dir_path = '/home/zijunhuang/AUC/log/OOD/'
for loss in losses:
    # df_EuroSAT = dir_path + loss + '/' + 'ood-_nwpu_ood_EuroSAT_ood.csv'
    # df_siri = dir_path + loss + '/' + 'ood-_RSSCN7_ood_siri_ood.csv'
    df_EuroSAT = dir_path + loss + '/' + 'ood-_cifar10_cifar100.csv'
    df_siri = dir_path + loss + '/' + 'ood-_cifar10_svhn.csv'
    print('-------------------- loss: {} ------------------------'.format(loss))

    print('---in : nwpu --- background: \\ --- out: EuroSAT ---')
    df_EuroSAT = pd.read_csv(df_EuroSAT)
    print_metric(df_EuroSAT)

    print('---in : RSSCN7 --- background: \\ --- out: siri ---')
    df_siri = pd.read_csv(df_siri)
    print_metric(df_siri)
    print('----------------------------------------------------------')