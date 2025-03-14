import torch
import torchvision
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
from datasets.osr_dataloader_generalized import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR, NWPU_OSR, RSSCN7_OSR, siri_OSR, AID_OSR, EuroSAT_OSR
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR, NWPU_OSR, RSSCN7_OSR, siri_OSR, AID_OSR, EuroSAT_OSR
from utils import Logger, save_networks, load_networks
from models.models import classifier32, classifier32ABN
import importlib
import torch.nn as nn
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.keys())
color_style = [mcolors.TABLEAU_COLORS[colors[0]], mcolors.TABLEAU_COLORS[colors[1]],mcolors.TABLEAU_COLORS[colors[2]],mcolors.TABLEAU_COLORS[colors[3]],mcolors.TABLEAU_COLORS[colors[4]]]

def plot_curve(features,labels,colors):
    # Normalize
    x_min = features.min(0)
    x_max = features.max(0)
    features = (features - x_min) / (x_max - x_min)
    for i in range(len(options['known']) + 1):
        indices = np.where(labels == i)
        plt.scatter(features[indices, 0], features[indices, 1], c=colors[i], label=str(i), s=5)

    plt.legend()
    plt.show()

def extract_features(dataloader, net):
    features = []
    labels = []
    net.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.cuda()
            feat, _ = net(data, True)
            # _, feat = net(data, True)
            features.append(feat.cpu())
            if label is not None:
                labels.append(label.cpu())
    
    features = torch.cat(features, dim=0).numpy()
    if len(labels) > 0:
        labels = torch.cat(labels, dim=0).numpy()
        return features, labels
    return features, None

# 如果可能的话，固定批次归一化层的行为
def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            


options = {}
options['dataset'] = 'cifar100'
options['known'] = [1, 0, 9, 8]
options['unknown'] = [33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
options['dataroot'] = './data'
options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
options['batch_size'] = 1
# options['img_size'] = 32
options['img_size'] = 64
options['temp'] = 1.0
options['weight_pl'] = 0.1 # for gcpl
options['feat_dim'] = 128
options['use_gpu'] = True
options['num_centers'] = 1
if 'mnist' in options['dataset']:
    Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
elif 'cifar10' == options['dataset']:
    Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
elif 'svhn' in options['dataset']:
    Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
elif 'cifar100' in options['dataset']:
    Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader = Data.train_loader, Data.test_loader
    out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    outloader = out_Data.test_loader
elif 'tiny_imagenet' in options['dataset']:
    Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
elif 'nwpu' in options['dataset']:
    Data = NWPU_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader 
elif 'RSSCN7' in options['dataset']:
    Data = RSSCN7_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader 
elif 'siri' in options['dataset']:
    Data = siri_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader 
elif 'AID' in options['dataset']:
    Data = AID_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader 
elif 'EuroSAT' in options['dataset']:
    Data = EuroSAT_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader 

# selfauc
path_criterion = '/home/zijunhuang/AUC/log/log/models/FAUCLoss/g-osr/cifar100/checkpoints/classifier32_FAUCLoss_50_0_False__criterion_epoch=598.pth'
path_model = '/home/zijunhuang/AUC/log/log/models/FAUCLoss/g-osr/cifar100/checkpoints/classifier32_FAUCLoss_50_0_False__epoch=598.pth'
options['loss'] = 'FAUCLoss' # MyLoss_new | FAUCLoss2
# path_criterion = '/home/zijunhuang/AUC/log/log/models/MyLoss_new/osr/cifar100_10/checkpoints/classifier32_MyLoss_new_50_0_False__criterion_epoch=0.pth'
# path_model = '/home/zijunhuang/AUC/log/log/models/MyLoss_new/osr/cifar100_10/checkpoints/classifier32_MyLoss_new_50_0_False__epoch=0.pth'
# options['loss'] = 'MyLoss_new' # MyLoss_new | FAUCLoss2
# path_criterion = '/home/zijunhuang/AUC/log/log/models/Softmax/osr/cifar100_10/checkpoints/classifier32_Softmax_10_0_False__criterion_epoch=0.pth'
# path_model = '/home/zijunhuang/AUC/log/log/models/Softmax/osr/cifar100_10/checkpoints/classifier32_Softmax_10_0_False__epoch=0.pth'
# options['loss'] = 'Softmax' # MyLoss_new | FAUCLoss2

options['gama'] = 0.3
options['num_classes'] = Data.num_classes
net = classifier32(num_classes=options['num_classes'])
net = nn.DataParallel(net).cuda()
Loss = importlib.import_module('loss.' + options['loss'])
criterion = getattr(Loss, options['loss'])(**options)
criterion = criterion.cuda()


net.load_state_dict(torch.load(path_model))
criterion.load_state_dict(torch.load(path_criterion))
set_bn_eval(net)
net.eval()  # 确保在评估模式
torch.manual_seed(0)  # 固定随机种子

# 然后使用这个函数
# closed_feature, closed_label = extract_features(testloader, net)
closed_feature, closed_label = extract_features(trainloader, net)
closed_feature = closed_feature[:400]
closed_label = closed_label[:400]
# print(len(closed_feature))
# print(len(closed_feature[0]))
# print(closed_feature)
# print(closed_label)

open_feature, _ = extract_features(outloader, net)
open_feature = open_feature[:100]  # 限制数量

# from sklearn.decomposition import PCA

# pca = PCA(n_components=50, random_state=42)
# closed_feature = pca.fit_transform(closed_feature)  # features.shape = (4000, 128)
# open_feature = pca.fit_transform(open_feature)


# 对所有运行使用相同的t-SNE参数
tsne_params = {
    'n_components': 2,
    # 'perplexity': 30,  # 尝试调整此值，通常为sqrt(n_samples)
    # 'learning_rate': 600,
    # 'n_iter': 5000,  # 增加迭代次数
    'random_state': 42,  # 固定随机种子
    # 'init':'random'
}

tsne = TSNE(**tsne_params)


# closed_feature = []
# closed_label = []
# open_feature = []
# open_label = []

# with torch.no_grad():
#     # For close-set
#     for data, labels in testloader:
#         data, labels = data.cuda(), labels.cuda()
#         with torch.set_grad_enabled(False):
#             x, y = net(data, True)
#         # closed_feature.append(x.cpu())
#         closed_feature.append(y.cpu())
#         closed_label.append(labels.cpu())


#     # For open-set
#     for batch_idx, (data, labels) in enumerate(outloader):
#         data = data.cuda()
#         with torch.set_grad_enabled(False):
#             x, y = net(data, True)
#         open_feature.append(y.cpu())
        # open_feature.append(x.cpu())

# closed_feature = torch.cat(closed_feature, dim=0).numpy()
# closed_label = torch.cat(closed_label, dim=0).numpy()
# open_feature = torch.cat(open_feature, dim=0).numpy()[:1000]
open_label   = np.full((len(open_feature),), len(options['known']))
all_features = np.concatenate((closed_feature, open_feature), axis=0)
all_labels = np.concatenate((closed_label, open_label), axis=0)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# closed_feature = scaler.fit_transform(closed_feature)
# all_features = scaler.fit_transform(all_features)

# print(len(closed_label))
# print(len(open_feature))

# paint_close
# tsne = TSNE(n_components=2, random_state=0)
closed_features_tsne = tsne.fit_transform(closed_feature)


# from sklearn.preprocessing import StandardScaler

# 针对封闭集特征
# scaler_closed = StandardScaler()
# closed_feature_scaled = scaler_closed.fit_transform(closed_feature)
# closed_features_tsne = tsne.fit_transform(closed_feature_scaled)





plt.figure()
# plt.figure(figsize=(10, 10))
colors = ['red', 'green', 'blue', 'purple', 'black', 'yellow',  'orange',   'pink', 'brown', 'gray', 'olive']

# Normalize
x_min = closed_features_tsne.min(0)
x_max = closed_features_tsne.max(0)
closed_features_tsne = (closed_features_tsne - x_min) / (x_max - x_min)
for i in range(len(options['known'])):
    indices = np.where(closed_label == i)
    plt.scatter(closed_features_tsne[indices, 0], closed_features_tsne[indices, 1], c=colors[i], label=str(i), s=5)

plt.legend(loc='upper right')

# paint open
# tsne = TSNE(n_components=2, random_state=0)
all_features_tsne = tsne.fit_transform(all_features)

plt.figure()
# plt.figure(figsize=(10, 10))
# 针对所有特征
# scaler_all = StandardScaler()
# all_features_scaled = scaler_all.fit_transform(all_features)
# all_features_tsne = tsne.fit_transform(all_features_scaled)

# Normalize
x_min = all_features_tsne.min(0)
x_max = all_features_tsne.max(0)
all_features_tsne = (all_features_tsne - x_min) / (x_max - x_min)
for i in range(len(options['known']) + 1):
    indices = np.where(all_labels == i)
    plt.scatter(all_features_tsne[indices, 0], all_features_tsne[indices, 1], c=colors[i], label=str(i), s=5)

plt.legend()
plt.show()

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# closed_feature_pca = pca.fit_transform(closed_feature)
# all_features_pca = pca.fit_transform(all_features)
# plot_curve(closed_feature_pca,closed_label,colors)
# plot_curve(all_features_pca,all_labels,colors)

