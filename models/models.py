import torch
import torch.nn as nn
from torch.nn import functional as F
from models.ABN import MultiBatchNorm

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*4*4, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x, rf=False):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*4*4)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)
        
        if rf:
            return x, y
        return y

class ReconstructiveModule(nn.Module):
    def __init__(self):
        super(ReconstructiveModule, self).__init__()

        # 使用反卷积将128维特征图重建到3x64x64
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1, padding=0, output_padding=0)
        # self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv7 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 添加批归一化层
        self.bn1_de = nn.BatchNorm2d(128)
        self.bn2_de = nn.BatchNorm2d(128)
        self.bn3_de = nn.BatchNorm2d(64)
        self.bn4_de = nn.BatchNorm2d(64)
        self.bn5_de = nn.BatchNorm2d(32)
        self.bn6_de = nn.BatchNorm2d(32)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1) 
        x = self.deconv1(x)
        x = self.bn1_de(x)
        x = nn.ReLU()(x)

        x = self.deconv2(x)
        x = self.bn2_de(x)
        x = nn.ReLU()(x)

        x = self.deconv3(x)
        x = self.bn3_de(x)
        x = nn.ReLU()(x)

        x = self.deconv4(x)
        x = self.bn4_de(x)
        x = nn.ReLU()(x)

        x = self.deconv5(x)
        x = self.bn5_de(x)
        x = nn.ReLU()(x)

        x = self.deconv6(x)
        x = self.bn6_de(x)
        x = nn.ReLU()(x)

        x = self.deconv7(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32(nn.Module):
    def __init__(self, num_classes=10):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        # self.hidden_layers = nn.ModuleList([nn.Linear(num_classes, num_classes) for _ in range(num_classes)])
        # self.classifiers = nn.ModuleList([nn.Linear(128, 1) for _ in range(num_classes)])
        # self.reconstructive_module = ReconstructiveModule()

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_feature=False, OVRNetwork=False, return_reconstruction=False):
        ori_x = x
        
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Features (Batchsize, 128)
        y = self.fc(x)          # Logits (B, num_classes)
        if OVRNetwork:
            logits = torch.stack([hidden(y) for hidden in self.hidden_layers], dim=1) # For One-vs-Rest Network(OVRN)
            # logits = torch.cat([clf(x) for clf in self.classifiers], dim=1) # new One-vs-Rest Network(FAUC)
            return y, logits
        elif return_reconstruction:
            re_x = self.reconstructive_module(x)
            return ori_x, re_x, y
        else:
            if return_feature:
                return x, y
            else:
                return y

def weights_init_ABN(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32ABN(nn.Module):
    def __init__(self, num_classes=10, num_ABN=2):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = MultiBatchNorm(64, num_ABN)
        self.bn2 = MultiBatchNorm(64, num_ABN)
        self.bn3 = MultiBatchNorm(128, num_ABN)

        self.bn4 = MultiBatchNorm(128, num_ABN)
        self.bn5 = MultiBatchNorm(128, num_ABN)
        self.bn6 = MultiBatchNorm(128, num_ABN)

        self.bn7 = MultiBatchNorm(128, num_ABN)
        self.bn8 = MultiBatchNorm(128, num_ABN)
        self.bn9 = MultiBatchNorm(128, num_ABN)
        self.bn10 = MultiBatchNorm(128, num_ABN)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init_ABN)
        self.cuda()

    def forward(self, x, return_feature=False, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        x = self.dr1(x)
        x = self.conv1(x)
        x, _ = self.bn1(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x, _ = self.bn2(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x, _ = self.bn3(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x, _ = self.bn4(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x, _ = self.bn5(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x, _ = self.bn6(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x, _ = self.bn7(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x, _ = self.bn8(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x, _ = self.bn9(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if return_feature:
            return x, y
        else:
            return y


