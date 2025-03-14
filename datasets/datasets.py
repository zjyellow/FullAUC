import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST
import numpy as np
from PIL import Image

# from utils import mkdir_if_missing

class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class KMNISTRGB(KMNIST):
    """KMNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'mnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = MNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = MNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

class KMNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'kmnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = KMNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = KMNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

class CIFAR10(object):
    def __init__(self, **options):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar10')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

class CIFAR100(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar100')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 100
        self.trainloader = trainloader
        self.testloader = testloader


class SVHN(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'svhn')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

class RSSCN7_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        
class RSSCN7_ood(object):
    def __init__(self,**options):

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'RSSCN7_ood')
        pin_memory = True if options['use_gpu'] else False

        trainset = RSSCN7_Filter(os.path.join(data_root, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = RSSCN7_Filter(os.path.join(data_root, 'test'), transform)
        print('All Test Data:', len(testset))
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 6
        self.trainloader = train_loader
        self.testloader = test_loader




class siri_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class siri_ood(object):
    def __init__(self,**options):

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'siri_ood')
        pin_memory = True if options['use_gpu'] else False

        trainset = siri_Filter(os.path.join(data_root, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = siri_Filter(os.path.join(data_root, 'test'), transform)
        print('All Test Data:', len(testset))
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        self.num_classes = 9
        self.trainloader = train_loader
        self.testloader = test_loader

class nwpu_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class nwpu_ood(object):
    def __init__(self,**options):

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'nwpu_ood')
        pin_memory = True if options['use_gpu'] else False

        trainset = nwpu_Filter(os.path.join(data_root, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = nwpu_Filter(os.path.join(data_root, 'test'), transform)
        print('All Test Data:', len(testset))
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        self.num_classes = 36
        self.trainloader = train_loader
        self.testloader = test_loader

class EuroSAT_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class EuroSAT_ood(object):
    def __init__(self,**options):

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'EuroSAT_ood')
        pin_memory = True if options['use_gpu'] else False

        trainset = EuroSAT_Filter(os.path.join(data_root, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = EuroSAT_Filter(os.path.join(data_root, 'test'), transform)
        print('All Test Data:', len(testset))
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        self.num_classes = 8
        self.trainloader = train_loader
        self.testloader = test_loader


def get_combined_trainloader(options):
    """
    组合 CIFAR10 训练集和 CIFAR100 测试集，CIFAR100 测试集的标签设为 11
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # CIFAR10 训练集
    data_root_cifar10 = os.path.join(options['dataroot'], 'cifar10')
    cifar10_trainset = torchvision.datasets.CIFAR10(
        root=data_root_cifar10, train=True, download=True, transform=transform_train
    )
    
    # CIFAR100 测试集
    data_root_cifar100 = os.path.join(options['dataroot'], 'cifar100')
    cifar100_testset = torchvision.datasets.CIFAR100(
        root=data_root_cifar100, train=False, download=True, transform=transform_train
    )

    # 修改 CIFAR100 测试集的标签，使其统一为 11
    class CIFAR100Modified(Dataset):
        def __init__(self, dataset, new_label):
            self.dataset = dataset
            self.new_label = new_label

        def __getitem__(self, index):
            img, _ = self.dataset[index]  # 忽略原始标签
            return img, self.new_label  # 统一新标签为 11

        def __len__(self):
            return len(self.dataset)

    cifar100_testset_modified = CIFAR100Modified(cifar100_testset, new_label=10)

    # 组合数据集
    combined_dataset = ConcatDataset([cifar10_trainset, cifar100_testset_modified])

    # 创建 DataLoader
    batch_size = options['batch_size']
    pin_memory = True if options['use_gpu'] else False
    combined_trainloader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True,
        num_workers=options['workers'], pin_memory=pin_memory,
    )

    return combined_trainloader

def get_cifar10loader(options):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    data_root_cifar10 = os.path.join(options['dataroot'], 'cifar10')
    cifar10_trainset = torchvision.datasets.CIFAR10(
        root=data_root_cifar10, train=True, download=True, transform=transform_train
    )

    total_len = len(cifar10_trainset)
    subset_len = int(total_len * 0.1)
    subset_indices = np.random.choice(total_len, subset_len, replace=False)
    subset_trainset = Subset(cifar10_trainset, subset_indices)

    batch_size = options['batch_size']
    pin_memory = True if options['use_gpu'] else False
    cifar10loader = DataLoader(
        subset_trainset, batch_size=batch_size, shuffle=True,
        num_workers=options['workers'], pin_memory=pin_memory,
    )
    return cifar10loader

def get_cifar100loader(options):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    data_root_cifar100 = os.path.join(options['dataroot'], 'cifar100')
    cifar100_testset = torchvision.datasets.CIFAR100(
        root=data_root_cifar100, train=False, download=True, transform=transform_train
    )
    class CIFAR100Modified(Dataset):
        def __init__(self, dataset, new_label):
            self.dataset = dataset
            self.new_label = new_label

        def __getitem__(self, index):
            img, _ = self.dataset[index]  # 忽略原始标签
            return img, self.new_label  # 统一新标签为 11

        def __len__(self):
            return len(self.dataset)
    cifar100_testset_modified = CIFAR100Modified(cifar100_testset, new_label=10)
    batch_size = options['batch_size']
    pin_memory = True if options['use_gpu'] else False
    cifar100loader = DataLoader(
        cifar100_testset_modified, batch_size=batch_size, shuffle=True,
        num_workers=options['workers'], pin_memory=pin_memory,
    )
    return cifar100loader

def get_svhnloader(options):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # SVHN 测试集
    data_root_svhn = os.path.join(options['dataroot'], 'svhn')
    svhn_testset = torchvision.datasets.SVHN(
        root=data_root_svhn, split='test', download=True, transform=transform_train
    )

    # 修改 SVHN 测试集的标签，使其统一为 11
    class SVHNModified(Dataset):
        def __init__(self, dataset, new_label):
            self.dataset = dataset
            self.new_label = new_label

        def __getitem__(self, index):
            img, _ = self.dataset[index]  # 忽略原始标签
            return img, self.new_label  # 统一新标签为 11

        def __len__(self):
            return len(self.dataset)

    svhn_testset_modified = SVHNModified(svhn_testset, new_label=10)
    batch_size = options['batch_size']
    pin_memory = True if options['use_gpu'] else False
    svhnloader = DataLoader(
        svhn_testset_modified, batch_size=batch_size, shuffle=True,
        num_workers=options['workers'], pin_memory=pin_memory,
    )
    return svhnloader


def get_combined_trainloader2(options):
    """
    组合 CIFAR10 训练集和 SVHN 测试集，SVHN 测试集的标签设为 10 (N+1)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # CIFAR10 训练集
    data_root_cifar10 = os.path.join(options['dataroot'], 'cifar10')
    cifar10_trainset = torchvision.datasets.CIFAR10(
        root=data_root_cifar10, train=True, download=True, transform=transform_train
    )
    
    # SVHN 测试集
    data_root_svhn = os.path.join(options['dataroot'], 'svhn')
    svhn_testset = torchvision.datasets.SVHN(
        root=data_root_svhn, split='test', download=True, transform=transform_train
    )

    # 修改 SVHN 测试集的标签，使其统一为 11
    class SVHNModified(Dataset):
        def __init__(self, dataset, new_label):
            self.dataset = dataset
            self.new_label = new_label

        def __getitem__(self, index):
            img, _ = self.dataset[index]  # 忽略原始标签
            return img, self.new_label  # 统一新标签为 11

        def __len__(self):
            return len(self.dataset)

    svhn_testset_modified = SVHNModified(svhn_testset, new_label=10)

    # 组合数据集
    combined_dataset = ConcatDataset([cifar10_trainset, svhn_testset_modified])

    # 创建 DataLoader
    batch_size = options['batch_size']
    pin_memory = True if options['use_gpu'] else False
    combined_trainloader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True,
        num_workers=options['workers'], pin_memory=pin_memory,
    )

    return combined_trainloader

__factory = {
    'mnist': MNIST,
    'kmnist': KMNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn':SVHN,
    'RSSCN7_ood':RSSCN7_ood,
    'siri_ood':siri_ood,
    'nwpu_ood':nwpu_ood,
    'EuroSAT_ood':EuroSAT_ood,
}

def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**options)