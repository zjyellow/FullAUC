import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, EMNIST
from torch.utils.data import random_split
import random

class EMNISTRGB(EMNIST):
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
class EMNIST_Filter(EMNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        targets = self.targets.data
        mask, new_targets = [], []
        if train:
            for i in range(len(targets)):
                if targets[i] in known and known_need:
                    mask.append(i)
                    new_targets.append(known.index(targets[i])) # 0 to N-1 as known
                elif targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as seen unknown
        else:
            count = known_count
            unknown_count = 0
            for i in range(len(targets)):
                if targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    # print(count, unknown_count)
                    if unknown_count == count:
                        break
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)

    def __split__(self, mode=''):
        targets = self.targets
        # targets = self.targets.data.numpy()
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 0.8)
        valset_size = len(targets) - testset_size
        test_indices = torch.tensor(shuffled_indices[:testset_size])
        val_indices = torch.tensor(shuffled_indices[testset_size:])
        if mode == 'test': # 'train'
            self.data = torch.index_select(self.data, 0, test_indices)
            self.targets = targets[shuffled_indices[:testset_size]]
        elif mode == 'val':
            self.data = torch.index_select(self.data, 0, val_indices)
            self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')




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

class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        targets = self.targets.data
        mask, new_targets = [], []
        if train:
            for i in range(len(targets)):
                if targets[i] in known and known_need:
                    mask.append(i)
                    new_targets.append(known.index(targets[i])) # 0 to N-1 as known
                elif targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as seen unknown
        else:
            count = known_count
            unknown_count = 0
            for i in range(len(targets)):
                if targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    # print(count, unknown_count)
                    if unknown_count == count:
                        break
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)

    def __split__(self, mode=''):
        targets = self.targets
        # targets = self.targets.data.numpy()
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 0.8)
        valset_size = len(targets) - testset_size
        test_indices = torch.tensor(shuffled_indices[:testset_size])
        val_indices = torch.tensor(shuffled_indices[testset_size:])
        if mode == 'test': # 'train'
            self.data = torch.index_select(self.data, 0, test_indices)
            self.targets = targets[shuffled_indices[:testset_size]]
        elif mode == 'val':
            self.data = torch.index_select(self.data, 0, val_indices)
            self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')

class MNIST_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))
        # self.seen_unknown=seen_unknown
        # self.unseen_unknown = list(set(list(range(0, 10))) - set(known)-set(seen_unknown))
        # self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Known Labels: {}, Background class Labels: {}, Unknown Labels: {}'.format(known, self.unknown, self.unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset_mnist = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        val_mnist = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)

        trainset_mnist.__Filter__(known=self.known)
        trainset_mnist.__split__(mode='test')

        val_mnist.__Filter__(known=self.known)
        val_mnist.__split__(mode='val')

        trainset_emnist = EMNIST_Filter(root='./data/emnist', split='letters', train=True, download=False, transform=train_transform)
        trainset_emnist.targets = [len(self.known)] * len(trainset_emnist.targets)
        # trainset_cifar10 = CIFAR10_Filter(root='./data/cifar10', train=True, download=False, transform=train_transform)
        # trainset_cifar10.targets = [len(self.known)] * len(trainset_cifar10.targets)
        # control background samples' quantities = number of samples of single known class
        num_images = 6000 #  (num of mnist trainset) / (num of classes) = 60000 / 10
        # indices = list(range(len(trainset_cifar10)))
        indices = list(range(len(trainset_emnist)))
        random.shuffle(indices)
        selected_indices = indices[:num_images]
        # trainset_cifar10 = torch.utils.data.Subset(trainset_cifar10, selected_indices)
        trainset_emnist = torch.utils.data.Subset(trainset_emnist, selected_indices)

        # train & test
        train_size_emnist = int(0.8 * len(trainset_emnist))
        val_size_emnist = len(trainset_emnist) - train_size_emnist
        train_dataset_emnist, val_dataset_emnist = random_split(trainset_emnist, [train_size_emnist, val_size_emnist])

        # train_size_cifar10 = int(0.8 * len(trainset_cifar10))
        # val_size_cifar10 = len(trainset_cifar10) - train_size_cifar10
        # train_dataset_cifar10, val_dataset_cifar10 = random_split(trainset_cifar10, [train_size_cifar10, val_size_cifar10])
        train_dataset = torch.utils.data.ConcatDataset([trainset_mnist, train_dataset_emnist])

        # train_dataset = torch.utils.data.ConcatDataset([trainset_mnist, train_dataset_cifar10])

        print('All Train Data: MNIST: {} + EMNIST: {} = {}'.format(len(trainset_mnist), len(train_dataset_emnist), len(train_dataset)))
        print('All Val Data: MNIST: {} + EMNIST: {} = {}'.format(len(val_mnist), len(val_dataset_emnist), len(val_mnist) + len(val_dataset_emnist)))

        # print('All Train Data: MNIST: {} + CIFAR10: {} = {}'.format(len(trainset_mnist), len(train_dataset_cifar10), len(train_dataset)))
        # print('All Val Data: MNIST: {} + CIFAR10: {} = {}'.format(len(val_mnist), len(val_dataset_cifar10), len(val_mnist) + len(val_dataset_cifar10)))

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            val_mnist, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            val_dataset_emnist, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        # self.val_open_loader = torch.utils.data.DataLoader(
        #     val_dataset_cifar10, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )

        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)

        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.known, seen_unknown = self.unknown, known_need=False)

        print('All Test Data: Closed: {}, Open-set: {}'.format(len(testset), len(outset)))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known, seen_unknown= [] ,known_need=True, train=True, known_count=0):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        if train:
            for i in range(len(targets)):
                if targets[i] in known and known_need:
                    mask.append(i)
                    new_targets.append(known.index(targets[i]))
                elif targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known))  # N as seen unknown
        else:
            count = known_count
            unknown_count = 0
            for i in range(len(targets)):
                if targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    # print(count, unknown_count)
                    if unknown_count == count:
                        break
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

    def __split__(self, mode=''):
        datas = torch.tensor(self.data)
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 0.8)
        valset_size = len(targets) - testset_size
        test_indices = torch.tensor(shuffled_indices[:testset_size])
        val_indices = torch.tensor(shuffled_indices[testset_size:])
        if mode == 'test':
            self.data = torch.index_select(datas, 0, test_indices).numpy()
            self.targets = targets[shuffled_indices[:testset_size]]
        elif mode == 'val':
            self.data = torch.index_select(datas, 0, val_indices).numpy()
            self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')

class CIFAR10_OSR(object):
    def __init__(self, known,seen_unknown, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown=seen_unknown
        # self.unseen_unknown = list(set(list(range(0, 10))) - set(known)-set(seen_unknown))
        # self.unknown = list(set(list(range(0, 10))) - set(known))
        print('Selected Known Labels: {}, Background class Labels: {}'.format(known, seen_unknown))

        # print('Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {} in Cifar10'.format(known, seen_unknown, self.unseen_unknown))


        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset_cifar10 = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        # print('All Train Data:', len(trainset))
        trainset_cifar10.__Filter__(known=self.known,seen_unknown=self.seen_unknown)

        trainset_cifar10 = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        # val_cifar10 = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)

        trainset_cifar10.__Filter__(known=self.known, seen_unknown=[])
        # trainset_cifar10.__split__(mode='test')

        # val_cifar10.__Filter__(known=self.known, seen_unknown=[])
        # val_cifar10.__split__(mode='val')

        trainset_cifar10_openset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        # val_cifar10_openset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        
        trainset_cifar10_openset.__Filter__(known=self.seen_unknown, seen_unknown=[])
        # trainset_cifar10_openset.__split__(mode='test')
        trainset_cifar10_openset.targets = [len(self.known)] * len(trainset_cifar10_openset.targets)

        num_images = 5000 #  (num of mnist trainset) / (num of classes) = 60000 / 10
        indices = list(range(len(trainset_cifar10_openset)))
        random.shuffle(indices)
        selected_indices_train = indices[:num_images]
        # num_train = int(num_images * 0.8)
        # selected_indices_train = indices[:num_train]
        # selected_indices_val = indices[num_train:num_images]
        trainset_cifar10_openset = torch.utils.data.Subset(trainset_cifar10_openset, selected_indices_train)
        # val_cifar10_openset = torch.utils.data.Subset(trainset_cifar10_openset, selected_indices_val)


        # val_cifar10_openset.__Filter__(known=self.seen_unknown, seen_unknown=[])
        # val_cifar10_openset.__split__(mode='val')
        # val_cifar10_openset.targets = [len(self.known)] * len(val_cifar10_openset.targets)

        train_dataset = torch.utils.data.ConcatDataset([trainset_cifar10, trainset_cifar10_openset])
        print('All Train Data: CIFAR10-CLOSED: {} + OPEN-SET: {} = {}'.format(len(trainset_cifar10), len(trainset_cifar10_openset), len(train_dataset)))
        # print('All Val Data: CIFAR10-CLOSED: {} + OPEN-SET: {} = {}'.format(len(val_cifar10), len(val_cifar10_openset), len(val_cifar10) + len(val_cifar10_openset)))


        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            trainset_cifar10, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            trainset_cifar10_openset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known, seen_unknown=[])
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        # testset_close = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        # testset_open = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        # valset_close = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        # valset_open = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)

        # testset_close.__split__(mode='test')
        # print('All Test Data:', len(testset_close))
        # testset_close.__Filter__(known=self.known)

        # testset_open.__split__(mode='test')
        # testset_open.__Filter__(known=self.known, seen_unknown = self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))


        # valset_close.__split__(mode='val')
        # print('All Dev Data:', len(valset_close))
        # valset_close.__Filter__(known=self.known)

        # valset_open.__split__(mode='val')
        # valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))

        # self.test_loader = torch.utils.data.DataLoader(
        #     testset_close, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )

        # self.out_loader = torch.utils.data.DataLoader(
        #     testset_open, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )

        # self.val_close_loader = torch.utils.data.DataLoader(
        #     valset_close, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )
        # self.val_open_loader = torch.utils.data.DataLoader(
        #     valset_open, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )

        # print('Train data: ', len(trainset))
        # print('Test Close-set: {} Open-set: {}'.format(len(testset_close),len(testset_open)))
        # print('Val  Close-set: {} Open-set: {}'.format(len(valset_close),len(valset_open)))





        # testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        # print('All Test Data:', len(testset))
        # testset.__Filter__(known=self.known)
        #
        # self.test_loader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )
        #
        # outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        # outset.__Filter__(known=self.unknown)
        #
        # self.out_loader = torch.utils.data.DataLoader(
        #     outset, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )
        #
        # print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        # print('All Test: ', (len(testset) + len(outset)))

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        if train:
            for i in range(len(targets)):
                if targets[i] in known and known_need:
                    mask.append(i)
                    new_targets.append(known.index(targets[i]))
                elif targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known))  # N as seen unknown
        else:
            count = known_count
            unknown_count = 0
            for i in range(len(targets)):
                if targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    # print(count, unknown_count)
                    if unknown_count == count:
                        break
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, unseen_unknown, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32, count=0):
        self.num_classes = len(known)
        self.known = known

        print('Unknown Unknown Class indexes in Cifar100: {}'.format(unseen_unknown))

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False
        
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known, seen_unknown=unseen_unknown, known_need=False, train=False, known_count=int(count/len(known)))
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Test CIFAR100 Open-set: {}'.format(len(testset)))

class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known, seen_unknown= [] ,known_need=True, train=True, known_count=0):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        if train:
            for i in range(len(targets)):
                if targets[i] in known and known_need:
                    mask.append(i)
                    new_targets.append(known.index(targets[i]))
                elif targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known))  # N as background-class
        else:
            count = known_count
            unknown_count = 0
            for i in range(len(targets)):
                if targets[i] in seen_unknown:
                    mask.append(i)
                    new_targets.append(len(known)) # N as background-class
                    unknown_count += 1
                    # print(count, unknown_count)
                    if unknown_count == count:
                        break
        self.data, self.labels = np.array(self.data[mask]), np.array(new_targets)

    def __split__(self, mode=''):
        datas = torch.tensor(self.data)
        targets = np.array(self.labels)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 0.8)
        valset_size = len(targets) - testset_size
        test_indices = torch.tensor(shuffled_indices[:testset_size])
        val_indices = torch.tensor(shuffled_indices[testset_size:])
        if mode == 'test':
            self.data = torch.index_select(datas, 0, test_indices).numpy()
            self.labels = targets[shuffled_indices[:testset_size]]
        elif mode == 'val':
            self.data = torch.index_select(datas, 0, val_indices).numpy()
            self.labels = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')

class SVHN_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/svhn', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        # self.num_classes = len(known)
        # self.known = known
        # self.seen_unknown = []
        # self.unknown = list(set(list(range(0, 10))) - set(known))
        # print('Selected Known Labels: {}, Background class Labels: {}, Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                           self.unknown))
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown=seen_unknown
        self.unseen_unknown = list(set(list(range(0, 10))) - set(known)-set(seen_unknown))
        self.unknown = list(set(list(range(0, 10))) - set(known))
        print('Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset_close = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        testset_open = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        valset_close = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        valset_open = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        
        testset_close.__split__(mode='test')
        print('All Test Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)
        
        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))
        
        valset_close.__split__(mode='val')
        print('All Val Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)
        
        valset_open.__split__(mode='val')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))
        
        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))



class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = self.imgs, self.targets
        # print(datas[6][1],targets[6])
        new_datas, new_targets = [], []
        if train:
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in known and known_need:
                    new_item = (datas[i][0], known.index(datas[i][1]))
                    new_datas.append(new_item)
                    new_targets.append(known.index(targets[i]))
                    # new_targets.append(targets[i])
                    # print(i, datas[i][1], targets[i])
                elif datas[i][1] in seen_unknown and unknown_count < known_count:
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known))  # N as seen unknown/unseen unknown
                    unknown_count += 1
        else:
            count = known_count # number of samples on single class
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in seen_unknown: # ('s_unknown' + 'u_unknown')
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    if unknown_count == count:
                        break
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

    def __split__(self, mode=''):
        # a = self.imgs
        # datas = torch.tensor(self.imgs[:][0])
        datas = self.imgs
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 0.5)
        valset_size = len(targets) - testset_size
        test_indices = shuffled_indices[:testset_size]
        val_indices = shuffled_indices[testset_size:]
        # test_indices = torch.tensor(shuffled_indices[:testset_size])
        # val_indices = torch.tensor(shuffled_indices[testset_size:])

        new_datas, new_targets = [], []
        if mode == 'test':
            # self.imgs = torch.index_select(datas, 0, test_indices)
            # self.samples = torch.index_select(datas, 0, test_indices)
            for i in range(len(test_indices)):
                new_item = (datas[test_indices[i]][0], datas[test_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[:testset_size]]
                # self.imgs = datas[test_indices]
                # self.samples = datas[test_indices]
        elif mode == 'val':
            for i in range(len(val_indices)):
                new_item = (datas[val_indices[i]][0], datas[val_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[testset_size:]]
            # self.imgs = torch.index_select(datas, 0, val_indices)
            # self.samples = torch.index_select(datas, 0, val_indices)
            # self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')


class Tiny_ImageNet_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/tiny_imagenet', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown = seen_unknown
        self.unseen_unknown = list(set(list(range(0, 200))) - set(known) - set(seen_unknown))
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print(
            'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown, known_count=500)
        # trainset.__Filter__(known=self.known, seen_unknown=self.unknown)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset_close = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val3'), transform)
        testset_open = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val3'), transform)
        valset_close = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val3'), transform)
        valset_open = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val3'), transform)

        testset_close.__split__(mode='test')
        print('All Test Close Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)

        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        valset_close.__split__(mode='val')
        print('All Val Close Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)

        valset_open.__split__(mode='val')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))




        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))



class RSSCN7_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = self.imgs, self.targets
        # print(datas[6][1],targets[6])
        new_datas, new_targets = [], []
        if train:
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in known and known_need:
                    new_item = (datas[i][0], known.index(datas[i][1]))
                    new_datas.append(new_item)
                    new_targets.append(known.index(targets[i]))
                    # new_targets.append(targets[i])
                    # print(i, datas[i][1], targets[i])
                elif datas[i][1] in seen_unknown and unknown_count < known_count:
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known))  # N as seen unknown/unseen unknown
                    unknown_count += 1
        else:
            count = known_count # number of samples on single class
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in seen_unknown: # ('s_unknown' + 'u_unknown')
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    if unknown_count == count:
                        break
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

    def __split__(self, mode=''):
        # a = self.imgs
        # datas = torch.tensor(self.imgs[:][0])
        datas = self.imgs
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 1)  #0.5
        valset_size = len(targets) - testset_size
        test_indices = shuffled_indices[:testset_size]
        val_indices = shuffled_indices[testset_size:]
        # test_indices = torch.tensor(shuffled_indices[:testset_size])
        # val_indices = torch.tensor(shuffled_indices[testset_size:])

        new_datas, new_targets = [], []
        if mode == 'test':
            # self.imgs = torch.index_select(datas, 0, test_indices)
            # self.samples = torch.index_select(datas, 0, test_indices)
            for i in range(len(test_indices)):
                new_item = (datas[test_indices[i]][0], datas[test_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[:testset_size]]
                # self.imgs = datas[test_indices]
                # self.samples = datas[test_indices]
        elif mode == 'val':
            for i in range(len(val_indices)):
                new_item = (datas[val_indices[i]][0], datas[val_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[testset_size:]]
            # self.imgs = torch.index_select(datas, 0, val_indices)
            # self.samples = torch.index_select(datas, 0, val_indices)
            # self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')


class RSSCN7_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/NWPU_split', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown = seen_unknown
        self.unseen_unknown = list(set(list(range(0, 7))) - set(known) - set(seen_unknown))
        self.unknown = list(set(list(range(0, 7))) - set(known))

        print(
            'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = RSSCN7_Filter(os.path.join(dataroot, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown, known_count=320)
        # trainset.__Filter__(known=self.known, seen_unknown=self.unknown)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        testset_close = RSSCN7_Filter(os.path.join(dataroot, 'test'), transform)
        testset_open= RSSCN7_Filter(os.path.join(dataroot, 'test'), transform)
        valset_close = RSSCN7_Filter(os.path.join(dataroot, 'val'), transform)
        valset_open= RSSCN7_Filter(os.path.join(dataroot, 'val'), transform)

        testset_close.__split__(mode='test')
        print('All Test Close Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)

        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))
        #testset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        valset_close.__split__(mode='test')
        print('All Val Close Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)

        valset_open.__split__(mode='test')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))
        #valset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))





        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))


class siri_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = self.imgs, self.targets
        # print(datas[6][1],targets[6])
        new_datas, new_targets = [], []
        if train:
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in known and known_need:
                    new_item = (datas[i][0], known.index(datas[i][1]))
                    new_datas.append(new_item)
                    new_targets.append(known.index(targets[i]))
                    # new_targets.append(targets[i])
                    # print(i, datas[i][1], targets[i])
                elif datas[i][1] in seen_unknown and unknown_count < known_count:
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known))  # N as seen unknown/unseen unknown
                    unknown_count += 1
        else:
            count = known_count # number of samples on single class
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in seen_unknown: # ('s_unknown' + 'u_unknown')
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    if unknown_count == count:
                        break
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

    def __split__(self, mode=''):
        # a = self.imgs
        # datas = torch.tensor(self.imgs[:][0])
        datas = self.imgs
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 1)  #0.5
        valset_size = len(targets) - testset_size
        test_indices = shuffled_indices[:testset_size]
        val_indices = shuffled_indices[testset_size:]
        # test_indices = torch.tensor(shuffled_indices[:testset_size])
        # val_indices = torch.tensor(shuffled_indices[testset_size:])

        new_datas, new_targets = [], []
        if mode == 'test':
            # self.imgs = torch.index_select(datas, 0, test_indices)
            # self.samples = torch.index_select(datas, 0, test_indices)
            for i in range(len(test_indices)):
                new_item = (datas[test_indices[i]][0], datas[test_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[:testset_size]]
                # self.imgs = datas[test_indices]
                # self.samples = datas[test_indices]
        elif mode == 'val':
            for i in range(len(val_indices)):
                new_item = (datas[val_indices[i]][0], datas[val_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[testset_size:]]
            # self.imgs = torch.index_select(datas, 0, val_indices)
            # self.samples = torch.index_select(datas, 0, val_indices)
            # self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')


class siri_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/NWPU_split', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown = seen_unknown
        self.unseen_unknown = list(set(list(range(0, 12))) - set(known) - set(seen_unknown))
        self.unknown = list(set(list(range(0, 12))) - set(known))

        print(
            'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = siri_Filter(os.path.join(dataroot, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown, known_count=160)
        # trainset.__Filter__(known=self.known, seen_unknown=self.unknown)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        testset_close = siri_Filter(os.path.join(dataroot, 'test'), transform)
        testset_open= siri_Filter(os.path.join(dataroot, 'test'), transform)
        valset_close = siri_Filter(os.path.join(dataroot, 'val'), transform)
        valset_open= siri_Filter(os.path.join(dataroot, 'val'), transform)

        testset_close.__split__(mode='test')
        print('All Test Close Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)

        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))
        #testset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        valset_close.__split__(mode='test')
        print('All Val Close Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)

        valset_open.__split__(mode='test')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))
        #valset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))





        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))



class AID_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = self.imgs, self.targets
        # print(datas[6][1],targets[6])
        new_datas, new_targets = [], []
        if train:
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in known and known_need:
                    new_item = (datas[i][0], known.index(datas[i][1]))
                    new_datas.append(new_item)
                    new_targets.append(known.index(targets[i]))
                    # new_targets.append(targets[i])
                    # print(i, datas[i][1], targets[i])
                elif datas[i][1] in seen_unknown and unknown_count < known_count:
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known))  # N as seen unknown/unseen unknown
                    unknown_count += 1
        else:
            count = known_count # number of samples on single class
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in seen_unknown: # ('s_unknown' + 'u_unknown')
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    if unknown_count == count:
                        break
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

    def __split__(self, mode=''):
        # a = self.imgs
        # datas = torch.tensor(self.imgs[:][0])
        datas = self.imgs
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 1)  #0.5
        valset_size = len(targets) - testset_size
        test_indices = shuffled_indices[:testset_size]
        val_indices = shuffled_indices[testset_size:]
        # test_indices = torch.tensor(shuffled_indices[:testset_size])
        # val_indices = torch.tensor(shuffled_indices[testset_size:])

        new_datas, new_targets = [], []
        if mode == 'test':
            # self.imgs = torch.index_select(datas, 0, test_indices)
            # self.samples = torch.index_select(datas, 0, test_indices)
            for i in range(len(test_indices)):
                new_item = (datas[test_indices[i]][0], datas[test_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[:testset_size]]
                # self.imgs = datas[test_indices]
                # self.samples = datas[test_indices]
        elif mode == 'val':
            for i in range(len(val_indices)):
                new_item = (datas[val_indices[i]][0], datas[val_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[testset_size:]]
            # self.imgs = torch.index_select(datas, 0, val_indices)
            # self.samples = torch.index_select(datas, 0, val_indices)
            # self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')


class AID_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/NWPU_split', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown = seen_unknown
        self.unseen_unknown = list(set(list(range(0, 30))) - set(known) - set(seen_unknown))
        self.unknown = list(set(list(range(0, 30))) - set(known))

        print(
            'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = AID_Filter(os.path.join(dataroot, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown, known_count=176)
        # trainset.__Filter__(known=self.known, seen_unknown=self.unknown)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        testset_close = AID_Filter(os.path.join(dataroot, 'test'), transform)
        testset_open= AID_Filter(os.path.join(dataroot, 'test'), transform)
        valset_close = AID_Filter(os.path.join(dataroot, 'val'), transform)
        valset_open= AID_Filter(os.path.join(dataroot, 'val'), transform)

        testset_close.__split__(mode='test')
        print('All Test Close Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)

        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))
        #testset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        valset_close.__split__(mode='test')
        print('All Val Close Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)

        valset_open.__split__(mode='test')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))
        #valset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))





        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))


class NWPU_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = self.imgs, self.targets
        # print(datas[6][1],targets[6])
        new_datas, new_targets = [], []
        if train:
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in known and known_need:
                    new_item = (datas[i][0], known.index(datas[i][1]))
                    new_datas.append(new_item)
                    new_targets.append(known.index(targets[i]))
                    # new_targets.append(targets[i])
                    # print(i, datas[i][1], targets[i])
                elif datas[i][1] in seen_unknown and unknown_count < known_count:
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known))  # N as seen unknown/unseen unknown
                    unknown_count += 1
        else:
            count = known_count # number of samples on single class
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in seen_unknown: # ('s_unknown' + 'u_unknown')
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    if unknown_count == count:
                        break
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

    def __split__(self, mode=''):
        # a = self.imgs
        # datas = torch.tensor(self.imgs[:][0])
        datas = self.imgs
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 1)  #0.5
        valset_size = len(targets) - testset_size
        test_indices = shuffled_indices[:testset_size]
        val_indices = shuffled_indices[testset_size:]
        # test_indices = torch.tensor(shuffled_indices[:testset_size])
        # val_indices = torch.tensor(shuffled_indices[testset_size:])

        new_datas, new_targets = [], []
        if mode == 'test':
            # self.imgs = torch.index_select(datas, 0, test_indices)
            # self.samples = torch.index_select(datas, 0, test_indices)
            for i in range(len(test_indices)):
                new_item = (datas[test_indices[i]][0], datas[test_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[:testset_size]]
                # self.imgs = datas[test_indices]
                # self.samples = datas[test_indices]
        elif mode == 'val':
            for i in range(len(val_indices)):
                new_item = (datas[val_indices[i]][0], datas[val_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[testset_size:]]
            # self.imgs = torch.index_select(datas, 0, val_indices)
            # self.samples = torch.index_select(datas, 0, val_indices)
            # self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')


class NWPU_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/NWPU_split', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown = seen_unknown
        self.unseen_unknown = list(set(list(range(0, 45))) - set(known) - set(seen_unknown))
        self.unknown = list(set(list(range(0, 45))) - set(known))

        print(
            'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = NWPU_Filter(os.path.join(dataroot, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown, known_count=560)
        # trainset.__Filter__(known=self.known, seen_unknown=self.unknown)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # testset_close = NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        # testset_open= NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        # valset_close = NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        # valset_open= NWPU_Filter(os.path.join(dataroot, 'test'), transform)

        # testset_close.__split__(mode='test')
        # print('All Test Close Data:', len(testset_close))
        # testset_close.__Filter__(known=self.known)

        # testset_open.__split__(mode='test')
        # testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        # valset_close.__split__(mode='val')
        # print('All Val Close Data:', len(valset_close))
        # valset_close.__Filter__(known=self.known)

        # valset_open.__split__(mode='val')
        # valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))

        testset_close = NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        testset_open= NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        valset_close = NWPU_Filter(os.path.join(dataroot, 'val'), transform)
        valset_open= NWPU_Filter(os.path.join(dataroot, 'val'), transform)

        testset_close.__split__(mode='test')
        print('All Test Close Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)

        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))
        #testset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        valset_close.__split__(mode='test')
        print('All Val Close Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)

        valset_open.__split__(mode='test')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))
        #valset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))


        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))

class EuroSAT_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known, seen_unknown=[], known_need=True, train=True, known_count=0):
        datas, targets = self.imgs, self.targets
        # print(datas[6][1],targets[6])
        new_datas, new_targets = [], []
        if train:
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in known and known_need:
                    new_item = (datas[i][0], known.index(datas[i][1]))
                    new_datas.append(new_item)
                    new_targets.append(known.index(targets[i]))
                    # new_targets.append(targets[i])
                    # print(i, datas[i][1], targets[i])
                elif datas[i][1] in seen_unknown and unknown_count < known_count:
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known))  # N as seen unknown/unseen unknown
                    unknown_count += 1
        else:
            count = known_count # number of samples on single class
            unknown_count = 0
            for i in range(len(datas)):
                if datas[i][1] in seen_unknown: # ('s_unknown' + 'u_unknown')
                    new_item = (datas[i][0], len(known))
                    new_datas.append(new_item)
                    new_targets.append(len(known)) # N as seen unknown
                    unknown_count += 1
                    if unknown_count == count:
                        break
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

    def __split__(self, mode=''):
        # a = self.imgs
        # datas = torch.tensor(self.imgs[:][0])
        datas = self.imgs
        targets = np.array(self.targets)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(targets))
        testset_size = int(len(targets) * 1)  #0.5
        valset_size = len(targets) - testset_size
        test_indices = shuffled_indices[:testset_size]
        val_indices = shuffled_indices[testset_size:]
        # test_indices = torch.tensor(shuffled_indices[:testset_size])
        # val_indices = torch.tensor(shuffled_indices[testset_size:])

        new_datas, new_targets = [], []
        if mode == 'test':
            # self.imgs = torch.index_select(datas, 0, test_indices)
            # self.samples = torch.index_select(datas, 0, test_indices)
            for i in range(len(test_indices)):
                new_item = (datas[test_indices[i]][0], datas[test_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[:testset_size]]
                # self.imgs = datas[test_indices]
                # self.samples = datas[test_indices]
        elif mode == 'val':
            for i in range(len(val_indices)):
                new_item = (datas[val_indices[i]][0], datas[val_indices[i]][1])
                new_datas.append(new_item)
                self.samples, self.imgs = new_datas, new_datas
                self.targets = targets[shuffled_indices[testset_size:]]
            # self.imgs = torch.index_select(datas, 0, val_indices)
            # self.samples = torch.index_select(datas, 0, val_indices)
            # self.targets = targets[shuffled_indices[testset_size:]]
        else:
            print('Wrong')


class EuroSAT_OSR(object):
    def __init__(self, known, seen_unknown, dataroot='./data/NWPU_split', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.seen_unknown = seen_unknown
        self.unseen_unknown = list(set(list(range(0, 10))) - set(known) - set(seen_unknown))
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print(
            'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, seen_unknown,
                                                                                                   self.unseen_unknown))
        # print(
        #     'Selected Known Labels: {}, Seen Unknown Labels: {}, Unseen Unknown Labels: {}'.format(known, self.unknown,
        #                                                                                            self.unseen_unknown))
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = EuroSAT_Filter(os.path.join(dataroot, 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known, seen_unknown=self.seen_unknown, known_count=1600)
        # trainset.__Filter__(known=self.known, seen_unknown=self.unknown)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # testset_close = NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        # testset_open= NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        # valset_close = NWPU_Filter(os.path.join(dataroot, 'test'), transform)
        # valset_open= NWPU_Filter(os.path.join(dataroot, 'test'), transform)

        # testset_close.__split__(mode='test')
        # print('All Test Close Data:', len(testset_close))
        # testset_close.__Filter__(known=self.known)

        # testset_open.__split__(mode='test')
        # testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        # valset_close.__split__(mode='val')
        # print('All Val Close Data:', len(valset_close))
        # valset_close.__Filter__(known=self.known)

        # valset_open.__split__(mode='val')
        # valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))

        testset_close = EuroSAT_Filter(os.path.join(dataroot, 'test'), transform)
        testset_open= EuroSAT_Filter(os.path.join(dataroot, 'test'), transform)
        valset_close = EuroSAT_Filter(os.path.join(dataroot, 'val'), transform)
        valset_open= EuroSAT_Filter(os.path.join(dataroot, 'val'), transform)

        testset_close.__split__(mode='test')
        print('All Test Close Data:', len(testset_close))
        testset_close.__Filter__(known=self.known)

        testset_open.__split__(mode='test')
        testset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))
        #testset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(testset_close)/len(known)))

        valset_close.__split__(mode='test')
        print('All Val Close Data:', len(valset_close))
        valset_close.__Filter__(known=self.known)

        valset_open.__split__(mode='test')
        valset_open.__Filter__(known=self.known, seen_unknown=self.unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))
        #valset_open.__Filter__(known=self.known, seen_unknown=self.unseen_unknown, known_need=False, train=False, known_count=int(len(valset_close)/len(known)))


        self.test_loader = torch.utils.data.DataLoader(
            testset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            testset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.val_close_loader = torch.utils.data.DataLoader(
            valset_close, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_open_loader = torch.utils.data.DataLoader(
            valset_open, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train data: ', len(trainset))
        print('Test Close-set: {} Open-set: {}'.format(len(testset_close), len(testset_open)))
        print('Val  Close-set: {} Open-set: {}'.format(len(valset_close), len(valset_open)))



if __name__ == '__main__':
    # options = {'dataset': 'tiny_imagenet', 'dataroot': './data/tiny_imagenet', 'outf': './log',
    #            'out_num': 50, 'batch_size': 128, 'lr': 0.001, 'gan_lr': 0.0002, 'max_epoch': 100,
    #            'stepsize': 30, 'temp': 1.0, 'num_centers': 1, 'weight_pl': 0.1, 'beta': 0.1,
    #            'model': 'classifier32', 'nz': 100, 'ns': 1, 'eval_freq': 1, 'print_freq': 100,
    #            'gpu': '0', 'seed': 0, 'use_cpu': False, 'save_dir': '../log', 'loss': 'OpenAUCLoss',
    #            'eval': False, 'cs': False, 'item': 0,
    #            'known': [95, 6, 145, 153, 0, 143, 31, 23, 189, 81, 20, 21, 89, 26, 36, 170,
    #                      102, 177, 108, 169],
    #            'unknown': [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 22, 24,
    #                        25, 27, 28, 29, 30, 32, 33, 34, 35, 37,38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    #                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #                        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88,
    #                        90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 109, 110,
    #                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
    #                        127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    #                        144, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161,
    #                        162, 163, 164, 165, 166, 167, 168, 171, 172, 173, 174, 175, 176, 178, 179, 180,
    #                        181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197,
    #                        198, 199], 'img_size': 64}
    # Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
    #                          img_size=options['img_size'])
    # trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    options = {'dataset': 'cifar10', 'dataroot': '/home/zjhuang/A2023/ARPL-master/data/cifar10', 'outf': './log', 'out_num': 10, 'batch_size': 128, 'lr': 0.001,
     'gan_lr': 0.0002, 'max_epoch': 600, 'stepsize': 30, 'temp': 1.0, 'num_centers': 1, 'weight_pl': 0.1, 'beta': 0.1,
     'model': 'classifier32', 'nz': 100, 'ns': 1, 'eval_freq': 1, 'print_freq': 100, 'gpu': '0', 'seed': 1234,
     'use_cpu': False, 'save_dir': '../log', 'loss': 'MyLoss', 'eval': False, 'cs': False, 'item': 0,
     'known': [0,1,2,3], 'seen_unknown': [4,5,6,7,8,9], 'unseen_unknown':[1, 5, 7],'img_size': 32}
    Data = CIFAR10_OSR(known=options['known'], seen_unknown=options['seen_unknown'],dataroot=options['dataroot'], batch_size=options['batch_size'],
                     img_size=options['img_size'])
    trainloader, testloader= Data.train_loader, Data.test_loader

    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.cuda(), labels.cuda()
    # Data.out_loader
    # trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    print(trainloader)
    print(testloader)
    # print(outloader)