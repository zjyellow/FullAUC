import os
import os.path as osp
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from core import evaluation
import libmr

def test(net, criterion, testloader, outloader, epoch=None, trainloader=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                if options['loss'] in ['OVRN']:
                    x, y = net(data, True, True)
                elif options['loss'] in ['FAUCLoss','FAUCLoss2']:
                    x, y = net(data, True)
                elif options['loss'] in ['MDL4OW']:
                    ori_x, re_x, logits = net(data, False, False, True)
                else:
                    x, y = net(data, True)
                if options['loss'] == 'OpenAUCLoss':
                    logits = y
                elif options['loss'] == 'Proxy_Anchor':
                    logits, _ = criterion(x, y, labels)
                elif options['loss'] == 'CACLoss':
                    distances, _ = criterion(x, y, labels, train_mode=False)
                    softmax = torch.nn.Softmax(dim = 1)
                    softmin = softmax(-distances)
                    invScores = 1-softmin
                    scores = distances*invScores
                    logits = -scores
                elif options['loss'] in ['PTLLoss']:
                    logits, _ = criterion(x, y, labels, train_mode=False) # use distance_classifier
                    logits = -logits
                elif options['loss'] in ['MDL4OW']:
                    recon_loss, logits = criterion(ori_x, re_x, logits) 
                else:
                    logits, _ = criterion(x, y)
                
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                # if options['loss'] in ['MDL4OW']:
                #     logits = recon_loss # use recon_loss to calculate AUROC and OSCR
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                if options['loss'] in ['OVRN']:
                    x, y = net(data, True, True)
                elif options['loss'] in ['FAUCLoss','FAUCLoss2']:
                    x, y = net(data, True)
                elif options['loss'] in ['MDL4OW']:
                    ori_x, re_x, logits = net(data, False, False, True)
                else:
                    x, y = net(data, True)
                if options['loss'] == 'OpenAUCLoss':
                    logits = y
                elif options['loss'] == 'Proxy_Anchor':
                    logits, _ = criterion(x, y, labels)
                elif options['loss'] == 'CACLoss':
                    distances, _ = criterion(x, y, labels, train_mode=False)
                    # distances, _ = criterion(x, y, labels, anchors=anchor_means)
                    softmax = torch.nn.Softmax(dim = 1)
                    softmin = softmax(-distances)
                    invScores = 1-softmin
                    scores = distances*invScores
                    logits = -scores
                elif options['loss'] in ['PTLLoss']:
                    logits, _ = criterion(x, y, labels, train_mode=False) # use distance_classifier
                    logits = -logits
                elif options['loss'] in ['MDL4OW']:
                    recon_loss, logits = criterion(ori_x, re_x, logits) 
                else:
                    logits, _ = criterion(x, y)
                # x, y = net(data, return_feature=True)
                # if options['loss'] in ['MDL4OW']:
                #     logits = recon_loss # use recon_loss to calculate AUROC and OSCR
                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    # print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results

def gather_outputs(net, dataloader, num_classes):
    """
        only for CACLoss.
        dataloader: 
            trainloader: compute modified anchor.
            testloader:  known classes for test.
            outloader: unknown classes for test.
        data_idx:
            0: compute modified anchor.
            1: known/unknown classes test.
        calculate_scores:
            False: compute modified anchor.
            True: known/unknown classes test.
        only_correct:
            True: compute modified anchor.
            False: known/unknown classes test.
    """
    X = []
    y = []

    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.cuda()
        targets = labels.cuda()
        embedding, logits = net(images, True) # embedding, logits
        distances = distance_classifier(logits, num_classes)

        _, predicted = torch.max(logits, 1)
        mask = predicted == targets

        logits = logits[mask]
        
        distances = distances[mask]
        targets = targets[mask]
        scores = logits
        X += scores.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

def distance_classifier(logits, num_classes):
    # Calculates euclidean distance from x to each class anchor
    # Returns n x m array of distance from input of batch_size n to anchors of size m
    anchors = torch.diag(torch.Tensor([10 for i in range(num_classes)])).cuda()

    n = logits.size(0) # 128
    m = num_classes # 6
    d = num_classes # 6
    logits = logits.unsqueeze(1).expand(n, m, d).double() # 128x6 -> 128x1x6 ->128x6x6
    anchors = anchors.unsqueeze(0).expand(n, m, d)
    dists = torch.norm(logits-anchors, 2, 2) # 128x6
    return dists