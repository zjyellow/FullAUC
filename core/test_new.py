import os
import os.path as osp
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import libmr

def get_curve_online(known, novel, stypes=['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes=['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    # mtypes = ['AUROC']
    # if verbose:
    #     print('      ', end='')
    #     for mtype in mtypes:
    #         print(' {mtype:6s}'.format(mtype=mtype), end='')
    #     print('')

    for stype in stypes:
        # if verbose:
        #     print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        # if verbose:
        #     print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[stype][mtype] = 100. * (-np.trapz(1. - fpr, tpr))
        # if verbose:
        #     print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        # if verbose:
        #     print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        # if verbose:
        #     print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        # if verbose:
        #     print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        #     print('')

    return results


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)
    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w
    return OSCR

def test(net, criterion, testloader, outloader, Val=False, mean_score = 0., threshold=0., epoch=None, **options):
    net.eval()
    correct, total = 0, 0
    close_score,mean_score = torch.tensor([]).cuda(), mean_score
    unknown_correct, unknown_total = 0, 0
    torch.cuda.empty_cache()
    known_labels = []
    known_pred = []
    unknown_labels = []
    unknown_pred = []
    _pred_k, _pred_u, _labels = [], [], []

    # Known(close-set) Acc
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
                    # logits, loss = criterion(y, labels, net.module.fc, x)
                    logits = y
                elif options['loss'] == 'GCACLoss':
                    distances, _ = criterion(x, y, labels, train_mode=False)
                    softmax = torch.nn.Softmax(dim = 1)
                    softmin = softmax(-distances)
                    invScores = 1-softmin
                    scores = distances*invScores
                    logits = -scores
                elif options['loss'] == 'GPTLLoss':
                    logits, _ = criterion(x, y, labels, train_mode=False) # use distance_classifier
                    logits = -logits
                elif options['loss'] in ['MDL4OW']:
                    recon_loss, logits = criterion(ori_x, re_x, logits) 
                else:
                    logits, _ = criterion(x, y)

                
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                _pred_k.append(logits.data.cpu().numpy()) # for AUROC and OSCR
                _labels.append(labels.data.cpu().numpy()) # for AUROC and OSCR

                if options['loss'] in ['MDL4OW']:
                    logits = recon_loss # use recon_loss to calculate AUROC and OSCR
                
                score, y_hat = torch.max(logits, dim=-1)
                # score = torch.sigmoid(score)
                
                close_score = torch.cat([close_score,score])
                known_labels = np.hstack((known_labels, labels.data.cpu().numpy()))
                known_pred = np.hstack((known_pred, predictions.data.cpu().numpy()))
        if Val:  
            if options['loss'] in ['FAUCLoss','FAUCLoss2']: # Only Val stage calculate the mean score
                mean_score = torch.mean(close_score)
                margain = options['gama']
                threshold = mean_score - margain
                print('Score_close: {:.4f} \t Margain: {:.4f} \t Threshold: {:.4f}'.format(mean_score,margain,threshold))
            else:
                pass
                # threshold = torch.quantile(close_score, 0.05)
                # print('5% quantile Threshold: {:.4f}'.format(threshold))
        
        if options['loss'] in ['MDL4OW']:
            mr = libmr.MR()
            mr.fit_high(close_score.data.cpu().numpy(), len(close_score.data.cpu().numpy()) * 0.5) # use EVT for open-set score

        # all unknown acc
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
                    # logits, loss = criterion(y, labels, net.module.fc, x)
                    logits = y
                elif options['loss'] == 'GCACLoss':
                    distances, _ = criterion(x, y, labels, train_mode=False)
                    softmax = torch.nn.Softmax(dim = 1)
                    softmin = softmax(-distances)
                    invScores = 1-softmin
                    scores = distances*invScores
                    logits = -scores
                elif options['loss'] == 'GPTLLoss':
                    logits, _ = criterion(x, y, labels, train_mode=False) # use distance_classifier
                    logits = -logits
                elif options['loss'] in ['MDL4OW']:
                    recon_loss, logits = criterion(ori_x, re_x, logits) 
                else:
                    logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy()) # for AUROC and OSCR
                unknown_score = logits
                
                unknown_score, unknown_predictions = unknown_score.data.max(1)
                
                if options['loss'] in ['FAUCLoss']:
                    unknown_predictions[unknown_score < threshold] = logits.size(1)
                elif options['loss'] in ['FAUCLoss2']:
                    unknown_predictions[unknown_score < threshold] = logits.size(1) - 1
                elif options['loss'] in ['MDL4OW']:
                    wscore = mr.w_score_vector(recon_loss.double().squeeze(1).data.cpu().numpy())
                    unknown_predictions[wscore > 0.5] = logits.size(1) - 1 
                else:
                    # unknown_predictions[unknown_score < threshold] = logits.size(1) - 1
                    pass
                unknown_total += labels.size(0)
                unknown_correct += (unknown_predictions == labels.data).sum()
                unknown_labels = np.hstack((unknown_labels, labels.data.cpu().numpy()))
                unknown_pred = np.hstack((unknown_pred, unknown_predictions.data.cpu().numpy()))


    # close set Accuracy
    acc = float(correct) * 100. / float(total)
    acc_all = float(unknown_correct + correct) * 100. / float(total + unknown_total)

    all_labels = np.hstack((known_labels,unknown_labels))
    all_pred = np.hstack((known_pred,unknown_pred))
    F1_macro = f1_score(all_labels, all_pred, average='macro') # f1 consider unknown class
    # print('Close Acc: {:.4f} \n All Acc: {:.4f} \n F1-score: {:.4f}'.format(acc, acc_all, F1_macro))
    results = dict()
    results['Acc_close'] = acc
    results['Acc_all'] = acc_all
    results['F1_score'] = F1_macro

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    ood_results = metric_ood(x1, x2)['Bas']

    results['AUROC'] = ood_results['AUROC']

    # OSCR
    _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels)
    results['OSCR'] = _oscr_socre * 100.

    results['TNR'] = ood_results['TNR']
    results['DTACC'] = ood_results['DTACC']
    results['AUIN'] = ood_results['AUIN']
    results['AUOUT'] = ood_results['AUOUT']
    
    return results, mean_score, threshold