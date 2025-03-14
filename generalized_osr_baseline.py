import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from models import gan
from models.models import classifier32, classifier32ABN
import os.path as osp
import sys
from datasets.osr_dataloader_generalized import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR, NWPU_OSR, RSSCN7_OSR, siri_OSR, AID_OSR, EuroSAT_OSR
from utils import Logger, save_networks, load_networks, save_networks2
from core import train, train_cs
from core.test_new import  test
from tensorboardX import SummaryWriter
# from torch.nn.utils import clip_grad_norm


parser = argparse.ArgumentParser("Training")
tensorboard_dir = "./log/log"
parser.add_argument('--exp_name', type=str, default='g-osr', help='For exp name')

# Dataset
parser.add_argument('--dataset', type=str, default='nwpu', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | RSSCN7 | nwpu | siri | AID | EuroSAT")
parser.add_argument('--out-dataset', type=str, default='cifar', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=10, help='For CIFAR100 10 or 50')
parser.add_argument('--gama', type=float, default=0.2, help='For MyLoss_new/MyLoss_new2')

# optimization
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")

parser.add_argument('--max-epoch', type=int, default=600)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='GPTLLoss') # FAUCLoss | FAUCLoss2| Softmax | GCPLoss | RPLoss | ARPLoss | OpenAUCLoss | OVRN | MyLoss_new2 | GMCAUC | GCACLoss
parser.add_argument('--eval', action='store_true', help="Eval", default=False)

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    print("dataroot: {}".format(options['dataroot']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], seen_unknown=options['seen_unknown'],dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader,val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], seen_unknown=options['seen_unknown'],dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
        out_Data = CIFAR100_OSR(known=options['known'], unseen_unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'], count=len(testloader.dataset))
        outloader = out_Data.test_loader
    elif 'tiny_imagenet' in options['dataset']:
        Data = Tiny_ImageNet_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'nwpu' in options['dataset']:
        Data = NWPU_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'RSSCN7' in options['dataset']:
        Data = RSSCN7_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'siri' in options['dataset']:
        Data = siri_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'AID' in options['dataset']:
        Data = AID_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    elif 'EuroSAT' in options['dataset']:
        Data = EuroSAT_OSR(known=options['known'], seen_unknown=options['seen_unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        val_close_loader, val_open_loader = Data.val_close_loader, Data.val_open_loader
    
    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    if options['loss'] in ['FAUCLoss']:
        net = classifier32(num_classes=options['num_classes']) # without filler class (FullAUC-NF)
    else:
        net = classifier32(num_classes=options['num_classes'] + 1) # with filler class, for baseline and FullAUC-F
    feat_dim = 128

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss']) 
    criterion = getattr(Loss, options['loss'])(**options)   
    print(Loss)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    model_path2 = os.path.join(options['outf'], 'log','models', options['loss'], options['exp_name'],options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], 50, options['item'])
    else:
        file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['item'])
        # classifier32_MyLoss_new_0_False    _.pth

    # eval after all train
    # if options['eval']: # False
    #     net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
    #     results = test(net, criterion, testloader, outloader, epoch=0, **options)
    #     print("Acc_close (%): {:.4f}\t Acc_all (%): {:.4f}\t F1-score (%): {:.4f}".format(results['Acc_close'],
    #                                                                                         results['Acc_all'],
    #                                                                                         results['F1_score']))
    #     return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    
    optimizer = torch.optim.Adam(params_list,lr=options['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=int(options['max_epoch'] / (2 + 1)),eta_min=5e-5)
    
    start_time = time.time()

    writer = SummaryWriter(
        os.path.join(tensorboard_dir, options['loss'], options['exp_name'], options['dataset'] + '_group_' + str(options['item']) + '_' + options['loss']))

    Acc_close_list = []; Acc_all_list = []; F1_score_list = []; AUROC_list = []; OSCR_list = []
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        _, loss_avg=train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        logging.info('Epoch {} Train info lr {}'.format(epoch,loss_avg))

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Val", options['loss'])
            _, cv_loss_avg = train(net, criterion, optimizer, val_close_loader, epoch=epoch, **options)
            val_results, mean_score, threshold = test(net, criterion, val_close_loader, val_open_loader, epoch=epoch, Val=True, **options)
            print("Acc_close (%): {:.4f}\t Acc_all (%): {:.4f}\t F1-score : {:.4f}\t AUROC :{:.4f}\t OSCR :{:.4f}".format(val_results['Acc_close'],
                                                                                              val_results['Acc_all'],
                                                                                              val_results['F1_score'],val_results['AUROC'],val_results['OSCR']))
            logging.info('Epoch {} CV info cv_acc_close {} cv_acc_all {} cv_F1_score {} cv_auroc {} cv_oscr {}'.format(epoch, val_results['Acc_close'], val_results['Acc_all'], val_results['F1_score'],val_results['AUROC'],val_results['OSCR']))

            print("==> Test", options['loss'])
            results, _, _ = test(net, criterion, testloader, outloader, epoch=epoch, Val=False, threshold=threshold, mean_score = mean_score, **options)

            print("CURRENT: Acc_close (%): {:.4f}\t Acc_all (%): {:.4f}\t F1-score : {:.4f}\t AUROC :{:.4f}\t OSCR :{:.4f}".format(results['Acc_close'], results['Acc_all'],results['F1_score'],results['AUROC'],results['OSCR']))

            Acc_close_list.append(results['Acc_close']); Acc_all_list.append(results['Acc_all']); F1_score_list.append(results['F1_score']); AUROC_list.append(results['AUROC']); OSCR_list.append(results['OSCR'])
            max_Acc_close = max(Acc_close_list); max_Acc_all = max(Acc_all_list); max_F1_score = max(F1_score_list); max_AUROC = max(AUROC_list); max_OSCR = max(OSCR_list); 
            results_max = {'Acc_close_max': max_Acc_close, 'Acc_all_max': max_Acc_all, 'F1_score_max': max_F1_score, 'AUROC_max': max_AUROC, 'OSCR_max': max_OSCR}
            print("MAX: Acc_close (%): {:.4f}\t Acc_all (%): {:.4f}\t F1-score : {:.4f}\t AUROC :{:.4f}\t OSCR :{:.4f}".format(results_max['Acc_close_max'], results_max['Acc_all_max'],results_max['F1_score_max'],results_max['AUROC_max'],results_max['OSCR_max']))

            lr = optimizer.param_groups[0]['lr']
            
            writer.add_scalar('epoch/cv_acc_close', val_results['Acc_close'],epoch)
            writer.add_scalar('epoch/cv_acc_all', val_results['Acc_all'], epoch)
            writer.add_scalar('epoch/cv_f1score',val_results['F1_score'],epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            writer.add_scalar('epoch/test_acc_close', results['Acc_close'], epoch)
            writer.add_scalar('epoch/test_acc_all', results['Acc_all'], epoch)
            writer.add_scalar('epoch/test_f1score', results['F1_score'], epoch)
            writer.add_scalar('epoch/train_loss', loss_avg, epoch)
            writer.add_scalar('epoch/val_AUROC', val_results['AUROC'], epoch)
            writer.add_scalar('epoch/val_OSCR', val_results['OSCR'], epoch)
            writer.add_scalar('epoch/test_AUROC', results['AUROC'], epoch)
            writer.add_scalar('epoch/test_OSCR', results['OSCR'], epoch)

            save_networks2(net, model_path2, file_name, criterion=criterion, epoch=epoch)
        
        if options['stepsize'] > 0:
            scheduler.step()
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    writer.close()
    return results_max

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()


    from split_new2 import splits_2020 as splits

    log_path = osp.join(options['outf'], 'log-g-osr', options['loss'])
    if not osp.exists(log_path):
        os.makedirs(log_path)

    filename = options['dataset'] + '_' + options['exp_name'] + '_' + 'logs.txt'
    log_path = osp.join(log_path, filename)
    sys.stdout = Logger(log_path)

    for i in range(len(splits[options['dataset']+ '_known'])):

        if (options['dataset'] == 'mnist') or (options['dataset'] == 'cifar10'):
            known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
            seen_unknown = []
            unknown = list(set(list(range(0, 10))) - set(known))
        if (options['dataset'] == 'svhn'):
            img_size = 32
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 200))) - set(known) - set(seen_unknown))
        if options['dataset'] == 'cifar100':
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = list(set(list(range(0, 10))) - set(known))
            unknown = splits[options['dataset']+'-'+str(options['out_num'])][len(splits[options['dataset']+'-'+str(options['out_num'])])-i-1]
        elif options['dataset'] == 'tiny_imagenet':
            img_size = 64
            # options['lr'] = 0.001
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 200))) - set(known) - set(seen_unknown))
        elif options['dataset'] == 'nwpu':
            img_size = 64
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 45))) - set(known) - set(seen_unknown))
        elif options['dataset'] == 'RSSCN7':
            img_size = 64
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 7))) - set(known) - set(seen_unknown))
        elif options['dataset'] == 'siri':
            img_size = 64
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 12))) - set(known) - set(seen_unknown))
        elif options['dataset'] == 'AID':
            img_size = 64
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 30))) - set(known) - set(seen_unknown))
        elif options['dataset'] == 'EuroSAT':
            img_size = 64
            known = splits[options['dataset'] + '_known'][len(splits[options['dataset'] + '_known']) - i - 1]
            seen_unknown = splits[options['dataset'] + '_seen_unknown'][len(splits[options['dataset'] + '_seen_unknown']) - i - 1]
            unknown = list(set(list(range(0, 10))) - set(known) - set(seen_unknown))
        else:
            unknown = list(set(list(range(0, 10))) - set(known)-set(seen_unknown))

        options.update(
            {
                'item':     i,
                'known':    known,
                'seen_unknown': seen_unknown,
                'unknown':  unknown,
                'img_size': img_size,
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar100':
            file_name = options['exp_name'] + '_' + options['dataset'] + '_' + str(options['out_num']) + '.csv'

        else:
            file_name = options['exp_name'] + '_' + options['dataset'] +  '.csv'

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        res['seen_unknown'] = seen_unknown
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))