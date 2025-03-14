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
import sys
from models import gan
from models.models import classifier32, classifier32ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR, NWPU_OSR, RSSCN7_OSR, siri_OSR, AID_OSR, EuroSAT_OSR
from utils import Logger, save_networks, load_networks
from core import train, train_cs, test
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser("Training")
tensorboard_dir = "./log/log"
parser.add_argument('--exp_name', type=str, default='osr', help='For exp name')
# Dataset
parser.add_argument('--dataset', type=str, default='AID', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | nwpu | RSSCN7 | siri | AID | EuroSAT")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=10, help='For CIFAR100')
parser.add_argument('--gama', type=float, default=0.2, help='For MyLoss/MyLoss_new')
# optimization
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
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
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='FAUCLoss | CACLoss | Softmax | OpenAUCLoss | ARPLoss | GCPLoss | OVRN | PTLLoss')
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

    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    net = classifier32(num_classes=options['num_classes'])
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

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'log','models', options['loss'], options['exp_name'],options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if options['dataset'] == 'cifar100':
        if options['out_num'] == 10:
            model_path += '_10'
        else:
            model_path += '_50'
        file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], options['out_num'], options['item'])
    else:
        file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['item'])

    # if options['eval']:
    #     net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
    #     results = test(net, criterion, testloader, outloader, epoch=0, **options)
    #     print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

    #     return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]

    optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=int(options['max_epoch'] / (2 + 1)),
                                                                            eta_min=5e-5)

    start_time = time.time()
    writer = SummaryWriter(
        os.path.join(tensorboard_dir, options['loss'], options['exp_name'], options['dataset'] + '_group_' + str(options['item']) + '_' + options['loss']))
    
    TNR_list = []; AUROC_list = []; DTACC_list = []; AUIN_list = []; AUOUT_list = []; ACC_list = []; OSCR_list = []
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        _, loss_avg = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        logging.info('Epoch {} Train info lr {}'.format(epoch, loss_avg))
        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])


            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)

            print("current: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('epoch/train_loss',loss_avg,epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            writer.add_scalar('epoch/test_ACC', results['ACC'], epoch)
            writer.add_scalar('epoch/test_AUROC', results['AUROC'], epoch)
            writer.add_scalar('epoch/test_OSCR', results['OSCR'], epoch)
            save_networks(net, model_path, file_name, criterion=criterion)

            TNR_list.append(results['TNR']); AUROC_list.append(results['AUROC']); DTACC_list.append(results['DTACC']); AUIN_list.append(results['AUIN']); AUOUT_list.append(results['AUOUT']); ACC_list.append(results['ACC']); OSCR_list.append(results['OSCR'])
            max_TNR = max(TNR_list); max_AUROC = max(AUROC_list); max_DTACC = max(DTACC_list); max_AUIN = max(AUIN_list); max_AUOUT = max(AUOUT_list); max_ACC = max(ACC_list); max_OSCR = max(OSCR_list); 
            results_max = {'TNR_max': max_TNR, 'AUROC_max': max_AUROC, 'DTACC_max': max_DTACC, 'AUIN_max': max_AUIN, 'AUOUT_max': max_AUOUT, 'ACC_max': max_ACC, 'OSCR_max': max_OSCR}
            print("MAX: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_max['ACC_max'], results_max['AUROC_max'], results_max['OSCR_max']))
            
        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    return results_max

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()
    
    from split import splits_2020 as splits
    
    log_path = os.path.join(options['outf'], 'log-osr', options['loss'])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    filename = options['dataset'] + '_' + options['exp_name'] + '_' + 'logs.txt'
    log_path = os.path.join(log_path, filename)
    sys.stdout = Logger(log_path)


    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset']+'-'+str(options['out_num'])][len(splits[options['dataset']])-i-1]
        elif options['dataset'] == 'tiny_imagenet':
            img_size = 64
            # options['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        elif options['dataset'] == 'nwpu':
            img_size = 64
            unknown = list(set(list(range(0, 45))) - set(known))
        elif options['dataset'] == 'RSSCN7':
            img_size = 64
            unknown = list(set(list(range(0, 7))) - set(known))
        elif options['dataset'] == 'siri':
            img_size = 64
            unknown = list(set(list(range(0, 12))) - set(known))
        elif options['dataset'] == 'AID':
            img_size = 64
            unknown = list(set(list(range(0, 30))) - set(known))
        elif options['dataset'] == 'EuroSAT':
            img_size = 64
            unknown = list(set(list(range(0, 10))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))


        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar100':
            file_name = '{}_{}_{}.csv'.format(options['dataset'], options['out_num'], options['exp_name'])
        else:
            file_name = options['exp_name'] + '_' + options['dataset'] + '_' + str(options['gama']) +'.csv'

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))