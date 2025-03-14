import os
import sys
import argparse
import datetime
import time
import csv
import os.path as osp
import numpy as np
import warnings
import importlib
import pandas as pd
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.utils as vutils

import datasets.datasets as datasets
from models.models import ConvNet
from models.resnet import ResNet34
from models.resnetABN import resnet34ABN
from models import gan
from utils import Logger, save_networks, save_GAN, load_networks, mkdir_if_missing
from core import train, train_cs, test
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser("PTLLoss")
tensorboard_dir = "./log/ood/log"
parser.add_argument('--exp_name', type=str, default='ood-', help='For exp name')

# dataset
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')

parser.add_argument('--dataset', type=str, default='nwpu_ood') # in: nwpu_ood out: EuroSAT_ood | in: RSSCN7_ood out: siri_ood
parser.add_argument('--out-dataset', type=str, default='EuroSAT_ood')# 
parser.add_argument('--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")

# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model") 
parser.add_argument('--max-epoch', type=int, default=600)

parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--loss', type=str, default='FAUCLoss | CACLoss | Softmax | OpenAUCLoss | ARPLoss | GCPLoss | OVRN | PTLLoss | CACLoss')
parser.add_argument('--gama', type=float, default=0.2, help='For FAUCLoss')

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='resnet34')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Samples", default=False)
parser.add_argument('--num-centers', type=int, default=1)
args = parser.parse_args()
options = vars(args)

log_path = osp.join(options['outf'], 'OOD', options['loss'])
if not osp.exists(log_path):
    os.makedirs(log_path)
# print(options)
filename = options['dataset'] + '_' + options['out_dataset'] + '_' + options['exp_name'] + '_' + 'logs.txt'
log_path = osp.join(log_path, filename)
sys.stdout = Logger(log_path)

def main():
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    feat_dim = 2 if 'cnn' in options['model'] else 512

    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    dataset = datasets.create(options['dataset'], **options)
    out_dataset = datasets.create(options['out_dataset'], **options)
    
    trainloader, testloader = dataset.trainloader, dataset.testloader
    outloader = out_dataset.testloader

    options.update(
        {
            'num_classes': dataset.num_classes
        }
    )

    print("Creating model: {}".format(options['model']))
    if 'cnn' in options['model']:
        net = ConvNet(num_classes=dataset.num_classes)
    else:
        if options['cs']:
            net = resnet34ABN(num_classes=dataset.num_classes, num_bns=2)
        else:
            net = ResNet34(dataset.num_classes)

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
        criterion = criterion.cuda()
    
    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['dataset'], options['loss'], str(options['weight_pl']), str(options['cs']))
    # if options['eval']:
    #     net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
    #     results = test(net, criterion, testloader, outloader, epoch=0, **options)
    #     print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
    #     return

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    

    optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=int(options['max_epoch'] / (2 + 1)),eta_min=5e-5)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=3,
    #     min_lr=1e-4,
    #     threshold=0.01,
    # )

    start_time = time.time()
    writer = SummaryWriter(
        os.path.join(tensorboard_dir, options['loss'], options['exp_name'],
                     options['dataset'] + '_' + options['out_dataset'] + '_' + options['loss']))

    score_now = 0.0
    TNR_list = []; AUROC_list = []; DTACC_list = []; AUIN_list = []; AUOUT_list = []; ACC_list = []; OSCR_list = []
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        # if options['cs']:
        #     train_cs(net, netD, netG, criterion, criterionD,
        #         optimizer, optimizerD, optimizerG,
        #         trainloader, epoch=epoch, **options)

        _, loss_avg = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test")
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            
            TNR_list.append(results['TNR']); AUROC_list.append(results['AUROC']); DTACC_list.append(results['DTACC']); AUIN_list.append(results['AUIN']); AUOUT_list.append(results['AUOUT']); ACC_list.append(results['ACC']); OSCR_list.append(results['OSCR'])

            # combined_list = [[TNR_item, AUROC_item, DTACC_item, AUIN_item, AUOUT_item, ACC_item, OSCR_item] for TNR_item, AUROC_item, DTACC_item, AUIN_item, AUOUT_item, ACC_item, OSCR_item in zip(TNR_list, AUROC_list, DTACC_list, AUIN_list, AUOUT_list, ACC_list, OSCR_list)]
            # combined_numpy = np.array(combined_list)
            # max_index = np.argmax(combined_numpy[:, 6])
            # max_value_row = combined_numpy[max_index]
            # max_TNR = max_value_row[0]; max_AUROC = max_value_row[1]; max_DTACC = max_value_row[2]; max_AUIN = max_value_row[3]; max_AUOUT = max_value_row[4]; max_ACC = max_value_row[5]; max_OSCR = max_value_row[6]

            max_TNR = max(TNR_list); max_AUROC = max(AUROC_list); max_DTACC = max(DTACC_list); max_AUIN = max(AUIN_list); max_AUOUT = max(AUOUT_list); max_ACC = max(ACC_list); max_OSCR = max(OSCR_list); 

            results_max = {'TNR_max': max_TNR, 'AUROC_max': max_AUROC, 'DTACC_max': max_DTACC, 'AUIN_max': max_AUIN, 'AUOUT_max': max_AUOUT, 'ACC_max': max_ACC, 'OSCR_max': max_OSCR}
            print("MAX: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_max['ACC_max'], results_max['AUROC_max'], results_max['OSCR_max']))

            save_networks(net, model_path, file_name, criterion=criterion)
            # if options['cs']:
            #     save_GAN(netG, netD, model_path, file_name)
            #     fake = netG(fixed_noise)
            #     GAN_path = os.path.join(model_path, 'samples')
            #     mkdir_if_missing(GAN_path)
            #     vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(GAN_path, epoch), normalize=True)
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('epoch/train_loss', loss_avg, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            writer.add_scalar('epoch/test_ACC', results_max['ACC_max'], epoch)
            writer.add_scalar('epoch/test_AUROC', results_max['AUROC_max'], epoch)
            writer.add_scalar('epoch/test_OSCR', results_max['OSCR_max'], epoch)
            writer.add_scalar('epoch/test_TNR', results_max['TNR_max'], epoch)
            writer.add_scalar('epoch/test_DTACC', results_max['DTACC_max'], epoch)
            writer.add_scalar('epoch/test_AUIN', results_max['AUIN_max'], epoch)
            writer.add_scalar('epoch/test_AUOUT', results_max['AUOUT_max'], epoch)

        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

    file_name = options['exp_name'] + '_' + options['dataset'] + '_' + options['out_dataset'] +  '.csv'
    df = pd.DataFrame(results_max,index=[0])
    dir_path = '/home/zijunhuang/AUC/log/OOD/' + options['loss']
    df.to_csv(os.path.join(dir_path, file_name))

    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

if __name__ == '__main__':
    main()

