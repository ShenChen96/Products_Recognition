import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from copy import deepcopy
import importlib
import logging
import sys
sys.path.append("../")

import argparse
from collections import defaultdict
from tqdm import tqdm

from zzClassifier.datasets import Product_Dataloader
from zzClassifier.losses import rzloss
from zzClassifier.models import gan
from zzClassifier.models.resnet import resnet18
from zzClassifier.models.models import classifier32, classifier32ABN
from zzClassifier.core import train, train_cs, test
from utils import Logger, save_networks, load_networks


# Dataset
def get_args():
    parser = argparse.ArgumentParser("Training")

    parser.add_argument('--dataset', type=str, default='product')

    parser.add_argument('--train_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/train')
    parser.add_argument('--val_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/val')
    parser.add_argument('--unknown_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/unknow')

    parser.add_argument('--outf', type=str, default='./log')
    parser.add_argument('--out-num', type=int, default=50, )

    # optimization
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
    parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
    parser.add_argument('--max-epoch', type=int, default=100)
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
    parser.add_argument('--loss', type=str, default='ARPLoss')
    parser.add_argument('--eval', action='store_true', help="Eval", default=False)
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    options = vars(args)

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

    Data = Product_Dataloader(train_dataroot=options['train_dataroot'], val_dataroot=options['val_dataroot'],
                              unknown_dataroot=options['unknown_dataroot'], batch_size=options['batch_size'],
                              img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    if options['cs']:
        net = classifier32ABN(num_classes=options['num_classes'])
    else:
        net = classifier32(num_classes=options['num_classes'])
    feat_dim = 128

    if options['cs']:
        print("Creating GAN")
        nz, ns = options['nz'], 1

        netG = gan.Generator32(1, nz, 64, 3)
        netD = gan.Discriminator32(1, 3, 64)

        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )

    Loss = importlib.import_module('zzClassifier.losses.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['cs'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                                results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]

    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['cs']:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

    start_time = time.time()

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

        if options['cs']:
            train_cs(net, netD, netG, criterion, criterionD,
                     optimizer, optimizerD, optimizerG,
                     trainloader, epoch=epoch, **options)

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                                    results['OSCR']))

            save_networks(net, model_path, file_name, criterion=criterion)

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    main()