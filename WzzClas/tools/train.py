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

import argparse
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append("../")

from zzClassifier.datasets import Product_Dataloader_Close, Product_Dataloader_Open
from zzClassifier.losses import rzloss
from zzClassifier.models import gan
from zzClassifier.models.resnet import resnet18
from zzClassifier.models.models import classifier32, classifier32ABN
from zzClassifier.core import train, train_cs, test, build_optimizer
from zzClassifier.core.model_builder import build_model
from utils import Logger, save_networks, load_networks

# Dataset
def get_args():
    parser = argparse.ArgumentParser("Training")

    # dataset
    parser.add_argument('--dataset', type=str, default='product')
    parser.add_argument('--train_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/train')
    parser.add_argument('--val_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/val')
    parser.add_argument('--unknown_train_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/unknow_train')
    parser.add_argument('--unknown_val_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/unknow_val')

    parser.add_argument('--outf', type=str, default='./log')
    parser.add_argument('--out-num', type=int, default=50, )

    # optimization
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
    parser.add_argument('--optimizer', type=str, default='SGD', help="optimizer")
    parser.add_argument('--decay_step', type=float, default=2, help="LEARNING_DECAY_STEP")
    parser.add_argument('--decay_gamma', type=float, default=0.9, help="LEARNING_DEACAY_GAMMA")
    parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--stepsize', type=int, default=30)
    parser.add_argument('--temp', type=float, default=1.0, help="temp")
    parser.add_argument('--num-centers', type=int, default=1)

    # model
    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
    parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
    parser.add_argument('--model_name', type=str, default='RejectModel')
    parser.add_argument('--backbone', type=str, default='resnet18')

    # misc
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ns', type=int, default=1)
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--eval_period', type=int, default=2)
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

    # load dataset
    print("{} Preparation".format(options['dataset']))

    Data = Product_Dataloader_Open(train_dataroot=options['train_dataroot'], val_dataroot=options['val_dataroot'],
                              unknown_train_dataroot=options['unknown_train_dataroot'], unknown_val_dataroot=options['unknown_val_dataroot'],
                              batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, unknow_trainloader, unknow_valloader = Data.train_loader, Data.test_loader, Data.unknow_train_loader, Data.unknow_val_loader

    options['num_classes'] = Data.num_classes

    # initial model
    print("Creating model: {}".format(options['model_name']))
    net = build_model(options)

    # initial optimizer
    optimizer, scheduler = build_optimizer(options, net)

    if options['model_name'] == 'RejectModel':
        loss = rzloss(margin=0.25, gamma=80)
    else:
        loss = nn.CrossEntropyLoss()

    # training model
    best_score = 0
    for epoch in range(options['epoch']):
        model = trainer.train_epoch(model, optimizer, loss, epoch, options)

        # validation
        if epoch % options['eval_period'] == 0:
            val_acc, _, _, _ = validator(
                model, optimizer, loss, epoch, best_score, options)

            # update the best validation accuracy
            if val_acc > best_score:
                best_score = val_acc

            # save weight

if __name__ == '__main__':
    main()