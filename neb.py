from functools import reduce
import itertools
import json
import operator
import numpy as np
import torch
import sys
import argparse

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from typing import Union, Dict

from cifar10.model_loader import *
import dataloader
import evaluation
from torchsummary import summary
import datetime
from loguru import logger
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, NLLLoss, Parameter
from torch import nn
from math import sqrt
from typing import Iterable, Dict, Union
from torch.optim.lr_scheduler import StepLR
from wrappers import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--file', '-f', default='N1.json', help='result file name')
    # parser.add_argument('--begin', '-b', default='0:0:0', help='begin args')
    # parser.add_argument('--end', '-e', default='7:0:0', help='end args')
    parser.add_argument('-k', type=int, default=5, help='elastic constant')
    parser.add_argument('--npivots', '-n', type=int, default=5, help='number of pivots')

    parser.add_argument('--model', '-m', default='vgg9', help='model type')
    parser.add_argument('--model_folder', '-mf', default='', help='model folder')
    # parser.add_argument('--model', '-m', default='resnet56', help='model type')

    parser.add_argument('--model1', default='tn08/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 1')
    parser.add_argument('--model2', default='tn07/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 2')




    # parser.add_argument('--model1', default='R56_01/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 1')
    # parser.add_argument('--model2', default='R56_02/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 2')

    args = parser.parse_args()

    f = open(args.file, mode='a')
    def pin(x):
        logger.info(x)
        f.write(x+'\n')
        f.flush()

    net = load(args.model)

    net_container = construct_wrapper(net)

    net_container.to('cuda')

    
    trainloader, testloader = dataloader.load_dataset()
    net.load_state_dict(torch.load(args.model1)['state_dict'])
    coord1 = net_container.get_coords()
    # trloss, tracc = evaluation.eval_loss(net_container.model.model.model, CrossEntropyLoss(), trainloader)
    # print(trloss,tracc)
    # trloss, tracc = evaluation.eval_loss(net_container.model.model.model, CrossEntropyLoss(), testloader)
    # print(trloss,tracc)
    # n2 = construct_wrapper

    net.load_state_dict(torch.load(args.model2)['state_dict'])
    coord2 = net_container.get_coords()
    # trloss, tracc = evaluation.eval_loss(net_container.model.model.model, CrossEntropyLoss(), trainloader)
    # print(trloss,tracc)
    # trloss, tracc = evaluation.eval_loss(net_container.model.model.model, CrossEntropyLoss(), testloader)
    # print(trloss,tracc)
    # exit(0)
    assert (coord1 == coord2).sum() != coord1.shape[0]
    print((coord1 == coord2).sum(), '/', coord1.shape)
    k = -0.1

    neb = NEB(net_container, coord1, coord2, spring_constant=k)
    # for i in neb.path_coords:
        # eval_coord(net_container,i,)
    # exit(0)
    neb.path_coords.requires_grad_(True)

    op = SGD(neb.parameters(), lr=0.04, momentum=0.9)
    neb.path_coords.requires_grad_(False)
    sch = StepLR(op, 390 * 30, 0.1, -1)

    mxl = float('inf')
    true_mxl = float('inf')
    save_cnt = 0

    fdir = f'paths08_07_k={k}'
    if not os.path.exists(fdir):
        os.mkdir(fdir)

    ms = torch.load(cat('paths08_07_k=0A', 'neb_path3.pkl'))
    for mi in ms:
        eval_coord(net_container, mi)
    # eval_coord(net_container, [0])
    # eval_coord(net_container, torch.load(cat('paths08_07_k=infA', 'neb_path0.pkl'))[-1])
    exit(0)


    
    for _ in pbar(range(390 * 60), "NEB"):
        l = neb.apply(True)
        op.step()
        sch.step()
        eval_coord(net_container, neb.path_coords[-1])
        exit(0)
        if l < mxl * 0.9:
            torch.save(neb.path_coords, cat(fdir, f'neb_path{save_cnt}.pkl'))
            save_cnt += 1
            mxl = l
            pin(f"step {_}, loss: {l}")
        if l < true_mxl:
            true_mxl = l
            torch.save(neb.path_coords, cat(fdir, f'minpath.pkl'))


    # torch.save(neb.path_coords, f'paths2/neb_path_final.pkl')


    # pin(json.dumps({
    #     'model1': args.model1,
    #     'model2': args.model2,
    #     'method': 'neb',
    # }))



    f.close()



    # print('time used:', (t2-t1).total_seconds())
    # print(a,b)
