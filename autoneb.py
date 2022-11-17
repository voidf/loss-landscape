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
from fill import highest
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
    net_container.model.batch_size = 512

    
    trainloader, testloader = dataloader.load_dataset()
    net.load_state_dict(torch.load(args.model1)['state_dict'])
    coord1 = net_container.get_coords()

    net.load_state_dict(torch.load(args.model2)['state_dict'])
    coord2 = net_container.get_coords()

    assert (coord1 == coord2).sum() != coord1.shape[0]
    print((coord1 == coord2).sum(), '/', coord1.shape)
    k = float('inf')

    neb_configs = [
        (equal, 3),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
        (highest, 2, 'dense_train_loss'),
    ]

    optim_configs = [
        (0.1, 30),
        (0.1, 1000),
        (0.1, 1000),
        (0.1, 1000),
        (0.1, 2000),
        (0.1, 2000),
        (0.01, 1000),
        (0.01, 1000),
        (0.01, 1000),
        (0.01, 1000),
        (0.001, 1000),
        (0.001, 1000),
        (0.001, 1000),
        (0.001, 1000),
    ]


    
    # sch = StepLR(op, 390 * 30, 0.1, -1)

    mxl = float('inf')
    true_mxl = float('inf')
    save_cnt = 0

    fdir = f'autopaths08_07_k={k}'
    if not os.path.exists(fdir):
        os.mkdir(fdir)

    res_dict = init_res_dict(coord1, coord2)

    for p, (nc, oc) in enumerate(zip(neb_configs, optim_configs)):
        t1, t2 = nc[0](res_dict, *nc[1:])
        res_dict.update({'path_coords': t1, 'target_distances': t2})
        print(f'Phaze {p}/{len(neb_configs)}')
        neb = NEB(net_container, res_dict['path_coords'], res_dict['target_distances'], spring_constant=k)
        neb.path_coords.requires_grad_(True)
        op = SGD(neb.parameters(), lr=oc[0], momentum=0.9)
        neb.path_coords.requires_grad_(False)

        for _ in pbar(range(oc[1]), 'NEB'):
            l = neb.apply(True)
            op.step()
            if l < mxl * 0.9:
                torch.save(neb.path_coords, cat(fdir, f'path{save_cnt}.pkl'))
                save_cnt += 1
                mxl = l
                pin(f"step {_}, loss: {l}")
            if l < true_mxl:
                true_mxl = l
                torch.save(neb.path_coords, cat(fdir, f'min.pkl'))
        res_dict.update({"path_coords": neb.path_coords.clone().to("cpu")})
        analysis = neb.analyse(9)
        saddle_analysis = {key: value for key, value in analysis.items() if "saddle_" in key}
        logger.debug(f"Found saddle: {saddle_analysis}.")
        res_dict.update(analysis)



    # torch.save(neb.path_coords, f'paths2/neb_path_final.pkl')


    # pin(json.dumps({
    #     'model1': args.model1,
    #     'model2': args.model2,
    #     'method': 'neb',
    # }))



    f.close()



    # print('time used:', (t2-t1).total_seconds())
    # print(a,b)
