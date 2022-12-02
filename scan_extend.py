from datetime import datetime
from functools import partial
import traceback
from typing import Sequence
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import copy
from loguru import logger

from cifar10.model_loader import *
from dataloader import load_dataset
from net_plotter import get_weights
from wrappers import *
import torch.multiprocessing as mp
import evaluation
import re
    
# https://github.com/VainF/Torch-Pruning/issues/49 by @Serjio42
def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    n_remain = round_to*max(int(total_parameters - n_to_prune)//round_to, 1)
    return max(total_parameters - n_remain, 0)

def apply(weights, amount=0.0, round_to=1, p=1)->  Sequence[int]:  # return index
    if amount<=0: return []
    n = len(weights)
    l1_norm = torch.norm( weights.view(n, -1), p=p, dim=1 )
    n_to_prune = int(amount*n) if isinstance(amount, float) else amount 
    n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
    if n_to_prune == 0: return []
    threshold = torch.kthvalue(l1_norm, k=n_to_prune).values 
    indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
    return indices

if __name__ == '__main__':
    # network_type = 'lenet'
    # network_type = 'vgg9'
    network_type = 'resnet56'
    # network_type = 'resnet56_noshort'

    fn = f'acc_line_0.1x10.dict'
    # proj = ('lenet1', 'lenet_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    # proj = ('tn09', 'vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    proj = ('R56_01', 'resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    # proj = ('R56N_04', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    # proj = ('R56N_DL',)
    projdir = partial(cat, *proj)
    amountdir = f'amount=0.4'

    mo, scale, resolution = re.match(r'acc_([a-z]*?)_([0-9\.]*?)x([0-9]*?).dict', fn).groups()
    scale = float(scale)
    resolution = int(resolution)

    amount = re.match(r'amount=([0-9\.]*?)$', amountdir).groups()[0]
    amount = float(amount)
    accli: dict = torch.load(projdir(amountdir, fn))
    acclibak = copy.deepcopy(accli)


    net = load(network_type)
    workers = 5
    fi = open('logs.txt', mode='a')

    def pin(x: str):
        logger.info(x)
        fi.write(x + '\t' + str(datetime.now()) + '\n')
        fi.flush()

    mng = mp.Manager()
    q1 = mng.Queue(maxsize=workers+1)
    q2 = mng.Queue()
    
    # 进程池
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
            args=(network_type, q1, q2), kwargs={
                # 'dataset': 'mnist', 'raw_data': True
            }
        ) for _ in range(workers)
    ]



    t7 = torch.load(projdir('model_300.t7'))
    net.load_state_dict(t7['state_dict'])

    from net_plotter import create_random_direction, set_weights

    paramap = {}
    if mo != 'no':
        for k, v in net.named_modules():
            if isinstance(v, nn.modules.conv._ConvNd):
                paramap[k + '.bias'] = paramap[k + '.weight'] = apply(v.weight, amount)
            elif mo == 'line' and isinstance(v, nn.Linear):
                paramap[k + '.bias'] = paramap[k + '.weight'] = apply(v.weight, 1.0)

            # 把偏差也干掉

    pin(f'Seed: {torch.seed()}')
    if not os.path.exists(projdir('x.direction')):
        dx = create_random_direction(net)
        dy = create_random_direction(net)

        torch.save(dx, projdir('x.direction'))
        torch.save(dy, projdir('y.direction'))
    else:
        dx = torch.load(projdir('x.direction'))
        dy = torch.load(projdir('y.direction'))


    if mo == 'rev':
        for ind, (name, param) in enumerate(net.named_parameters()):
            if name in paramap:
                for k in range(len(dx[ind])):
                    if k not in paramap[name]:
                        dx[ind][k].mul_(0)
                        dy[ind][k].mul_(0) # 反选
            else:
                dx[ind].mul_(0)
                dy[ind].mul_(0)
    else:
        for ind, (name, param) in enumerate(net.named_parameters()):
            if name in paramap:
                for k in paramap[name]:
                    dx[ind][k].mul_(0)
                    dy[ind][k].mul_(0) # 冻住“不重要的参数”

    for x in consumers:
        x.start()



    needle = copy.deepcopy(net)

    tasksiz = 0

    todo = list(accli.keys())
    while todo:
        cx, cy = todo.pop()
        for X, Y in zip([0, 1, 0, -1], [1, 0, -1, 0]):
            x = cx + X
            y = cy + Y
            if (x, y) not in accli:
                set_weights(needle, get_weights(net), (dx, dy), (x * scale, y * scale))
                tasksiz+=1
                q1.put(((x, y), copy.deepcopy(needle.state_dict())))

    # for x in range(-resolution, resolution + 1):
    #     for y in range(-resolution, resolution + 1):
    #         set_weights(needle, get_weights(net), (dx, dy), (x * scale, y * scale))
    #         tasksiz += 1
    #         q1.put(((x, y), copy.deepcopy(needle.state_dict())))
    # accli = {}

    for _ in consumers:
        q1.put(None)
    for x in consumers:
        x.join()

    for _ in range(tasksiz):
        x, *res = q2.get()
        accli[x] = res

    if not os.path.exists(projdir(f'amount={amount}')):
        os.mkdir(projdir(f'amount={amount}'))
    assert (accli.keys() & acclibak.keys()) == acclibak.keys()
    torch.save(accli, projdir(f'amount={amount}', fn))
    