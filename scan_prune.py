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

def plot_curve(points, acc, name = '3D_curve', savePath='figures_autopath/3d_PCA', mode=3):
    points = np.array(points)
    print(points.shape)
    print('acc len:', len(acc))
    print(acc[:100])

    fig = plt.figure()
    match mode:
        case 3:
            ax = fig.add_subplot(projection='3d')
        case 2:
            ax = fig.add_subplot()
        case 1:
            ax = fig.add_subplot()

    # color = np.linspace(0, points.shape[0] - 1, points.shape[0], dtype=float)
    cblue = np.array([.0, .0, 255.0])
    cred = np.array([255.0, 0.0, 0.0])

    def make_color(x):
        return '#%02x%02x%02x' % tuple(int(i) for i in x)

    col = copy.deepcopy(acc)

    for p, i in enumerate(acc):
        col[p] = make_color(lerp(cred, cblue, i / 100.0))

    # x, y, z = points.T

    #ax.scatter(
        #x,
        #y,
        #z,
        #c=color,
        #s=50,
    #)
    # ax.plot(
    #     x, y, z,
    #     #c=color,
    # )
    match mode:
        case 3 | 2:
            ax.scatter(*points.T, c=acc,s=[10+i*10 for i in range(len(acc))],)
        case 1:
            ax.scatter(*points.T, c=acc,s=[10+i*10 for i in range(len(acc))],)
    # plt.scatter(*points.T, c=acc,s=[10+i*10 for i in range(len(acc))],)

    # for i in range(points.shape[0]):
    #     ax.text(*points[i],s=str(i))

    # ax.set_zlabel('Z Label')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')

    # plt.show()
    #fig.savefig('./3d.png')
    fig.savefig('/'.join(['.', savePath, name + '_3d_PCA.png']))
    
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
    network_type = 'vgg9'
    net = load(network_type)
    mode = 1

    wra = construct_wrapper(net)

    fi = open('logs.txt', mode='a')
    def pin(x: str):
        logger.info(x)
        fi.write(x + '\t' + str(datetime.now()) + '\n')
        fi.flush()

    mng = mp.Manager()
    q1 = mng.Queue(maxsize=4)
    q2 = mng.Queue()
    
    # 进程池
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
            args=(network_type, q1, q2), kwargs={
                # 'dataset': 'mnist', 'raw_data': True
            }
        ) for _ in range(4)
    ]

    # proj = ('lenet1', 'lenet_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    proj = ('tn09', 'vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    projdir = partial(cat, *proj)

    t7 = torch.load(projdir('model_300.t7'))
    net.load_state_dict(t7['state_dict'])
    accli = []

    from net_plotter import create_random_direction, set_weights

    paramap = {}
    for k, v in net.named_modules():
        if isinstance(v, nn.modules.conv._ConvNd):
            paramap[k + '.bias'] = paramap[k + '.weight'] = apply(v.weight, 0.4)
        elif isinstance(v, nn.Linear):
            paramap[k + '.bias'] = paramap[k + '.weight'] = apply(v.weight, 1.0)

            # 把偏差也干掉

    pin(f'Seed: {torch.seed()}')
    # dx = create_random_direction(net)
    # dy = create_random_direction(net)

    # torch.save(dx, projdir('x.direction'))
    # torch.save(dy, projdir('y.direction'))
    dx = torch.load(projdir('x.direction'))
    dy = torch.load(projdir('y.direction'))

    for ind, (name, param) in enumerate(net.named_parameters()):
        if name in paramap:
            for k in paramap[name]:
                dx[ind][k].mul_(0)
                dy[ind][k].mul_(0) # 冻住“不重要的参数”
        # if name in paramap:
        #     for k in range(len(dx[ind])):
        #         if k not in paramap[name]:
        #             dx[ind][k].mul_(0)
        #             dy[ind][k].mul_(0) # 反选
        # else:
            # dx[ind].mul_(0)
            # dy[ind].mul_(0)

    for x in consumers:
        x.start()

    resolution = 10

    needle = copy.deepcopy(net)

    tasksiz = 0

    for x in range(-resolution, resolution + 1):
        for y in range(-resolution, resolution + 1):
            set_weights(needle, get_weights(net), (dx, dy), (x, y))
            tasksiz += 1
            q1.put(((x, y), copy.deepcopy(needle.state_dict())))
    accli = {}

    for _ in consumers:
        q1.put(None)
    for x in consumers:
        x.join()

    for _ in range(tasksiz):
        x, *res = q2.get()
        accli[x] = res

    torch.save(accli, projdir('acc_line.dict'))
    # acc.extend(accli)
    # pin(f"{cat(scandir, k)}: {accli}")


    # coords = np.array(coords)

    # for i in coords:
    #     i[:] -= coords[-1]
    # print('PCA...')

    # if mode != 1:
    #     pca = PCA(n_components=mode)
    #     pca.fit(coords)
    #     ax = pca.components_[:mode]
    # else:
    #     dire = coords[-1] - coords[0]
    #     dire = dire / np.sqrt(np.sum(dire**2))
    #     ax = [dire]

    # acc = [x[3] for x in acc]

    # pos = []

    # for p, i in enumerate(coords):
    #     if mode == 1:
    #         pos.append([i.dot(x) for x in ax] + [acc[p]])
    #     else:
    #         pos.append([i.dot(x) for x in ax])

            


    # print('plotting...')
    # print(pos, acc)
    # plot_curve(pos, acc, k, savePath=f"figures_autopath/{mode}d", mode=mode)
    
    

    # coords = torch.tensor(coords)
    # torch.save(coords, 'vgg9_all11.pkl')