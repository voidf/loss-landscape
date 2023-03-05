import argparse
from datetime import datetime
from functools import partial
from itertools import chain
import json
from math import prod
import random
import shutil
import traceback
from typing import List, Sequence
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import copy
from loguru import logger
from torch import optim
from cifar10 import dataloader

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


def scan(
    arch = 'resnet56',
    from_epoch = 20,
    to_epoch = 300,
    lr = 0.0001,
    mom = 0.9,
    wd = 0.0005,
    bs = 128,
    opt = 'sgd',
    seed = 29,
    base_directory = 'trained/R56_09/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_20B2'
):
    # parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--dataset', default='cifar10', help='dataset type [cifar10/mnist]')
    # args = parser.parse_args()

    # network_type = 'lenet'
    # network_type = 'vgg9'
    # network_type = 'resnet56_noshort'



    # from_epoch = 20
    # to_epoch = from_epoch + 300
    # lr = 0.0001
    # mom = 0.9
    # wd = 0.0005
    # bs = 128
    # opt = 'sgd'
    # seed = 29

    criterion = nn.CrossEntropyLoss()

    # proj = ('lenet1', 'lenet_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    # proj = ('tn09', 'vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', 'model_100B3')
    # proj = ('R56_09', 'resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', 'model_10B5', )
    # proj = ('R56N_04', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')
    # proj = ('R56N_DL',)
    projdir = partial(cat, base_directory)

    # mo, scale, resolution = re.match(r'acc_([a-z]*?)_([0-9\.]*?)x([0-9]*?).dict', fn).groups()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = load(arch)
    fi = open('logs.txt', mode='a')

    def pin(x: str):
        logger.info(x)
        fi.write(x + '\t' + str(datetime.now()) + '\n')
        fi.flush()

    pin(f'Seed: {seed}')



    fn = f'model_{from_epoch}'
    
    # t7 = torch.load(projdir(fn + '.t7'))
    
    with open(projdir(fn + '.json'), 'r') as fil:
        t7 = json.load(fil)

    net.cuda()
    from safetensors import safe_open

    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    with safe_open(projdir(fn + '.safetensors'), framework="pt", device="cuda:0") as fil:
        write_weights(net, fil.get_tensor('param'))
        write_buf_no_nbt(net, fil.get_tensor('buf'))
        write_nbt(net, fil.get_tensor('nbt'))
        
    # net.load_state_dict(t7['state_dict'])

    criterion = criterion.cuda()

    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NameError("Unknown optimizer name")

    if t7.get('optimizer', '') == opt:
        ot7 = torch.load(projdir(f'opt_state_{from_epoch}.t7'))
        optimizer.load_state_dict(ot7['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    if not os.path.exists(projdir(f'model_{from_epoch + 1}.safetensors')):
        branch_dir = '.'
    else:
        idx = 1
        s = set() # MEX
        for i in os.listdir(projdir()):
            if os.path.isdir(projdir(i)) and i.startswith(fn + 'B'):
                if len(os.listdir(projdir(i))) == 0:
                    os.rmdir(projdir(i))
                else:
                    s.add(int(i.rsplit('B', 1)[1]))
        while idx in s: idx += 1

        branch_dir = f'{fn}B{idx}'
        os.mkdir(projdir(branch_dir))
        print(f'make dir {projdir(branch_dir)}')


    trainloader, testloader = dataloader.c10()


    from main import train, test
    from safetensors.torch import save_file


    for e in range(from_epoch + 1, to_epoch + 1):
        loss, train_err = train(trainloader, net, criterion, optimizer, True)
        test_loss, test_err = test(testloader, net, criterion, True)

        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (e, loss, train_err, test_err, test_loss)
        pin(status)

        # Save checkpoint.
        acc = 100. - test_err
        state = {
            'tea': acc,
            'tel': test_loss,
            'tra': 100. - train_err,
            'trl': loss,
            'batch_size': bs,
            'random_seed': seed,
            'epoch': e,
            'optimizer': opt,
            'lr': lr,
            'momentum': mom,
            'weight_decay': wd,
            # 'state_dict': net.state_dict(),
        }
        save_file({
            'param': cat_tensor(get_weights(net)),
            'buf': cat_tensor(get_buf_no_nbt(net)),
            'nbt': cat_tensor(get_nbt(net))
            }, projdir(branch_dir, f'model_{e}.safetensors'))
        with open(projdir(branch_dir, f'model_{e}.json'), 'w') as fil:
            json.dump(state, fil)
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        # torch.save(state, projdir(branch_dir, f'model_{e}.t7'))
        torch.save(opt_state, projdir(branch_dir, f'opt_state_{e}.t7'))
        # if e == from_epoch + 10:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.1
        #     lr = 0.1

        # elif e == from_epoch + 150 or e == from_epoch + 225 or e == from_epoch + 275:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        #     lr *= 0.1



    # if not os.path.exists(projdir(f'amount={amount}')):
    #     os.mkdir(projdir(f'amount={amount}'))
    # assert (accli.keys() & acclibak.keys()) == acclibak.keys()
    # torch.save(accli, projdir(f'amount={amount}', fn))

def iterate_params_buffers(net: nn.Module, skip_num_batches_tracked=True):
    offset = 0
    for param in net.parameters():
        size = reduce(operator.mul, param.data.shape)
        data = param
        yield offset, data.data, size, False
        offset += size
    for buffer in net.buffers():
        if len(buffer.shape) == 0:
            if skip_num_batches_tracked:
                continue # 跳过num_batches_tracked
            # else:

        size = prod(buffer.shape)
        yield offset, buffer, size, True
        offset += size

def iterate_params(net: nn.Module):
    offset = 0
    for param in net.parameters():
        size = reduce(operator.mul, param.data.shape)
        data = param
        yield offset, data.data, size, False
        offset += size

def get_states(net: nn.Module, skip_num_batches_tracked=True): 
    # for n, p in chain(net.named_parameters(), net.named_buffers()):
        # if len(p.shape) == 0:
            # print(n, p.shape)
    return [p.data for n, p in chain(net.named_parameters(), net.named_buffers()) if (not n.endswith('num_batches_tracked') or not skip_num_batches_tracked)]

def get_buf_no_nbt(net: nn.Module):
    return [p.data for n, p in net.named_buffers() if not n.endswith('num_batches_tracked')]

def write_buf_no_nbt(net: nn.Module, ts: Tensor):
    sz = 0
    for k, v in net.named_buffers():
        if not k.endswith('num_batches_tracked'):
            l = prod(v.shape)
            v.data = ts[sz:sz+l].reshape(v.shape)
            sz += l

def get_nbt(net: nn.Module):
    return [p.data for n, p in net.named_buffers() if n.endswith('num_batches_tracked')]

def write_nbt(net: nn.Module, ts: Tensor):
    ctr = 0
    for k, v in net.named_buffers():
        if k.endswith('num_batches_tracked'):
            v.fill_(ts[ctr])
            ctr += 1

def cat_tensor(li: List[Tensor]) -> Tensor:
    # return torch.cat(li)
    sz = 0
    for p in li:
        sz += prod(p.data.shape)
    buf = torch.empty(sz, dtype=torch.float32)
    o = 0
    for p in li:
        sz = prod(p.data.shape)
        buf[o:o+sz] = p.data.view(-1)
        o += sz
    return buf

def write_weights(net: nn.Module, ts: Tensor):
    for offset, data, size, is_buffer in iterate_params(net):
        # if len(data.shape) == 0:
        #     data.fill_(ts[offset:offset + size][0])
        # else:
        data[:] = ts[offset:offset + size].reshape(data.shape)

def write_states(net: nn.Module, ts: Tensor, skip_num_batch_tracked=True):
    for offset, data, size, is_buffer in iterate_params_buffers(net, skip_num_batch_tracked):
        if len(data.shape) == 0:
            if skip_num_batch_tracked:
                continue
            else:
                data.fill_(ts[offset:offset + size][0])
        else:
            data[:] = ts[offset:offset + size].reshape(data.shape)



def calc():
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', 'model_100B3', )
    projdir = partial(cat, *proj)

    net = load('resnet56_noshort')

    # dx = torch.load(projdir('x.direction'))
    # dy = torch.load(projdir('y.direction'))
    dx = torch.load(r'E:\loss-landscape\R56N_05\resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1\model_10B5\x.direction')
    dy = torch.load(r'E:\loss-landscape\R56N_05\resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1\model_10B5\y.direction')

    dx = cat_tensor(dx)
    dy = cat_tensor(dy)

    target = r'E:\loss-landscape\R56N_05\resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1\model_10B5\model_300.t7'

    O = torch.load(target)
    net.load_state_dict(O['state_dict'])
    del O
    w = cat_tensor(get_weights(net))
    with torch.no_grad():
        w=w.cuda()
        dx=dx.cuda()
        dy=dy.cuda()
        nm = dx.norm() ** 2
        print('cos:', torch.dot(dx, dy)/(dx.norm()*dy.norm()).item())

        # px = w * dx
        # py = w * dy

        nil = []
        for cd in os.listdir(projdir()):
            if cd.startswith('model_') and cd.endswith('.t7'):
                n = cd.removeprefix('model_').removesuffix('.t7')
                n = int(n)
                nil.append(n)
        for n in sorted(nil):
            t7 = torch.load(projdir(f'model_{n}.t7'))
            net.load_state_dict(t7['state_dict'])
            v = cat_tensor(get_weights(net))
            v = v.cuda()
            dist = v - w
            tx = dist.dot(dx) / nm # 在dx的尺度上的比例
            ty = dist.dot(dy) / nm
            print(n, tx, ty, t7['acc'], t7['train_loss'])
            torch.save((tx, ty), projdir(f'model_{n}.proj'))

def draw():
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    proj = ('R56_09', 'resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', 'model_100B3', )
    projdir = partial(cat, *proj)
    # net = load('resnet56_noshort')
    net = load('resnet56')
    with torch.no_grad():
        li = []
        metas = []
        sz = []
        for pi, pf in enumerate(
            (
                '.', 
                'model_10B1',
                'model_20B1',
                'model_30B1',
                'model_40B1',
                'model_50B1',
                'model_60B1',
                'model_70B1',
                'model_80B1',
                'model_90B1',
                'model_100B1',
            )):
        # for pi, pf in enumerate(('.', '..', '../model_10B5')):
        # for pi, pf in enumerate(('.', )):
            nil = []
            for cd in os.listdir(projdir(pf)):
                if cd.startswith('model_') and cd.endswith('.t7'):
                    n = cd.removeprefix('model_').removesuffix('.t7')
                    n = int(n)
                    nil.append(n)

            for ind, n in enumerate(sorted(nil)):
                cd = f'model_{n}.t7'
                t7 = torch.load(projdir(pf, cd))
                net.load_state_dict(t7['state_dict'])
                t7.pop('state_dict')
                v = cat_tensor(get_weights(net)).numpy()
                li.append(v)
                metas.append(t7)
                if ind and t7['lr'] != metas[-2]['lr']:
                    print(t7['lr'], metas[-2]['lr'], pf, n)
                    sz.append(160)
                # elif t7['epoch'] == 275:
                    # sz.append(320)
                else:
                    sz.append(5)
                    # sz.append([5, 20, 80][pi])
        # model = TSNE(3, perplexity=10, n_iter=3000, learning_rate='auto')
        model = PCA(3)
        li = np.array(li)
        model.fit(li)
        axis = model.components_[:3]
        torch.save(axis, 'pca3d.dir')
        pos = np.array([
            [i.dot(j) for j in axis] for i in li
        ])
        del li
        torch.save(pos, 'pca.proj3d')
        with open('pca.proj3d.json', 'w') as fil:
            json.dump({'d': pos.tolist()}, fil)
        # pos = model.fit_transform(np.array(li))

        import matplotlib.pyplot as plt
        from wrappers import lerp

        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        # fig, ax = plt.subplots()

        cblue = np.array([.0, .0, 0.0])
        cred = np.array([255.0, 0.0, 0.0])

        def make_color(x):
            return '#%02x%02x%02x' % tuple(int(i) for i in x)
        
        mx = max(i['acc'] for i in metas)
        mn = min(i['acc'] for i in metas)
        

        ax.scatter(*pos.T, c=[make_color(lerp(cred, cblue, (i['acc'] - mn) / (mx - mn))) for i in metas], s=sz)
        plt.show()
    
def info():
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', 'model_10B5', )
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    # proj = ('R56N_05', 'resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    # proj = ('R56_09', 'resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', )
    proj = ('R56_09', 'resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1', 'model_10B1')

    projdir = partial(cat, *proj)

    f = open(projdir('info.json'), 'a')

    with torch.no_grad():
        nil = []
        for cd in os.listdir(projdir()):
            if cd.startswith('model_') and cd.endswith('.t7'):
                n = cd.removeprefix('model_').removesuffix('.t7')
                n = int(n)
                nil.append(n)
        for n in sorted(nil):
            cd = f'model_{n}.t7'
            t7 = torch.load(projdir(cd))
            t7.pop('state_dict')
            f.write(json.dumps(t7)+'\n')



        
    

if __name__ == '__main__':
    # scan()
    # calc()
    draw()
    # info()