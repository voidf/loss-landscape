"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

from typing import Any, Mapping, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torch.autograd.variable import Variable
import datetime

import torch.multiprocessing as mp
from multiprocessing import Queue
from cifar10.model_loader import load

from dataloader import load_dataset
def err_print_exc(e):
    with open('exc.txt', 'a') as f:
        f.write(str(e)+'\n')

def batch_dispatcher(net: nn.Module, tasks: mp.Queue, dist: mp.Queue):
    """批次并发，不可用，计算结果不对，速度下降"""
    # print('pid:', os.getpid())
    criterion = nn.CrossEntropyLoss()
    net.cuda()
    net.eval()

    with torch.no_grad():
        while 1:
            batch, targets = tasks.get()
            # batch, targets = torch.tensor(batch).cuda(), torch.tensor(targets).cuda()
            if batch is None:
                return
            batch, targets = batch.cuda(), targets.cuda()
            outputs: torch.Tensor = net(batch)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets).sum().item()
            dist.put((loss.item(), correct))
        
def epoch_dispatcher(net: nn.Module, q: mp.Queue, rank: int):
    """单点并发，可用"""
    # with torch.no_grad():
    # net.cuda()
    # net.eval()
    criterion = nn.CrossEntropyLoss()
    trainloader, testloader = load_dataset(threads=1)
    l1, a1 = eval_loss(net, criterion, trainloader)
    l2, a2 = eval_loss(net, criterion, testloader)
    q.put((rank, l1, a1, l2, a2))
    # return 

        # for batch_idx, (inputs, targets) in enumerate(trainloader):
def epoch_consumer(network_type: str, tasks: mp.Queue, results: mp.Queue, cuda_device=0, **datasetkws):
    """消费者侧并发，建议使用"""
    net = load(network_type)
    net.cuda(cuda_device)
    trainloader, testloader = load_dataset(threads=1, **datasetkws)
    criterion = nn.CrossEntropyLoss()
    while 1:
        task: Tuple[int, Mapping[str, Any]] = tasks.get()
        # rank, state_dict
        if task is None:
            return
        
        net.load_state_dict(task[1], strict=True)
        l1, a1 = eval_loss(net, criterion, trainloader)
        l2, a2 = eval_loss(net, criterion, testloader)
        results.put((task[0], l1, a1, l2, a2))


def eval_loss(net, criterion, loader, use_cuda=True, pool_size=2):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    t = datetime.datetime.now()

    with torch.no_grad():
        correct = 0
        total_loss = 0
        total = 0 # number of samples
        num_batch = len(loader)

        # if use_cuda:
        net.cuda()
        net.eval()

        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                # if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs: torch.Tensor
                outputs = net(inputs)
                # inputs = inputs.detach()
                # targets = targets.detach()
                # tasks.put((inputs, targets))
            # for x in range(pool_size):
                # tasks.put((None, None))
            # for proc in results:
                # proc.join()

            # pool.close()
            # pool.join()
            # tasks.join()
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                acc = predicted.eq(targets).sum().item()
                correct += acc
            

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    print(f'correct: {correct}/{total}, time:', (datetime.datetime.now()-t).total_seconds())
    return (total_loss/total).item(), 100.*correct/total
