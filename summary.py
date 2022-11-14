import itertools
import json
import torch
import sys
import argparse


from cifar10.model_loader import *
import dataloader
import evaluation
from torchsummary import summary
import datetime
from loguru import logger



def load_preferrences(netname: str):
    if netname == 'vgg9':
        return [21, 49]
    elif netname == 'resnet56':
        return [21, 49]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--file', '-f', default='T1.json', help='result file name')
    parser.add_argument('--begin', '-b', default='0:0:0', help='begin args')
    parser.add_argument('--end', '-e', default='7:0:0', help='end args')

    parser.add_argument('--model', '-m', default='vgg9', help='model type')
    parser.add_argument('--model_folder', '-mf', default='', help='model folder')
    # parser.add_argument('--model', '-m', default='resnet56', help='model type')

    parser.add_argument('--model1', default='tn10/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 1')
    parser.add_argument('--model2', default='tn09/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 2')

    # parser.add_argument('--model1', default='R56_01/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 1')
    # parser.add_argument('--model2', default='R56_02/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_300.t7', help='model 2')

    args = parser.parse_args()

    f = open(args.file, mode='a')
    def pin(x):
        logger.info(x)
        f.write(x+'\n')
        f.flush()


    # logger.add(args.file, mode='a', serialize=True)
    c = torch.load(args.model1)
    cd = c['state_dict']

    c2 = torch.load(args.model2)
    cd2 = c2['state_dict']

    net = load(args.model)
    res = net.load_state_dict(cd)
    print(res)
    net.eval()
    # for param in net.parameters():
    #     print(param.shape, param.size())

    # for k, v in net.named_parameters():
        # print(k, v.size())
        # v.data = cd2[k]
        
    print('=======')
    # for k, v in net.named_buffers():
        # print(k, v.size())
    #     la.append(k)
    #     lb.append(v.data)
        # v.data = cd2[k]
    # print(net.named_parameters())

    trainloader, testloader = dataloader.load_dataset(
        'cifar10', 
        'cifar10/data',
        128,
        2,
        False,
        1, 
        0,
        '', 
        '')

    cnt = 6
    ks = list(cd.keys())
    k1 = ks[:49]
    k2 = ks[49:56]
    k3 = ks[56:]
    # k1 = ks[:228]
    # k2 = ks[228:342]
    # k3 = ks[342:]

    def lerp(A, B, t): # 0 -> A, 1 -> B
        ret = A + (B - A) * t
        # assert ret.shape == A.shape == B.shape
        return ret
    
    pin(json.dumps({
        'model1': args.model1,
        'model2': args.model2,
    }))

    begins = [int(x) for x in args.begin.split(':')]
    ends = [int(x) for x in args.end.split(':')]
    for i1 in range(cnt+1):
        for i2 in range(cnt+1):
            for i3 in range(cnt+1):
                if [i1, i2, i3] < begins: continue
                if [i1, i2, i3] >= ends: break

                for kk, vv in itertools.chain(net.named_parameters(), net.named_buffers()):
                    if kk in k1: vv.data = lerp(cd[kk], cd2[kk], i1/cnt)
                    if kk in k2: vv.data = lerp(cd[kk], cd2[kk], i2/cnt)
                    if kk in k3: vv.data = lerp(cd[kk], cd2[kk], i3/cnt)
                t1 = datetime.datetime.now()
                train_loss, train_acc = evaluation.eval_loss(net, torch.nn.CrossEntropyLoss(),trainloader,True, 'train')
                test_loss, test_acc = evaluation.eval_loss(net, torch.nn.CrossEntropyLoss(),testloader,True,'test')
                t2 = datetime.datetime.now()
                print('time used:', (t2-t1).total_seconds())
                print(i1, i2, i3, train_acc, test_acc)
                logs = json.dumps({
                    'dimension': f'{i1}:{i2}:{i3}', 
                    'train_loss': train_loss,
                    'train_acc': train_acc, 
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                })
                pin(logs)
            net.clear_ckp(1, 'train')
            net.clear_ckp(1, 'test')
        net.clear_ckp(0, 'train')
        net.clear_ckp(0, 'test')
    f.close()



    # print('time used:', (t2-t1).total_seconds())
    # print(a,b)


# summary(net, (3, 32, 32))
# summary(load('resnet18'), (3, 32, 32))
# summary(load('resnet56'), (3, 32, 32))
# summary(load('densenet121'), (3, 32, 32))
