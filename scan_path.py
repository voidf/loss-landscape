import traceback
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import copy
from loguru import logger

from cifar10.model_loader import *
from dataloader import load_dataset
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
    

if __name__ == '__main__':
    # net = load('vgg9')
    network_type = 'resnet56'
    net = load(network_type)
    mode = 1

    wra = construct_wrapper(net)
    # wra.to('cuda') # 关键

    # ms = torch.load(cat('paths08_07_k=0A', 'neb_path3.pkl'))
    # for mi in ms:
    #     eval_coord(wra, mi)

    

    # scandir = 'paths08_07_k=0A'
    # scandir = 'autopaths08_07_k=inf'
    scandir = 'autopaths_resnet56_01_02_k=inf'

    fi = open('logs.txt', mode='a')
    def pin(x: str):
        logger.info(x)
        fi.write(x + '\n')
        fi.flush()

    trainloader, testloader = load_dataset()
    mng = mp.Manager()
    q1 = mng.Queue()
    q2 = mng.Queue()
    
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
            args=(network_type, q1, q2)
        ) for _ in range(4)
    ]
    for x in consumers:
        x.start()

    import os
    for k in os.listdir(scandir):
        if k.endswith('.pkl'):
            coords = []
            acc = []
            print(cat(scandir, k))
            try:
                m = torch.load(cat(scandir, k))
                accli = []
                accsave = cat(scandir, k + '.acc')
                for mi in m:
                    coords.append(mi.cpu().numpy())

                if os.path.exists(accsave):
                    with open(accsave, 'rb') as f:
                        accli = pickle.load(f)
                else:
                    # li = []

                    for ind, mi in enumerate(m):
                        # wra.set_coords_no_grad(mi)
                    # net.load_state_dict(m['state_dict'])
                        # li.append(list(eval_coord(wra, mi, trainloader, testloader)))
                        wra.set_coords_no_grad(mi)
                        q1.put((ind, copy.deepcopy(wra.model.model.model.state_dict())))
                        # li.append(mp.Process(target=evaluation.epoch_dispatcher, args=
                        #     (wra.model.model.model, q1, ind)
                        # ))
                        # li[-1].start()
                    accli = [None for _ in m]
                    # for x, proc in enumerate(li):
                    for _ in m:
                        # proc: mp.Process
                        # proc.join()
                        x, *res = q2.get()
                        accli[x] = res
                    # li.clear()

                    with open(accsave, 'wb') as f:
                        pickle.dump(accli, f)
                acc.extend(accli)
                pin(f"{cat(scandir, k)}: {accli}")

            except:
                traceback.print_exc()

            coords = np.array(coords)
            # with open('vgg9_33G.pkl', 'wb') as f:
            #     pickle.dump((coords, acc), f)
            # print('loading...')
            # with open('vgg9_33G.pkl', 'rb') as f:
            #     coords, acc = pickle.load(f)
            # print('loaded')

            for i in coords:
                i[:] -= coords[-1]
            print('PCA...')

            if mode != 1:
                pca = PCA(n_components=mode)
                pca.fit(coords)
                ax = pca.components_[:mode]
            else:
                dire = coords[-1] - coords[0]
                dire = dire / np.sqrt(np.sum(dire**2))
                ax = [dire]

            acc = [x[3] for x in acc]

            pos = []

            for p, i in enumerate(coords):
                if mode == 1:
                    pos.append([i.dot(x) for x in ax] + [acc[p]])
                else:
                    pos.append([i.dot(x) for x in ax])
            # with open('direct3.pkl', 'wb') as f:
            #     pickle.dump(ax, f)
            # with open('projected.pkl', 'wb') as f:
                # pickle.dump((pos, acc), f)
            # with open('projected.pkl', 'rb') as f:
                # pos, acc = pickle.load(f)

            print('plotting...')
            print(pos, acc)
            plot_curve(pos, acc, k, savePath=f"figures_autopath/{mode}d", mode=mode)
    for _ in consumers:
        q1.put(None)
    for x in consumers:
        x.join()

    
    

    # coords = torch.tensor(coords)
    # torch.save(coords, 'vgg9_all11.pkl')