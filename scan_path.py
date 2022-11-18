import traceback
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from loguru import logger

from cifar10.model_loader import *
from wrappers import *


def plot_curve(points, acc, savePath=''):
    points = np.array(points)
    print(points.shape)
    print('acc len:', len(acc))
    print(acc[:100])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    color = np.linspace(0, points.shape[0] - 1, points.shape[0], dtype=float)
    cblue = np.array([.0, .0, 255.0])
    cred = np.array([255.0, 0.0, 0.0])

    def make_color(x):
        return '#%02x%02x%02x' % tuple(int(i) for i in x)

    for p, i in enumerate(acc):
        acc[p] = make_color(lerp(cred, cblue, i / 100.0))

    x, y, z = points.T

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
    ax.scatter(
        x, y, z,
        c=acc,
        s=10,
    )

    # for i in range(points.shape[0]):
    #     ax.text(*points[i],s=str(i))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    #fig.savefig('./3d.png')
    fig.savefig('/'.join(['.', savePath, '3D_curve.png']))

if __name__ == '__main__':
    net = load('vgg9')

    wra = construct_wrapper(net)
    wra.to('cuda') # 关键

    # ms = torch.load(cat('paths08_07_k=0A', 'neb_path3.pkl'))
    # for mi in ms:
    #     eval_coord(wra, mi)

    coords = []
    acc = []

    scandir = 'paths08_07_k=0A'

    fi = open('logs.txt', mode='a')
    def pin(x: str):
        logger.info(x)
        fi.write(x + '\n')
        fi.flush()

    import os
    for k in os.listdir(scandir):
        if k.endswith('.pkl'):
            print(cat(scandir, k))
            try:
                m = torch.load(cat(scandir, k))
                li = []
                accsave = cat(scandir, k + '.acc')
                for mi in m:
                    coords.append(mi.cpu().numpy())

                if os.path.exists(accsave):
                    with open(accsave, 'rb') as f:
                        li = pickle.load(f)
                else:
                    for mi in m:
                        # wra.set_coords_no_grad(mi)
                    # net.load_state_dict(m['state_dict'])
                        li.append(list(eval_coord(wra, mi)))
                    with open(accsave, 'wb') as f:
                        pickle.dump(li, f)
                acc.extend(li)
                pin(f"{cat(scandir, k)}: {li}")

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

    pca = PCA(n_components=3)
    pca.fit(coords)
    ax = pca.components_[:3]

    acc = [x[3] for x in acc]

    pos = []

    for i in coords:
        pos.append([i.dot(x) for x in ax])
    # with open('direct3.pkl', 'wb') as f:
    #     pickle.dump(ax, f)
    # with open('projected.pkl', 'wb') as f:
        # pickle.dump((pos, acc), f)
    # with open('projected.pkl', 'rb') as f:
        # pos, acc = pickle.load(f)

    print('plotting...')
    print(pos, acc)
    plot_curve(pos, acc)

    

    # coords = torch.tensor(coords)
    # torch.save(coords, 'vgg9_all11.pkl')