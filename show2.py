import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
import os
from loguru import logger

config = None
# fd = r'tn09/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/amount=0.4'
# fd = r'R56N_DL/amount=0.4'
fd = r'R56N_05/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/model_10B5/amount=0.4'
# fd = r'R56_01/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/amount=0.4'


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f')
    parser.add_argument('--all', '-a', action='store_true')
    return parser.parse_args()

def _3D_plot(points):
    x, y, z = points.T
    color = np.linspace(0, 5, z.shape[0])
    Z = np.array([z, color])
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #surf = ax.plot_surface(x, y, Z,)
    ax.scatter(x, y, z, s=5)
    title = '-'.join(config.file.split('/'))
    ax.set_title(f'{title}')
    fig.savefig(f'./img/{title}.png')

def surface_3d(points, fn):
    x, y, z = points.T
    sq = int(len(x) ** 0.5)
    X = x.reshape(sq, sq)
    Y = y.reshape(sq, sq)
    Z = z.reshape(sq, sq)
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title = '-'.join(fn.split('/') + ['surf'])
    ax.set_title(f'{title}')
    fig.savefig(f'{fd}/{title}(heat).png')
    plt.show()


def main(fn):
    global config
    data = torch.load(fn)
    points = []
    for key, value in data.items():
        point = list(key)
        point += [value[0]]
        point = np.array(point)
        points.append(point)

    points = np.array(points)
    
    logger.info(f'points shape:{points.shape}')
    sortedPoints = points[:]
    dim = sortedPoints.shape[-1]
    for i in range(dim - 1, -1, -1):
        sortedPoints = sortedPoints[sortedPoints[:, i].argsort(kind='mergesort')]
    surface_3d(sortedPoints, fn)


if __name__ == '__main__':
    for x in os.listdir(fd):
        if x.endswith('.dict'):
            plot = '/'.join([fd, x])
            logger.info(plot)
            main(plot)