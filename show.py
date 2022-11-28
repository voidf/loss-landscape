import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import numpy as np
import os

from loguru import logger

config = None
# fd = r'R56_01/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/amount=0.4'
# fd = r'R56N_01/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1/amount=0.4'
fd = r'R56N_DL/amount=0.4'


def _3D_plot(points, fn):
    x, y, z = points.T
    color = np.linspace(0, 5, z.shape[0])
    Z = np.array([z, color])
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #surf = ax.plot_surface(x, y, Z,)
    ax.scatter(x, y, z, s=5)
    title = '-'.join(fn.split('/'))
    ax.set_title(f'{title}')
    plt.show()
    fig.savefig(f'{fd}/{title}.png')

def main(fn):
    data = torch.load(fn)
    points = []
    for key, value in data.items():
        point = list(key)
        point += [value[0]]
        point = np.array(point)
        points.append(point)

    points = np.array(points)
    logger.info(f'points shape:{points.shape}')
    _3D_plot(points, fn)


if __name__ == '__main__':
    for x in os.listdir(fd):
        if x.endswith('.dict'):
            plot = '/'.join([fd, x])
            logger.info(plot)
            main(plot)