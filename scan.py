import traceback
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

from cifar10.model_loader import *
from wrappers import *



def plot_curve(points, acc, savePath=''):
    points = np.array(points)
    print(points.shape)

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
    # net = load('vgg9')

    # wra = construct_wrapper(net)
    # coords = []
    # acc = []

    # import os
    # for i in os.listdir('.'):
    #     if i.startswith('tn'):
    #         for j in os.listdir(i):
    #             for k in os.listdir(cat(i,j)):
    #                 if k.startswith('model_'):
    #                     print(k)
    #                     try:
    #                         m = torch.load(cat(i, j, k))
    #                         net.load_state_dict(m['state_dict'])
    #                         coords.append(wra.get_coords().numpy())
    #                         acc.append(m['acc'])
    #                     except:
    #                         traceback.print_exc()
    #         print(i)
    # coords = np.array(coords)
    # with open('vgg9_33G.pkl', 'wb') as f:
    #     pickle.dump((coords, acc), f)
    # print('loading...')
    # with open('vgg9_33G.pkl', 'rb') as f:
    #     coords, acc = pickle.load(f)
    # print('loaded')

    # for i in coords[::-1]:
    #     i[:] -= coords[0]
    # print('PCA...')

    # pca = PCA(n_components=3)
    # pca.fit(coords)
    # ax = pca.components_[:3]
    # pos = []

    # for i in coords:
    #     pos.append([i.dot(x) for x in ax])
    # with open('direct3.pkl', 'wb') as f:
    #     pickle.dump(ax, f)
    # with open('projected.pkl', 'wb') as f:
        # pickle.dump((pos, acc), f)
    with open('projected.pkl', 'rb') as f:
        pos, acc = pickle.load(f)

    print('plotting...')
    print(pos, acc)
    plot_curve(pos, acc)

    

    # coords = torch.tensor(coords)
    # torch.save(coords, 'vgg9_all11.pkl')