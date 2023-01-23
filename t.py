import os
from wrappers import cat
d = {}


for a, b, c in os.walk(cat('trained', 'R56_09', 'resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1')):
    # print(a,b,c)
    sp = os.path.split(a)
    t7s = [i for i in c if i.endswith('.t7')]
    if t7s:
        d[cat(*sp[1:])] = t7s

print(d)