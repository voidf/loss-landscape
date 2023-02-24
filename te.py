# import api
# x = api.t7_to_tensor('resnet56_noshort', r'trained\R56N_04\resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1\model_90B1\model_132.t7')
# print(x)

from torchsummary import summary

from cifar10.model_loader import load


summary(load('vgg9'), (3, 32, 32))
summary(load('resnet56'), (3, 32, 32))
summary(load('resnet56_noshort'), (3, 32, 32))
