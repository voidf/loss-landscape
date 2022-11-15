from functools import reduce
import itertools
import json
import operator
import json5
import numpy as np
import torch
import sys
import argparse

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from typing import Union, Dict

import yaml

from cifar10.model_loader import *
import dataloader
import evaluation
from torchsummary import summary
import datetime
from loguru import logger
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, NLLLoss, Parameter
from torch import nn
from math import sqrt
from typing import Iterable, Dict, Union

from torch.optim.lr_scheduler import StepLR


try:
    from tqdm import tqdm as pbar
except ModuleNotFoundError:
    class pbar:
        def __init__(self, iterable=None, desc=None, total=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            yield from self.iterable

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update(self, N=None):
            pass

def lerp(A, B, t): # 0 -> A, 1 -> B
    return A + (B - A) * t

def param_init(mod: Module):
    if isinstance(mod, nn.Linear):
        n = mod.in_features
        mod.weight.data.normal_(0, 1. / sqrt(n))
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        n = mod.in_channels
        for k in mod.kernel_size:
            n *= k
        mod.weight.data.normal_(0, 1. / sqrt(n))
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)):
        n = mod.in_channels
        for k in mod.kernel_size:
            n *= k
        mod.weight.data.normal_(0, 1. / sqrt(n))
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        mod.reset_parameters()
    elif hasattr(mod, "initialise_randomly"):
        mod.initialise_randomly()
    elif hasattr(mod, "init_params"):
        mod.init_params()
    elif len(mod._parameters) == 0:
        # Module has no parameters on its own
        pass
    else:
        print("Don't know how to initialise %s" % mod.__class__.__name__)



class CompareModel(Module):
    ERROR = "error"
    ERROR_5 = "error_5"
    LOSS = "loss"

    def __init__(self, model: Module, loss: Module):
        super().__init__()
        self.model = model # 这个包的才是网络模型
        self.loss = loss

    def forward(self, data, target, **kwargs):
        soft_pred = self.model(data, **kwargs)
        return self.loss(soft_pred, target)

    def analyse(self, data, target):
        # Compute some statistics over the given batch
        soft_pred = self.model(data)
        hard_pred = soft_pred.data.sort(1, True)[1]

        hard_pred_correct = hard_pred[:].eq(target.data.view(-1, 1)).cumsum(1)
        error = 1 - hard_pred_correct[:, 0].float().mean().item()
        loss = self.loss(soft_pred, target).item()
        return {
            CompareModel.ERROR: error,
            CompareModel.LOSS: loss,
        }

def ensure_data_loader(data_provider: Union[Dataset, DataLoader], batch_size, **kwargs) -> DataLoader:
    if isinstance(data_provider, Dataset):
        return DataLoader(data_provider, batch_size=batch_size, **kwargs)
    # data_provider.batch_size = batch_size
    return data_provider

import threading, queue
class DataModel(Module): # 用来喂数据的包装Model的类
    def __init__(self, model: CompareModel, datasets: Dict[str, Union[Dataset, DataLoader]], bs=128):
        super().__init__()

        self.model = model

        self.datasets = datasets
        self.dataset_loaders = {}
        self.dataset_iters = {}
        self.mq = queue.Queue()
        self.lp: Union[threading.Thread, None] = None
        self.batch_size = bs

    def iterloader(self, loader):
        print(f'reading async from {loader}')
        try:
            self.mq.put(iter(loader))
        except Exception as e:
            print(e)

    def forward(self, dataset="train", **kwargs):
        # Retrieve batch
        while True:
            # Make sure that there is a non-empty iterator
            if dataset not in self.dataset_iters:
                if dataset not in self.dataset_loaders:
                    if dataset == 'train':
                        self.dataset_loaders[dataset] = ensure_data_loader(self.datasets[dataset], batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)
                loader = self.dataset_loaders[dataset]
                # self.dataset_iters[dataset] = iter(loader) # 高开销


                if self.lp is None:
                    self.dataset_iters[dataset] = iter(loader) # 高开销
                else:
                    self.lp.join()
                    try:
                        self.dataset_iters[dataset] = self.mq.get_nowait()
                    except queue.Empty:
                        self.lp = None
                        continue
                    
                self.lp = threading.Thread(name='AsyncIterLoader', target=self.iterloader, args=(loader,))
                self.lp.start()

                
            iterator = self.dataset_iters[dataset]

            try:
                batch = next(iterator)
                break
            except StopIteration:
                del self.dataset_iters[dataset]

        # Apply model on batch and use returned loss
        device_batch = self.batch_to_device(batch)
        return self.model(*device_batch, **kwargs)

    def batch_to_device(self, batch):
        device = list(self.model.parameters())[0].device
        return [item.to(device) for item in batch]


class ModelWrapper():
    """
    Wrapper around model. Inner model should handle data loading and return a value to be minimized.
    """

    def __init__(self, model: DataModel, parameters: Iterable[Parameter] = None, buffers: Iterable[Tensor] = None):
        super().__init__()
        self.model = model

        if parameters is None:
            parameters = model.parameters()
        self.stored_parameters = list(parameters)
        if buffers is None:
            buffers = model.buffers()
        self.stored_buffers = list(buffers)

        self.number_of_dimensions = sum(size for _, _, size, _ in self.iterate_params_buffers())
        device = self.stored_parameters[0].device
        self.coords = torch.empty(self.number_of_dimensions, dtype=torch.float32).to(device).zero_()
        self.coords.grad = self.coords.clone().zero_()

    def get_device(self):
        return self.coords.device

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        # noinspection PyProtectedMember
        self.stored_buffers = list(self.model.buffers())
        self._check_device()

    def parameters(self):
        return self.stored_parameters

    def initialise_randomly(self):
        self.model.apply(param_init)

    def iterate_params_buffers(self, gradient=False):
        offset = 0
        for param in self.stored_parameters:
            size = reduce(operator.mul, param.data.shape)
            data = param
            yield offset, data.data if not gradient else data.grad.data, size, False
            offset += size
        for buffer in self.stored_buffers:
            size = reduce(operator.mul, buffer.shape, 1)
            yield offset, buffer if not gradient else None, size, True
            offset += size

    def _check_device(self):
        new_device = self.stored_parameters[0].device
        if self.coords.device != self.coords.device:
            self.coords.to(new_device)

    def _coords_to_model(self):
        self._check_device()

        final = 0
        for offset, data, size, is_buffer in self.iterate_params_buffers():
            # Copy coordinates
            if len(data.shape) == 0:
                data.fill_(self.coords[offset:offset + size][0])
            else:
                data[:] = self.coords[offset:offset + size].reshape(data.shape)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def _coords_to_cache(self):
        self._check_device()

        final = 0
        for offset, tensor, size, is_buffer in self.iterate_params_buffers():
            # Copy coordinates
            self.coords[offset:offset + size] = tensor.data.view(-1)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def _grad_to_cache(self):
        self._check_device()

        final = 0
        for offset, tensor, size, is_buffer in self.iterate_params_buffers(True):
            # Copy gradient
            if tensor is None:
                self.coords.grad[offset:offset + size] = 0
            else:
                self.coords.grad[offset:offset + size] = tensor.data.view(-1)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def get_coords(self, target: Tensor = None, copy: bool = True, update_cache: bool = True) -> Tensor:
        """
        Retrieve the coordinates of the current model.

        :param target: If given, copy the data to this destination.
        :param copy: Copy the data before returning it.
        :param update_cache: Before copying, retrieve the current coordinates from the model. Set `False` only if you are sure that they have been retrieved before.
        :return: A tensor holding the coordinates.
        """
        assert target is None or copy, "Must copy if target is specified"

        if update_cache:
            self._coords_to_cache()

        if target is None:
            if copy:
                return self.coords.clone()
            else:
                return self.coords.detach()
        else:
            target[:] = self.coords.to(target.device)
            return target

    def get_grad(self, target: Tensor = None, copy: bool = True, update_cache: bool = True):
        """
        Retrieve the gradient of the current model.

        :param target:
        :param copy:
        :param update_cache: Before copying, retrieve the current gradient from the model. Set `False` only if you are sure that it has been retrieved before.
        :return:
        """
        assert target is None or copy, "Must copy if target is specified"

        if update_cache:
            self._grad_to_cache()

        if target is None:
            if copy:
                return self.coords.grad.clone()
            else:
                return self.coords.grad.detach()
        else:
            target[:] = self.coords.grad.to(target.device)
            return target

    def set_coords_no_grad(self, coords, update_model=True):
        self._check_device()
        self.coords[:] = coords.to(self.coords.device)

        if update_model:
            self._coords_to_model()

    def apply(self, gradient=False, **kwargs):
        # Forward data -> loss
        if gradient:
            self.model.zero_grad()
            self.model.train()
        else:
            self.model.eval()
        with torch.set_grad_enabled(gradient):
            loss = self.model(**kwargs)

        # Backpropation
        if gradient:
            loss.backward()
        return loss.item()

    def analyse(self):
        self.model.eval()
        with torch.set_grad_enabled(False):
            return self.model.analyse()

from fill import equal

class NEB():
    def __init__(self, model: ModelWrapper, start: Tensor, end, equal_piece_cnt=5, spring_constant=0.08, wd=0):
        """
        Creates a NEB instance that is prepared for evaluating the band.

        For computing gradients, adapt_to_config has to be called with valid values.
        """
        self.model = model
        
        # t = torch.cat((start, end), 0)
        nt = start.new(2, *start.shape)
        nt[0][:] = start
        nt[1][:] = end
        a, b = equal({
            'path_coords': nt,
            'target_distances': torch.ones(1)
        }, equal_piece_cnt)

        # t = [lerp(start, end, x) for x in torch.linspace(0, 1, equal_piece_cnt)]
        self.path_coords = a.cuda()
        self.target_distances = b.cuda()
        # self.target_distances = torch.zeros(self.path_coords.shape[0] - 1)
        self._check_device()

        self.spring_constant = spring_constant
        self.weight_decay = wd

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self._check_device()

    def _check_device(self):
        new_device = self.model.stored_parameters[0].device
        if new_device != self.path_coords.device:
            previous = self.path_coords
            self.path_coords = self.path_coords.to(new_device)
            if self.path_coords.grad is not None:
                self.path_coords.grad = previous.grad.to(new_device)
            if self.target_distances is not None:
                self.target_distances = self.target_distances.to(new_device)

    def _assert_grad(self):
        if self.path_coords.grad is None:
            self.path_coords.grad = self.path_coords.new(self.path_coords.shape).zero_()

    def parameters(self):
        return [self.path_coords]

    def apply(self, gradient=False, **kwargs):
        npivots = self.path_coords.shape[0]
        losses = self.path_coords.new(npivots)

        # Redistribute if spring_constant == inf
        assert self.target_distances is not None or not gradient, "Cannot compute gradient if target distances are unavailable"

        # Assert gradient storage is available
        if gradient:
            self._assert_grad()

        # Compute losses (and gradients)
        for i in range(npivots):
            self.model.set_coords_no_grad(self.path_coords[i])
            losses[i] = self.model.apply(gradient and (0 < i < npivots - 1))
            if gradient and (0 < i < npivots - 1):
                # If the coordinates were modified, move them back to the cache
                self.model.get_coords(update_cache=True, target=self.path_coords[i])
                self.model.get_grad(update_cache=True, target=self.path_coords.grad[i])

                assert self.weight_decay >= 0
                if self.weight_decay > 0:
                    self.path_coords.grad[i] += self.weight_decay * self.path_coords[i]
            elif gradient:
                # Make sure no gradient is there on the endpoints
                self.path_coords.grad[i].zero_()
        print('losses max:', losses.max())
        # Compute NEB gradients as in (Henkelmann & Jonsson, 2000)
        if gradient:
            distances = (self.path_coords[:-1] - self.path_coords[1:]).norm(2, 1)
            for i in range(1, npivots - 1):
                d_prev, d_next = distances[i - 1].item(), distances[i].item()
                td_prev, td_next = self.target_distances[i - 1].item(), self.target_distances[i].item()
                l_prev, loss, l_next = losses[i - 1].item(), losses[i].item(), losses[i + 1].item()

                # Compute tangent
                tangent = self.compute_tangent(d_next, d_prev, i, l_next, l_prev, loss)

                # Project gradients perpendicular to tangent
                self.path_coords.grad[i] -= self.path_coords.grad[i].dot(tangent) * tangent
                print('grad norm:', self.path_coords.grad[i].norm())

                assert self.spring_constant > 0
                if self.spring_constant < float("inf"):
                    # Spring force parallel to tangent
                    t: Tensor = ((d_prev - td_prev) - (d_next - td_next)) * self.spring_constant * tangent
                    self.path_coords.grad[i] += t
                    print('t:', t.norm())
            print('distance max:', distances.max())

        return losses.max().item()

    def compute_tangent(self, d_next, d_prev, i, l_next, l_prev, loss):
        if l_prev < loss > l_next or l_prev > loss < l_next:
            # Interpolate tangent at maxima/minima to make convergence smooth
            t_prev = (self.path_coords[i] - self.path_coords[i - 1]) / d_prev
            t_next = (self.path_coords[i + 1] - self.path_coords[i]) / d_next
            l_max = max(abs(loss - l_prev), abs(loss - l_next))
            l_min = min(abs(loss - l_prev), abs(loss - l_next))
            if l_prev > l_next:
                tangent = l_min * t_prev + l_max * t_next
            else:
                tangent = l_max * t_prev + l_min * t_next
            return tangent / (tangent.norm() + 1e-30)
        elif l_prev > l_next:
            # Tangent to the previous
            return (self.path_coords[i] - self.path_coords[i - 1]) / d_prev
        else:
            # Tangent to the next
            return (self.path_coords[i + 1] - self.path_coords[i]) / d_next

def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--file', '-f', default='Coo.json', help='result file name')
    # parser.add_argument('--begin', '-b', default='0:0:0', help='begin args')
    # parser.add_argument('--end', '-e', default='7:0:0', help='end args')
    parser.add_argument('-k', type=int, default=5, help='elastic constant')
    parser.add_argument('--npivots', '-n', type=int, default=5, help='number of pivots')

    parser.add_argument('--model_folder', '-mf', default='', help='model folder')
    parser.add_argument('--model', '-m', default='vgg9', help='model type')
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

    net = load(args.model)
    # ne2 = load(args.model)
    # ne2.load_state_dict(torch.load(args.model2)['state_dict'])

    trainloader, testloader = dataloader.load_dataset()

    net_container = ModelWrapper(DataModel(
        CompareModel(net, CrossEntropyLoss()),
        {
            'train': trainloader,
            'test': testloader
        }
    ))

    # for _ in pbar(range(390 * 70), "NEB"):
    #     neb.apply(True)
    #     op.step()

    coords: Tensor = torch.load('neb_path.pkl')
    # coords_diff = coords.clone().fill_(coords[0])
    for x in range(coords.shape[0]-1, -1, -1):
        net_container.set_coords_no_grad(coords[x])
        net_container.model.eval()

        loss, acc = evaluation.eval_loss(net_container.model.model.model, CrossEntropyLoss(), trainloader)
        pin(f"train:[{loss},{acc}]")
        loss, acc = evaluation.eval_loss(net_container.model.model.model, CrossEntropyLoss(), testloader)
        pin(f"test:[{loss},{acc}]")

        coords[x][:] -= coords[0]
        # print(coords[x].norm())
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(coords.cpu().numpy())
    x, y, z = pca.components_[:3]
    ld = []
    mx = 0.0
    for p, i in enumerate(coords):
        ci = i.cpu().numpy()
        ld.append([ci.dot(x),ci.dot(y),ci.dot(z),])
        mx = max([mx] + ld[-1])
        # pin(str({
        #     'x': ld[-1][0],
        #     'y': ld[-1][1],
        #     'z': ld[-1][2],
        # }))
        pin(str(ld[-1]))
    
    

    # torch.save(neb.path_coords, 'neb_path2.pkl')

    pin(json.dumps({
        'model1': args.model1,
        'model2': args.model2,
        'method': 'neb',
    }))

    f.close()



    # print('time used:', (t2-t1).total_seconds())
    # print(a,b)
