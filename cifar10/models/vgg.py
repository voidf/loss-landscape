import torch
import torch.nn as nn
import os


cfg = {
    'VGG9A':  [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

mem_cache = {}

def key(b, ckp, pf): return f'plot_cache/{pf}-{ckp}-{b}.ckp'

def chk_ckp(b, ckp, pf): 
    return key(b, ckp, pf) in mem_cache
    # return os.path.exists(f'plot_cache/{pf}-{ckp}-{b}.ckp')

def save_ckp(x, b, ckp, pf): 
    mem_cache[key(b,ckp,pf)] = x
    # torch.save(x, f'plot_cache/{pf}-{ckp}-{b}.ckp')
def load_ckp(b, ckp, pf): 
    return mem_cache[key(b,ckp,pf)]
    # print('use cache:', f'plot_cache/{pf}-{ckp}-{b}.ckp')
    # return torch.load(f'plot_cache/{pf}-{ckp}-{b}.ckp')

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.input_size = 32
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps, 10)

    def clear_ckp(self, ckp, pf):
        for i in list(mem_cache.keys()):
        # for i in os.listdir('plot_cache'):
            if i.startswith(f'plot_cache/{pf}-{ckp}-') and i.endswith('.ckp'):
                mem_cache.pop(i)
                # os.remove('plot_cache' + i)

    def forward(self, x, batch_idx=None, ckp=2, ckp_prefix='train'):
        if batch_idx is None:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out = self.classifier(out)
            return out

        out = None
        if ckp <= 0 or not chk_ckp(batch_idx, 0, ckp_prefix):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            save_ckp(out, batch_idx, 0, ckp_prefix)
        if ckp <= 1 or not chk_ckp(batch_idx, 1, ckp_prefix):
            if out is None: out = load_ckp(batch_idx, 0, ckp_prefix)
            out = self.fc(out)
            save_ckp(out, batch_idx, 1, ckp_prefix)
        if out is None: out = load_ckp(batch_idx, 1, ckp_prefix)
        out = self.classifier(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG9():
    return VGG('VGG9')

def VGG9A():
    return VGG('VGG9A')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')
