from cmath import isnan
import functools
from fastapi import HTTPException
import json
import math
import operator
import traceback
from typing import List, Optional
from cv2 import reduce
import fastapi
import os
from pydantic import BaseModel
from sklearn.decomposition import PCA
import torch
import uvicorn
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
from cifar10.model_loader import load
from net_plotter import get_weights
from wrappers import cat
from torch import Tensor
import torch.optim
from scan_traj import cat_tensor, get_states, write_buf_no_nbt, write_states, write_weights
import torch.multiprocessing as mp
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from safetensors import safe_open

model_dir = 'trained/'
worker_cnt = 5
# pool = None

from fastapi import BackgroundTasks, FastAPI, Query, Request, Response

def train_consumer(q1: mp.Queue):
    from scan_traj import scan
    while 1:
        task = q1.get()
        if task is None:
            return
        try:
            scan(*task)
        except:
            traceback.print_exc()

def t7_to_tensor(arch, fp, return_t7=False, tensor_key='state_dict', skip_num_batches_tracked=True) -> Tensor:
    net = load(arch)
    t7 = torch.load(fp)
    net.load_state_dict(t7.pop(tensor_key))
    ten = cat_tensor(get_states(net, skip_num_batches_tracked))
    if return_t7:
        return ten, t7
    return ten

# def optim_to_tensor(arch, fp, opt='sgd', optim_key='optimizer') -> Tensor:
#     net = load(arch)
#     if opt == 'sgd':
#         o = torch.optim.SGD(net.parameters(), 0.1)
#     o.load_state_dict(torch.load(fp))
#     return cat_tensor(get_states(o))


def u_resolver(src: list[str], proj: str, pf='model_', sf='.safetensors') -> list[str]:
    for j in os.listdir(proj):
        for p, i in enumerate(src):
            path = i.split('/')
            m = f'{pf}{path.pop()}{sf}'
            for idx, i in enumerate(path):
                f, b = i.split('.')
                path[idx] = f'model_{f}B{b}'
            src[p] = cat(proj, j, *path, m)
    return src

def preload() -> FastAPI:
    """多worker fork出来前先进行一些通用东西的初始化"""
    
    if not os.path.exists('log'):
        os.mkdir('log')
    app = FastAPI()
    app.add_middleware( # 允许跨域第一版
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.middleware('http') # TODO: [insecure] set to a fixed origin
    async def cors_everywhere(request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = request.headers.get('origin', '*')
        return response



    # app.include_router(v1_router)
    return app

app = preload()

@app.on_event("startup")
async def _():
    FastAPICache.init(InMemoryBackend())

@app.get('/models')
@cache()
async def _():
    l = []
    for i in os.listdir(model_dir):
        l.append(i)
    return l

@app.get('/list')
@cache()
async def _(proj: str):
    proj = model_dir + proj
    d = {}
    root = True
    for i in os.listdir(proj): # 跳过超参记录夹
        for dp, dn, fn in os.walk(cat(proj, i)):
            sp = dp.removeprefix(cat(proj, i)).split(os.sep)
            t7s = [i for i in fn if i.endswith('.safetensors') and i.startswith('model_')]
            key = cat(*sp[1:])
            if root:
                root = False
                key = ''
            if t7s:
                d[key] = t7s
    l = []
    dd = {}
    for k, v in d.items():
        pk = k.split('/')
        pk3 = []
        kp = []
        if k:
            for p, i in enumerate(pk):
                # logger.debug(f"i:{i} pk:{pk} k:{k}")
                father, no = i.removeprefix('model_').split('B')
                # pk[p] = int(father)
                # pk2.append(int(no))
                pk3.append(f'{father}.{no}')
                kp.append((int(father), int(no)))
        else:
            pk = []
        # p3s = cat(*pk3)
        for w in v:
            l.append({
                'kp': kp,
                'p3': pk3,
                'x': int(w.removeprefix('model_').removesuffix('.safetensors'))
            })
    import functools
    def lcmp(x, y):
        if len(x['kp']) < len(y['kp']):
            return -1
        elif len(x['kp']) > len(y['kp']):
            return 1
        elif x['kp'] < y['kp']:
            return 1
        elif x['kp'] > y['kp']:
            return -1
        elif x['x'] < y['x']:
            return -1
        elif x['x'] > y['x']:
            return 1
        else:
            return 0
    l.sort(key=functools.cmp_to_key(lcmp)
    # lambda it: it['p'] + it['p2'] + [it['x']]
    )
    for it in l:
        it['l'] = p3s = cat(*it.pop('p3'))
        it.pop('kp')
        it['y'] = dd.setdefault(p3s, len(dd))
        it['u'] = cat(p3s, str(it['x'])).removeprefix('/')


    # logger.debug(l)
    ddl = [None for _ in dd]
    for k, v in dd.items(): ddl[v] = k

    return {'label': ddl, 'points': l}

class ArgsPCA(BaseModel):
    arch: str
    selection: List[str]
    proj: str
    weight: bool = True

def translate_u_path(u: str) -> List[str]:
    path = u.split('/')
    m = f'model_{path.pop()}.safetensors'
    for p, i in enumerate(path):
        f, b = i.split('.')
        path[p] = f'model_{f}B{b}'
    path.append(m)
    return path

@app.post('/pca')
@cache()
async def _(s: ArgsPCA):
    logger.info(s)
    if len(s.selection) == 0:
        raise HTTPException(400, 'empty selection')
    import numpy as np
    proj = model_dir + s.proj
    pca = PCA(2)

    nplist = []
    meta = []
    from safetensors import safe_open

    avg = None
    
        
    net = load(s.arch)
    # weight_len = sum(map(lambda x: reduce(operator.mul, x.data.shape), net.parameters()))

    for j in os.listdir(proj):
        for i in s.selection:
            with safe_open(fp := cat(proj, j, *translate_u_path(i)), framework='np') as fil:
                param: np.ndarray = fil.get_tensor('param')
                buf: np.ndarray = fil.get_tensor('buf')
                # nbt: np.ndarray = fil.get_tensor('nbt')
            cated = np.concatenate((param, buf))
            if avg is None:
                avg = cated.copy()
            else:
                avg += cated
            with open(fp.removesuffix('safetensors') + 'json') as fil:
                t7 = json.load(fil)
            # ts, t7 = t7_to_tensor(s.arch, cat(proj, j, *translate_u_path(i)), True)
            # ts = ts.numpy() # 输出： param + buf
            if s.weight:
                nplist.append(param)
            else:
                nplist.append(cated)
            # nplist.append(ts[:weight_len])
            meta.append(t7)


    # for ts in enum_path():
        # nplist.append(ts)
    # avg = nplist.mean(axis=0)
    avg /= len(nplist)
    nplist = np.array(nplist)

    # pca.fit(nplist)
    logger.debug('nplist len: {}', len(nplist))
    newlist = pca.fit_transform(nplist).tolist()
    axis: np.ndarray = pca.components_[:2]
    # for p, it in enumerate(enum_path()):
        # i = newlist[p]
    for p, i in enumerate(newlist):
        # D = (nplist[p] - avg)
        # X = D.dot(axis[0])
        # Y = D.dot(axis[1])
        # DX = i[0] - X
        # DY = i[1] - Y # 等价的，误差在e-07数量级左右
        newlist[p] = {
            'x': i[0],
            'y': i[1],
            'u': s.selection[p],
            'l': cat(*s.selection[p].split('/')[:-1]),
        }
    return {
        'axis': axis.tolist(),
        'mean': avg.tolist(),
        'coord': newlist,
        'meta': meta
    }

@app.get('/info')
@cache()
async def _(p: str, proj: str):
    path = u_resolver([p], model_dir + proj)[0]
    with open(path.removesuffix('safetensors')+'json', 'r') as f:
        return json.load(f)
    # t7 = torch.load(path)
    # t7.pop('state_dict')
    # return t7

class ArgsTrain(BaseModel):
    u: str
    lr: float
    bs: int
    mom: float
    wd: float
    seed: int
    op: str
    e: int
    arch: str
    proj: str

@app.post('/train')
async def _(a: ArgsTrain, b: BackgroundTasks, r: Response):
    path = u_resolver([a.u], model_dir + a.proj)[0]
    dire = path.rsplit('/', 1)[0]
    from_epoch = int(a.u.rsplit('/')[-1])
    t_queue.put((
        a.arch,
        from_epoch,
        a.e,
        a.lr,
        a.mom,
        a.wd,
        a.bs,
        a.op,
        a.seed,
        dire))
    r.status_code = 202
    return r
        
class ArgsHeatmap(BaseModel):
    xstep: int
    ystep: int
    xstep_rate: float
    ystep_rate: float
    arch: str
    mean: Optional[List[float]] = None
    xdir: List[float]
    ydir: List[float]
    u: Optional[str] = None
    proj: Optional[str] = None

@app.post('/heatmap')
@cache()
async def _(a: ArgsHeatmap):
    import evaluation
    import copy

    mng = mp.Manager()
    q1 = mng.Queue(maxsize=worker_cnt)
    q2 = mng.Queue()
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
            args=(a.arch, q1, q2)
        ) for _ in range(worker_cnt)
    ]
    for x in consumers: x.start()

    net = load(a.arch)
    if a.u:
        proj = model_dir + a.proj
    
        for j in os.listdir(proj):
            with safe_open(cat(proj, j, *translate_u_path(a.u)), framework="pt", device='cpu') as fil:
                # param = fil.get_tensor('param')
                # buf = fil.get_tensor('buf')
                write_weights(net, param := fil.get_tensor('param'))
                write_buf_no_nbt(net, buf := fil.get_tensor('buf'))
            
            a.mean = torch.cat((param, buf))
            # a.mean = t7_to_tensor(a.arch, cat(proj, j, *translate_u_path(a.u)))
    else:
        a.mean = torch.tensor(a.mean)
        write_states(net, a.mean)

    if len(a.mean) > len(a.xdir):
        weight_mode = True
        a.mean = a.mean[:len(a.xdir)]
    elif len(a.mean) == len(a.xdir):
        weight_mode = False
    else:
        raise HTTPException(500, f'mean shape:{len(a.mean)} != {len(a.xdir)}')
    #     # weight mode
    #     ts: Tensor = torch.zeros(len(a.mean))
    #     ts[:len(a.xdir)] = a.xdir[:]
    #     a.xdir = ts
    #     ts[:len(a.ydir)] = a.ydir[:]
    #     a.ydir = ts
    # else:
    a.xdir = torch.tensor(a.xdir)
    a.ydir = torch.tensor(a.ydir)

    
    # needle = copy.deepcopy(net)
    with torch.no_grad():
        for x in range(-a.xstep, a.xstep + 1):
            for y in range(-a.ystep, a.ystep + 1):
                cur = a.mean + x * a.xstep_rate * a.xdir + y * a.ystep_rate * a.ydir
                if weight_mode:
                    write_weights(net, cur)
                else:
                    write_states(net, cur)
                q1.put(((x, y), copy.deepcopy(net.state_dict())))
            # set_weights(needle, get_weights)


    for _ in consumers: q1.put(None)
    for x in consumers: x.join()

    ret = []
    for _ in range((2 * a.xstep + 1) * (2 * a.ystep + 1)):
        (x, y), train_loss, train_acc, test_loss, test_acc = q2.get()
        if math.isnan(train_loss) or math.isnan(test_loss):
            continue
        ret.append({
            'x': x * a.xstep_rate,
            'y': y * a.ystep_rate,
            'trl': train_loss,
            'tra': train_acc,
            'tel': test_loss,
            'tea': test_acc,
        })
    logger.debug(ret)
    return ret
    # net = load(a.arch)


if __name__ == "__main__":
    global t_manager, t_queue, t_consumers
    t_manager = mp.Manager()
    t_queue = t_manager.Queue()
    t_consumers = [
        mp.Process(target=train_consumer,
            args=(t_queue,)
        ) for _ in range(worker_cnt)
    ]
    for x in t_consumers: x.start()
    uvicorn.run(app, port=40000)
