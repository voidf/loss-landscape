from cmath import isnan
from http.client import HTTPException
import math
from typing import List
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
from scan_traj import cat_tensor, get_states, write_states

model_dir = 'trained/'
# pool = None

from fastapi import FastAPI, Query, Request

def t7_to_tensor(arch, fp) -> Tensor:
    net = load(arch)
    t7 = torch.load(fp)
    net.load_state_dict(t7['state_dict'])
    return cat_tensor(get_states(net))

def u_resolver(src: list[str], proj: str, pf='model_', sf='.t7') -> list[str]:
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

@app.get('/models')
async def _():
    l = []
    for i in os.listdir(model_dir):
        l.append(i)
    return l

@app.get('/list')
async def _(proj: str):
    proj = model_dir + proj
    d = {}
    root = True
    for i in os.listdir(proj): # 跳过超参记录夹
        for dp, dn, fn in os.walk(cat(proj, i)):
            sp = dp.removeprefix(cat(proj, i)).split(os.sep)
            t7s = [i for i in fn if i.endswith('.t7') and i.startswith('model_')]
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
                'x': int(w.removeprefix('model_').removesuffix('.t7'))
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

@app.post('/pca')
async def _(s: ArgsPCA):
    logger.info(s)
    if len(s.selection) == 0:
        raise HTTPException(400, 'empty selection')
    import numpy as np
    proj = model_dir + s.proj
    pca = PCA(2)

    nplist = []
    # def enum_path():
    for j in os.listdir(proj):
        for i in s.selection:
            path = i.split('/')
            m = f'model_{path.pop()}.t7'
            for p, i in enumerate(path):
                f, b = i.split('.')
                path[p] = f'model_{f}B{b}'
            ts = t7_to_tensor(s.arch, cat(proj, j, *path, m)).numpy() # 输出： param + buf
            nplist.append(ts)

    # for ts in enum_path():
        # nplist.append(ts)
    nplist = np.array(nplist)
    avg = nplist.mean(axis=0)

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
        'coord': newlist
    }

@app.get('/info')
async def _(p: str, proj: str):
    path = u_resolver([p], model_dir + proj)[0]
    t7 = torch.load(path)
    t7.pop('state_dict')
    return t7

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
async def _(a: ArgsTrain):
    path = u_resolver([a.u], model_dir + a.proj)[0]
    dire = path.rsplit('/', 1)[0]
    from_epoch = int(a.u.rsplit('/')[-1])
    from scan_traj import scan
    scan(
        a.arch,
        from_epoch,
        a.e,
        a.lr,
        a.mom,
        a.wd,
        a.bs,
        a.op,
        a.seed,
        dire
    )
        
class ArgsHeatmap(BaseModel):
    xstep: int
    ystep: int
    xstep_rate: float
    ystep_rate: float
    arch: str
    mean: List[float]
    xdir: List[float]
    ydir: List[float]

@app.post('/heatmap')
async def _(a: ArgsHeatmap):
    from net_plotter import set_weights
    import torch.multiprocessing as mp
    import evaluation
    import copy

    mng = mp.Manager()
    wk = 4
    q1 = mng.Queue(maxsize=wk+1)
    q2 = mng.Queue()
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
            args=(a.arch, q1, q2)
        ) for _ in range(wk)
    ]
    for x in consumers: x.start()

    net = load(a.arch)
    a.mean = torch.tensor(a.mean)
    a.xdir = torch.tensor(a.xdir)
    a.ydir = torch.tensor(a.ydir)

    write_states(net, a.mean)
    
    # needle = copy.deepcopy(net)
    with torch.no_grad():
        for x in range(-a.xstep, a.xstep + 1):
            for y in range(-a.ystep, a.ystep + 1):
                cur = a.mean + x * a.xstep_rate * a.xdir + y * a.ystep_rate * a.ydir
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
    # pool = ProcessPoolExecutor(4)
    uvicorn.run(app, port=40000)
