from cmath import isnan
import datetime
import functools
import random
import shutil
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
from dataloader import load_dataset
from evaluation import eval_loss
from net_plotter import get_weights
from pref import find_arch
from wrappers import cat
from torch import Tensor
import torch.optim
from scan_traj import cat_tensor, get_buf_no_nbt, get_nbt, get_states, write_buf_no_nbt, write_nbt, write_states, write_weights
import torch.multiprocessing as mp
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from safetensors import safe_open
from main import init_params

import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True

model_dir = 'trained/'
worker_cnt = 5
# pool = None

from fastapi import BackgroundTasks, FastAPI, Query, Request, Response

history_dir = 'history'
def ensure_his_dir(subdir: str) -> str:
    if not os.path.exists(hd := cat(history_dir, subdir)):
        os.mkdir(hd)
    return hd

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
    @app.middleware('http') # TODO: [insecure] set to argu fixed origin
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
    selection: List[str]
    proj: str
    weight: bool = True

def translate_u_path(u: str, dir_only=False, suffix='safetensors') -> List[str]:
    path = u.split('/')
    m = f'model_{path.pop()}.{suffix}'
    for p, i in enumerate(path):
        f, b = i.split('.')
        path[p] = f'model_{f}B{b}'
    if not dir_only:
        path.append(m)
    return path

@app.post('/pca')
# @cache(expire=60)
async def _(argu: ArgsPCA):
    logger.info(argu)
    if len(argu.selection) == 0:
        raise HTTPException(400, 'empty selection')
    proj = model_dir + argu.proj
    pca = PCA(2)

    nplist = []
    avg = None
        
    net = load(find_arch(argu.proj))
    # weight_len = sum(map(lambda x: reduce(operator.mul, x.data.shape), net.parameters()))

    for j in os.listdir(proj):
        for i in argu.selection:
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
            if argu.weight:
                nplist.append(param)
            else:
                nplist.append(cated)
            # nplist.append(ts[:weight_len])


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
            'u': argu.selection[p],
            'l': cat(*argu.selection[p].split('/')[:-1]),
        }
    return {
        'axis': axis.tolist(),
        'mean': avg.tolist(),
        'coord': newlist,
    }

class ArgsMeta(BaseModel):
    selection: List[str]
    proj: str

@app.post('/meta')
# @cache()
async def _(argu: ArgsMeta):
    logger.info(argu)
    if len(argu.selection) == 0:
        raise HTTPException(400, 'empty selection')
    proj = model_dir + argu.proj
    meta = []
    for j in os.listdir(proj):
        for i in argu.selection:
            with open(cat(proj, j, *translate_u_path(i)).removesuffix('safetensors') + 'json') as fil:
                t7 = json.load(fil)
            meta.append(t7)
    return meta


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
    proj: str

@app.post('/train')
async def _(argu: ArgsTrain, r: Response):
    path = u_resolver([argu.u], model_dir + argu.proj)[0]
    dire = path.rsplit('/', 1)[0]
    from_epoch = int(argu.u.rsplit('/')[-1])
    t_queue.put((
        find_arch(argu.proj),
        from_epoch,
        argu.e,
        argu.lr,
        argu.mom,
        argu.wd,
        argu.bs,
        argu.op,
        argu.seed,
        dire))
    r.status_code = 202
    return r

class ArgsNewproj(BaseModel):
    lr: float
    bs: int
    mom: float
    wd: float
    seed: int
    op: str
    e: int
    proj: str
    arch: str

def name_save_folder(args):
    save_folder = args.arch + '_' + str(args.op) + '_lr=' + str(args.lr)
    save_folder += '_bs=' + str(args.bs)
    save_folder += '_wd=' + str(args.wd)
    save_folder += '_mom=' + str(args.mom)
    save_folder += '_seed=' + str(args.seed)
    return save_folder

@app.post('/newproj')
async def _(a: ArgsNewproj, r: Response):
    """{
  "lr": 0.1,
  "bs": 128,
  "mom": 0.9,
  "wd": 0.0005,
  "seed": 30,
  "op": "sgd",
  "e": 150,
  "proj": "R56_10",
  "arch": "resnet56"
}"""
    net = load(a.arch)
    init_params(net)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    proj = model_dir + a.proj
    if os.path.exists(proj):
        raise HTTPException(403)
    os.mkdir(proj)
    fd = cat(proj, name_save_folder(a))
    os.mkdir(fd)
    projdir = functools.partial(cat, fd)
    with open(projdir('model_0.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'batch_size': a.bs,
            'random_seed': a.seed,
            'epoch': 0,
            'optimizer': a.op,
            'lr': a.lr,
            'momentum': a.mom,
            'weight_decay': a.wd,
        }, f)
    
    from safetensors.torch import save_file
    save_file({
        'param': cat_tensor(get_weights(net)),
        'buf': cat_tensor(get_buf_no_nbt(net)),
        'nbt': cat_tensor(get_nbt(net))
    }, projdir('model_0.safetensors'))
    t_queue.put((
        a.arch,
        0,
        a.e,
        a.lr,
        a.mom,
        a.wd,
        a.bs,
        a.op,
        a.seed,
        fd))
    

    r.status_code = 202
    return r

class ArgsHeatmap(BaseModel):
    xstep: int
    ystep: int
    xstep_rate: float
    ystep_rate: float
    mean: Optional[List[float]] = None
    xdir: List[float]
    ydir: List[float]
    u: Optional[str] = None
    proj: str

@app.post('/heatmap')
async def _(argu: ArgsHeatmap):
    import evaluation
    import copy

    mng = mp.Manager()
    q1 = mng.Queue(maxsize=worker_cnt)
    q2 = mng.Queue()
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
            args=(find_arch(argu.proj), q1, q2)
        ) for _ in range(worker_cnt)
    ]
    for x in consumers: x.start()

    net = load(find_arch(argu.proj))
    if argu.u:
        proj = model_dir + argu.proj
    
        for j in os.listdir(proj):
            with safe_open(cat(proj, j, *translate_u_path(argu.u)), framework="pt", device='cpu') as fil:
                # param = fil.get_tensor('param')
                # buf = fil.get_tensor('buf')
                write_weights(net, param := fil.get_tensor('param'))
                write_buf_no_nbt(net, buf := fil.get_tensor('buf'))
            
            argu.mean = torch.cat((param, buf))
            # argu.mean = t7_to_tensor(argu.arch, cat(proj, j, *translate_u_path(argu.u)))
    else:
        argu.mean = torch.tensor(argu.mean)
        write_states(net, argu.mean)

    if len(argu.mean) > len(argu.xdir):
        weight_mode = True
        argu.mean = argu.mean[:len(argu.xdir)]
    elif len(argu.mean) == len(argu.xdir):
        weight_mode = False
    else:
        raise HTTPException(500, f'mean shape:{len(argu.mean)} != {len(argu.xdir)}')
    #     # weight mode
    #     ts: Tensor = torch.zeros(len(argu.mean))
    #     ts[:len(argu.xdir)] = argu.xdir[:]
    #     argu.xdir = ts
    #     ts[:len(argu.ydir)] = argu.ydir[:]
    #     argu.ydir = ts
    # else:
    argu.xdir = torch.tensor(argu.xdir)
    argu.ydir = torch.tensor(argu.ydir)

    
    # needle = copy.deepcopy(net)
    with torch.no_grad():
        for x in range(-argu.xstep, argu.xstep + 1):
            for y in range(-argu.ystep, argu.ystep + 1):
                cur = argu.mean + x * argu.xstep_rate * argu.xdir + y * argu.ystep_rate * argu.ydir
                if weight_mode:
                    write_weights(net, cur)
                else:
                    write_states(net, cur)
                q1.put(((x, y), copy.deepcopy(net.state_dict())))
            # set_weights(needle, get_weights)


    for _ in consumers: q1.put(None)
    for x in consumers: x.join()

    ret = []
    for _ in range((2 * argu.xstep + 1) * (2 * argu.ystep + 1)):
        (x, y), train_loss, train_acc, test_loss, test_acc = q2.get()
        if math.isnan(train_loss) or math.isnan(test_loss):
            continue
        ret.append({
            'x': x * argu.xstep_rate,
            'y': y * argu.ystep_rate,
            'trl': train_loss,
            'tra': train_acc,
            'tel': test_loss,
            'tea': test_acc,
        })
    logger.debug(ret)
    with open(cat(ensure_his_dir('heatmap'), datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + '.json'), 'w') as f:
        json.dump(ret, f)

    return ret

class ArgsDisturb(BaseModel):
    u: str
    mag: float
    proj: str

@app.post('/disturb')
async def _(a: ArgsDisturb):
    from safetensors.torch import save_file

    with torch.no_grad():
        arch = find_arch(a.proj)
        net = load(arch)
        proj = model_dir + a.proj
        from_epoch = int(a.u.rsplit('/', 1)[-1])
        fn = f'model_{from_epoch}'
        e = from_epoch + 1
        for pa in os.listdir(proj): pass
        projdir = functools.partial(cat, proj, pa, *translate_u_path(a.u, dir_only=True))


        with safe_open(projdir(fn + '.safetensors'), framework='pt', device='cuda') as fil:
            param: torch.Tensor = fil.get_tensor('param')
            buf: torch.Tensor = fil.get_tensor('buf')
            nbt: torch.Tensor = fil.get_tensor('nbt')
        
        init_params(net)
        random_param = cat_tensor(get_weights(net)) # 用kaiming_normal_

        param += a.mag * random_param # [-a.mag, a.mag] kaiming_normal_分布
        write_weights(net, param)
        write_buf_no_nbt(net, buf)
        write_nbt(net, nbt)

        if not os.path.exists(projdir(f'model_{e}.safetensors')):
            branch_dir = '.'
        else:
            idx = 1
            s = set() # MEX
            for i in os.listdir(projdir()):
                if os.path.isdir(projdir(i)) and i.startswith(fn + 'B'):
                    if len(os.listdir(projdir(i))) == 0:
                        os.rmdir(projdir(i))
                    else:
                        s.add(int(i.rsplit('B', 1)[1]))
            while idx in s: idx += 1

            branch_dir = f'{fn}B{idx}'
            os.mkdir(projdir(branch_dir))
            print(f'make dir {projdir(branch_dir)}')

        js = {}
        try:
            with open(projdir(fn + '.json') , 'r') as f:
                js = json.load(f)
        except:
            print('[not found]', fn + '.json')
        js['epoch'] = e
        js['disturb'] = a.mag

        t1, t2 = load_dataset(threads=2)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        js['trl'], js['tra'] = eval_loss(net, criterion, t1)
        js['tel'], js['tea'] = eval_loss(net, criterion, t2)

    save_file({
        'param': param,
        'buf': buf,
        'nbt': nbt
        }, projdir(branch_dir, f'model_{e}.safetensors'))
    with open(projdir(branch_dir, f'model_{e}.json'), 'w') as fil:
        json.dump(js, fil)
    try:
        shutil.copy(projdir(f"opt_state_{from_epoch}.t7"), projdir(branch_dir, f"opt_state_{e}.t7"))
    except Exception as exc:
        print(exc)
    return js

class ArgsDistance(BaseModel):
    proj: str
    selection: List[str]

@app.post('/distance')
async def _(a: ArgsDistance):
    di = []
    if len(a.selection) < 2:
        return di
    proj = model_dir + a.proj
    for pa in os.listdir(proj): pass

    for i in range(len(a.selection)):
        ti = cat(proj, pa, *translate_u_path(a.selection[i]))
        with safe_open(ti, framework='pt', device='cuda:0') as f:
            pi: np.ndarray = f.get_tensor('param')

        for j in range(i + 1, len(a.selection)):
            tj = cat(proj, pa, *translate_u_path(a.selection[j]))
            with safe_open(tj, framework='pt', device='cuda:0') as f:
                pj: np.ndarray = f.get_tensor('param')
            di.append(torch.norm(pi - pj).item())

    # assert len(a.selection) == 2
    # his_file = cat(proj, pa, 'distance_history.json')
    # with open(his_file, 'a', encoding='utf-8') as f:
        # f.write(f'{a.selection[0]}\t{a.selection[1]}\t{di[0]}\n')


    return di

# @app.get('/distance_history')
# async def _(proj: str):
#     proj = model_dir + proj
#     for pa in os.listdir(proj): pass
#     his_file = cat(proj, pa, 'distance_history.json')
    
#     if not os.path.exists(his_file):
#         return []
#     li = []
#     with open(his_file, 'r', encoding='utf-8') as f:
#         for elem in f.read().split('\n'):
#             li.append(list(elem.split('\t')))
#             # p1, p2, dist = elem.split('\t')
#             # li.append([p1, p2, float(dist)])
#     return li

class ArgsDumpSnapshot(BaseModel):
    proj: str
    j: str

@app.post('/snapshot')
async def _(a: ArgsDumpSnapshot):
    proj = model_dir + a.proj
    js = json.loads(a.j)
    js['t'] = datetime.datetime.now().timestamp()
    for pa in os.listdir(proj): pass
    with open(cat(proj, pa, 'snapshots.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(js) + '\n')

@app.get('/snapshot')
async def _(proj: str):
    proj = model_dir + proj
    for pa in os.listdir(proj): pass
    filename = cat(proj, pa, 'snapshots.jsonl')
    if not os.path.exists(filename): return []
    with open(filename, 'r', encoding='utf-8') as f:
        ret = f.read().split('\n')
    while ret and not ret[-1]:
        ret.pop()
    return ret

class ArgsDeleteSnapshot(BaseModel):
    proj: str
    index: int
@app.delete('/snapshot')
async def _(a: ArgsDeleteSnapshot):
    proj = model_dir + a.proj
    for pa in os.listdir(proj): pass
    filename = cat(proj, pa, 'snapshots.jsonl')
    if not os.path.exists(filename):
        raise HTTPException(404)
    with open(filename, 'r+', encoding='utf-8') as f:
        ret = f.read().split('\n')
        ret.pop(a.index)
        f.seek(0)
        f.write('\n'.join(ret)) # 有后导换行，不用另外写 + '\n' 大概
        f.truncate()








if __name__ == "__main__":
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
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
