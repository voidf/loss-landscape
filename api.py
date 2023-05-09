import base64
import copy
import datetime
import functools
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import traceback
from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, HTTPException, Query, Request,
                     Response)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from safetensors import safe_open
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from torch import Tensor

import evaluation
from cifar10.model_loader import load
from dataloader import load_dataset
from evaluation import eval_loss
from main import init_params
from net_plotter import get_weights
from pref import find_arch
from scan_traj import (cat_tensor, get_buf_no_nbt, get_nbt, get_states,
                       write_buf_no_nbt, write_nbt, write_states,
                       write_weights)
from wrappers import cat

torch.backends.cuda.matmul.allow_tf32 = True

MODEL_DIR = 'trained/'
WORKER_CNT = 8
PCA_CACHE_DIR = '_pca_cache/'
PCA_CACHE_CNT = 200
# pool = None


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
    app.add_middleware(  # 允许跨域第一版
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware('http')  # TODO: [insecure] set to argu fixed origin
    async def cors_everywhere(request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = request.headers.get(
            'origin', '*')
        return response

    # app.include_router(v1_router)
    return app


app = preload()


@app.get('/models')
async def _():
    l = []
    for i in os.listdir(MODEL_DIR):
        l.append(i)
    return l


@app.get('/list')
async def _(proj: str):
    proj = MODEL_DIR + proj
    d = {}
    root = True
    for i in os.listdir(proj):
        pass  # 跳过超参记录夹
    for dp, dn, fn in os.walk(cat(proj, i)):
        sp = dp.removeprefix(cat(proj, i)).split(os.sep)
        t7s = [i for i in fn if i.endswith(
            '.safetensors') and i.startswith('model_')]
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
            for p, i in enumerate(pk):  # 拆目录
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
    l.sort(key=functools.cmp_to_key(lcmp)  # 保证branch显示顺序
           # lambda it: it['p'] + it['p2'] + [it['x']]
           )
    for it in l:
        it['l'] = p3s = cat(*it.pop('p3'))
        it.pop('kp')
        it['y'] = dd.setdefault(p3s, len(dd))
        it['u'] = cat(p3s, str(it['x'])).removeprefix('/')

    return {'points': l}


class ArgsPCA(BaseModel):
    selection: List[str]
    proj: str
    weight: bool = True
    incr: int = 0


def translate_u_path(u: str, dir_only=False, suffix='safetensors') -> List[str]:
    path = u.split('/')
    m = f'model_{path.pop()}.{suffix}'
    for p, i in enumerate(path):
        f, b = i.split('.')
        path[p] = f'model_{f}B{b}'
    if not dir_only:
        path.append(m)
    return path

def get_tensor_from_file(fp: str):
    with safe_open(fp, framework='np') as fil:
        param: np.ndarray = fil.get_tensor('param')
        try:
            buf: np.ndarray = fil.get_tensor('buf')
        except:
            buf = np.array([])
    return param, buf

def gather_selected_tensors(proj: str, selection: list[str], only_weight=True, batch=0):
    if len(selection) == 0:
        raise HTTPException(400, 'empty selection')
    proj = MODEL_DIR + proj

    nplist = []
    # avg = None

    # weight_len = sum(map(lambda x: reduce(operator.mul, x.data.shape), net.parameters()))

    for j in os.listdir(proj):
        for i in selection:
            fp = cat(proj, j, *translate_u_path(i))
            param, buf = get_tensor_from_file(fp)
                # nbt: np.ndarray = fil.get_tensor('nbt')
            # if calc_avg:
            #     if avg is None:
            #         avg = cated.copy()
            #     else:
            #         avg += cated
            # ts, t7 = t7_to_tensor(s.arch, cat(proj, j, *translate_u_path(i)), True)
            # ts = ts.numpy() # 输出： param + buf
            if only_weight:
                nplist.append(param)
            else:
                cated = np.concatenate((param, buf))
                nplist.append(cated)
            if batch > 0 and len(nplist) == batch:
                ret = np.array(nplist)
                nplist.clear()
                yield ret
            # nplist.append(ts[:weight_len])

    # nplist = np.array(nplist)
    if nplist:
        yield np.array(nplist)
    # if not calc_avg:
        # return nplist
    # avg /= len(nplist)
    # return avg, nplist


def generate_pca_cache_fn(proj: str, selection: list[str]):
    dig = hashlib.md5(','.join(selection).encode()).digest()
    b64 = f'{proj}-' + base64.b64encode(dig, altchars=b'+^').decode('utf-8')
    return PCA_CACHE_DIR + b64 + '.pkl'


def ensure_pca_cache(proj: str, selection: list[str], dim=2, only_weight=True, save_axis=True, increment=0) -> dict:
    # selection.sort() # 这个sort会打乱前端的显示
    fn = generate_pca_cache_fn(
        proj + f'-{dim}d-' + ['', 'ow-'][only_weight] + ['', 'ax-'][save_axis] + ['', f'incr{increment}-'][increment>0], sorted(selection))
    if os.path.exists(fn):
        print('cache found:', fn)
        with open(fn, 'rb') as f:
            return pickle.load(f)

    if increment > 0:
        pca = IncrementalPCA(dim, batch_size=increment)
        newlist = []
        for batch in gather_selected_tensors(proj, selection, only_weight, increment):
            pca.partial_fit(batch)
        for batch in gather_selected_tensors(proj, selection, only_weight, increment):
            newlist.extend(pca.transform(batch))

    else:
        nplist = next(gather_selected_tensors(proj, selection, only_weight, increment))
        pca = PCA(dim)
        newlist = pca.fit_transform(nplist).tolist()

    for p, i in enumerate(newlist):
        newlist[p] = {
            'x': i[0],
            'y': i[1],
            'u': selection[p],
            'l': cat(*selection[p].split('/')[:-1]),
        }

    if len(cached_files := os.listdir(PCA_CACHE_DIR)) >= PCA_CACHE_CNT:

        for p, i in enumerate(cached_files):
            cached_files[p] = (os.path.getatime(PCA_CACHE_DIR + i), i)
        cached_files.sort(key=lambda x: -x[0])
        while len(cached_files) >= PCA_CACHE_CNT:
            print('removed pca cache:', cached_files[-1])
            os.remove(PCA_CACHE_DIR + cached_files.pop()[1])

    d = {
        # 'mean': avg.tolist(),
        'coord': newlist,
    }
    if save_axis:
        d['axis'] = pca.components_[:dim].tolist()
    with open(fn, 'wb') as f:
        pickle.dump(d, f)
    return d


@app.post('/pca')
async def _(a: ArgsPCA):
    return {'coord': ensure_pca_cache(a.proj, a.selection, 2, a.weight, increment=a.incr)['coord']}


@app.post('/tsne')
async def _(a: ArgsPCA, pre_pca: int = 1):
    if pre_pca:
        pre_pca = ensure_pca_cache(
            a.proj, a.selection, dim=min(len(a.selection), 50), only_weight=a.weight, save_axis=False, increment=a.incr)  # 先投影为50维
        nplist = pre_pca['coord']
        logger.debug('pre_pca len: {}', len(nplist))
    else:
        nplist = gather_selected_tensors(a.proj, a.selection, a.weight, batch=0)

    tsne = TSNE(2, perplexity=10, n_iter=3000, learning_rate='auto')
    newlist = tsne.fit_transform(np.array(nplist)).tolist()
    # axis: np.ndarray = tsne.components_[:2]
    for p, i in enumerate(newlist):
        newlist[p] = {
            'x': i[0],
            'y': i[1],
            'u': a.selection[p],
            'l': cat(*a.selection[p].split('/')[:-1]),
        }
    return {
        # 'axis': axis.tolist(),
        # 'mean': avg.tolist(),
        'coord': newlist,
    }


class ArgsMeta(BaseModel):
    selection: List[str]
    proj: str


@app.post('/meta')
async def _(argu: ArgsMeta):
    logger.info(argu)
    if len(argu.selection) == 0:
        raise HTTPException(400, 'empty selection')
    proj = MODEL_DIR + argu.proj
    meta = []
    for j in os.listdir(proj):
        for i in argu.selection:
            with open(cat(proj, j, *translate_u_path(i)).removesuffix('safetensors') + 'json') as fil:
                t7 = json.load(fil)
            meta.append(t7)
    return meta


@app.get('/info')
async def _(p: str, proj: str):
    path = u_resolver([p], MODEL_DIR + proj)[0]
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


# 插值目录规范：{u1}_{u2}_x/y 表示在u1和u2之间线性插值x/y倍
# zlib.compress
# 在meta接口里带一个clamp的key给插值目录信息，不写学习率、随机种子、wd等等
# 在clamp接口里创建这些新分支点
# 前端靠clamp的key来划线，在meta读入时处理这个labelsu

@app.post('/train')
async def _(argu: ArgsTrain, r: Response):
    path = u_resolver([argu.u], MODEL_DIR + argu.proj)[0]
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

    proj = MODEL_DIR + a.proj
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

    saved_dict = {
        'param': cat_tensor(get_weights(net)),
    }
    # 这俩不一定有
    if x := get_buf_no_nbt(net):
        saved_dict['buf'] = cat_tensor(x)
    if x := get_nbt(net):
        saved_dict['nbt'] = cat_tensor(x)

    save_file(saved_dict, projdir('model_0.safetensors'))
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
    selection: List[str]
    xstep: int
    ystep: int
    xstep_rate: float
    ystep_rate: float
    # mean: Optional[List[float]] = None
    # xdir: List[float]
    # ydir: List[float]
    u: Optional[str]  # = None
    proj: str


@app.post('/heatmap')
async def _(argu: ArgsHeatmap):
    pca_data = ensure_pca_cache(argu.proj, argu.selection, 2, True)

    xdir = pca_data['axis'][0]
    ydir = pca_data['axis'][1]
    mea = pca_data.get('mean', [])

    mng = mp.Manager()
    q1 = mng.Queue(maxsize=WORKER_CNT)
    q2 = mng.Queue()
    consumers = [
        mp.Process(target=evaluation.epoch_consumer,
                   args=(find_arch(argu.proj), q1, q2)
                   ) for _ in range(WORKER_CNT)
    ]
    for x in consumers:
        x.start()

    net = load(find_arch(argu.proj))
    if argu.u:
        proj = MODEL_DIR + argu.proj

        for j in os.listdir(proj):
            with safe_open(cat(proj, j, *translate_u_path(argu.u)), framework="pt", device='cpu') as fil:
                # param = fil.get_tensor('param')
                # buf = fil.get_tensor('buf')
                write_weights(net, param := fil.get_tensor('param'))
                try:
                    write_buf_no_nbt(net, buf := fil.get_tensor('buf'))
                except:
                    buf = torch.tensor([])

            mea = torch.cat((param, buf))
            # mea = t7_to_tensor(argu.arch, cat(proj, j, *translate_u_path(argu.u)))
    else:
        mea = torch.tensor(mea)
        write_states(net, mea)

    if len(mea) > len(xdir):
        weight_mode = True
        mea = mea[:len(xdir)]
    elif len(mea) == len(xdir):
        weight_mode = False
    else:
        raise HTTPException(500, f'mean shape:{len(mea)} != {len(xdir)}')
    #     # weight mode
    #     ts: Tensor = torch.zeros(len(mea))
    #     ts[:len(xdir)] = xdir[:]
    #     xdir = ts
    #     ts[:len(ydir)] = ydir[:]
    #     ydir = ts
    # else:
    xdir = torch.tensor(xdir)
    ydir = torch.tensor(ydir)

    # needle = copy.deepcopy(net)
    with torch.no_grad():
        for x in range(-argu.xstep, argu.xstep + 1):
            for y in range(-argu.ystep, argu.ystep + 1):
                cur = mea + x * argu.xstep_rate * xdir + y * argu.ystep_rate * ydir
                if weight_mode:
                    write_weights(net, cur)
                else:
                    write_states(net, cur)
                q1.put(((x, y), copy.deepcopy(net.state_dict())))
            # set_weights(needle, get_weights)

    for _ in consumers:
        q1.put(None)
    for x in consumers:
        x.join()

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
        proj = MODEL_DIR + a.proj
        from_epoch = int(a.u.rsplit('/', 1)[-1])
        fn = f'model_{from_epoch}'
        e = from_epoch + 1
        for pa in os.listdir(proj):
            pass
        projdir = functools.partial(
            cat, proj, pa, *translate_u_path(a.u, dir_only=True))

        with safe_open(projdir(fn + '.safetensors'), framework='pt', device='cuda') as fil:
            param: torch.Tensor = fil.get_tensor('param')
            try:
                buf: torch.Tensor = fil.get_tensor('buf')
            except:
                buf = torch.tensor([])
            try:
                nbt: torch.Tensor = fil.get_tensor('nbt')
            except:
                nbt = torch.tensor([])

        init_params(net)
        random_param = cat_tensor(get_weights(net)).to('cuda')  # 用kaiming_normal_

        param += a.mag * random_param  # [-a.mag, a.mag] kaiming_normal_分布
        write_weights(net, param)
        write_buf_no_nbt(net, buf)
        write_nbt(net, nbt)

        if not os.path.exists(projdir(f'model_{e}.safetensors')):
            branch_dir = '.'
        else:
            idx = 1
            s = set()  # MEX
            for i in os.listdir(projdir()):
                if os.path.isdir(projdir(i)) and i.startswith(fn + 'B'):
                    if len(os.listdir(projdir(i))) == 0:
                        os.rmdir(projdir(i))
                    else:
                        s.add(int(i.rsplit('B', 1)[1]))
            while idx in s:
                idx += 1

            branch_dir = f'{fn}B{idx}'
            os.mkdir(projdir(branch_dir))
            print(f'make dir {projdir(branch_dir)}')

        js = {}
        try:
            with open(projdir(fn + '.json'), 'r') as f:
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
        shutil.copy(projdir(f"opt_state_{from_epoch}.t7"), projdir(
            branch_dir, f"opt_state_{e}.t7"))
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
    proj = MODEL_DIR + a.proj
    for pa in os.listdir(proj):
        pass

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
    proj = MODEL_DIR + a.proj
    js = json.loads(a.j)
    js['t'] = datetime.datetime.now().timestamp()
    for pa in os.listdir(proj):
        pass
    with open(cat(proj, pa, 'snapshots.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(js) + '\n')


@app.get('/snapshot')
async def _(proj: str):
    proj = MODEL_DIR + proj
    for pa in os.listdir(proj):
        pass
    filename = cat(proj, pa, 'snapshots.jsonl')
    if not os.path.exists(filename):
        return []
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
    proj = MODEL_DIR + a.proj
    for pa in os.listdir(proj):
        pass
    filename = cat(proj, pa, 'snapshots.jsonl')
    if not os.path.exists(filename):
        raise HTTPException(404)
    with open(filename, 'r+', encoding='utf-8') as f:
        ret = f.read().split('\n')
        ret.pop(a.index)
        f.seek(0)
        f.write('\n'.join(ret))  # 有后导换行，不用另外写 + '\n' 大概
        f.truncate()


class ArgsClamp(BaseModel):
    u1: str
    u2: str
    ctr: int
    proj: str


@app.post('/clamp')
async def _(a: ArgsClamp):
    from safetensors.torch import save_file

    from scan_traj import generate_mex_branch_dir
    from wrappers import lerp
    proj = MODEL_DIR + a.proj
    arch = find_arch(a.proj)
    for pa in os.listdir(proj):
        pass
    with torch.no_grad():
        with (safe_open(u1path := cat(proj, pa, *translate_u_path(a.u1)), framework='pt', device='cuda:0') as f1,
              safe_open(cat(proj, pa, *translate_u_path(a.u2)), framework='pt', device='cuda:0') as f2):
            d = {}
            for sta in ['param', 'buf']:
                try:
                    d[sta] = [f1.get_tensor(sta), f2.get_tensor(sta)]
                except:
                    print(sta, 'no found in checkpoint!')
            assert 'param' in d
        sp = u1path.rsplit('/', 1)
        assert len(sp) > 1
        from_epoch = int(sp.pop().removeprefix(
            'model_').removesuffix('.safetensors'))
        sp = '/'.join(sp)

        net = load(arch)
        mng = mp.Manager()
        q1 = mng.Queue(maxsize=WORKER_CNT)
        q2 = mng.Queue()
        consumers = [
            mp.Process(target=evaluation.epoch_consumer,
                       args=(arch, q1, q2)
                       ) for _ in range(WORKER_CNT)
        ]
        branch_dirs = []
        for x in consumers:
            x.start()
        for i in range(a.ctr):
            ckp = {k: lerp(v1, v2, (i + 1) / (a.ctr + 1))
                   for k, (v1, v2) in d.items()}
            branch_dir = generate_mex_branch_dir(sp, from_epoch)
            branch_dirs.append(branch_dir)
            save_file(ckp, cat(sp, branch_dir,
                      f'model_{from_epoch + 1}.safetensors'))
            write_weights(net, ckp['param'])
            if 'buf' in ckp:
                write_buf_no_nbt(net, ckp['buf'])
            net.cpu()
            q1.put((i, copy.deepcopy(net.state_dict())))
        for _ in consumers:
            q1.put(None)
        for x in consumers:
            x.join()
        r = []
        for _ in range(a.ctr):
            # tid, train_loss, train_acc, test_loss, test_acc = q2.get()
            r.append(q2.get())
        r.sort(key=lambda x: x[0])
        for i, train_loss, train_acc, test_loss, test_acc in r:
            with open(cat(sp, branch_dirs[i], f'model_{from_epoch + 1}.json'), 'w') as f:
                r[i] = {
                    'tea': test_acc,
                    'tel': test_loss,
                    'tra': train_acc,
                    'trl': train_loss,
                    'epoch': from_epoch + 1,
                    'clamp': f'{a.u1}_{a.u2}_{i+1}/{a.ctr + 1}'
                }
                json.dump(r[i], f)
        return r


if __name__ == "__main__":
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    if not os.path.exists(PCA_CACHE_DIR):
        os.mkdir(PCA_CACHE_DIR)
    global t_manager, t_queue, t_consumers
    t_manager = mp.Manager()
    t_queue = t_manager.Queue()
    t_consumers = [
        mp.Process(target=train_consumer,
                   args=(t_queue,)
                   ) for _ in range(WORKER_CNT)
    ]
    for x in t_consumers:
        x.start()
    uvicorn.run(app, port=40000)
