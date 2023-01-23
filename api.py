import fastapi
import os
import uvicorn
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
pool = None

from fastapi import FastAPI, Query, Request

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
    for i in os.listdir('trained'):
        l.append(i)
    return l

@app.get('/list')
async def _(proj: str):
    from wrappers import cat
    proj = 'trained/' + proj
    d = {}
    # root: list = d.setdefault('root', [])
    root = True
    for i in os.listdir(proj):
        for dp, dn, fn in os.walk(cat(proj, i)):
            sp = os.path.split(dp)
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
        # pk2 = []
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
            # w: str
            l.append({
                # 'id': len(l), 
                # 'p': pk,
                # 'p2': pk2,
                'kp': kp,
                'p3': pk3,
                # 'p3s': p3s,
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

    return {'label': dd, 'points': l}




if __name__ == "__main__":
    pool = ProcessPoolExecutor(4)

    uvicorn.run(app, port=40000)
