from itertools import chain
import os
from safetensors import safe_open
from safetensors.torch import save_file
from dataloader import load_dataset
from evaluation import eval_loss
from net_plotter import get_weights
from scan_traj import get_buf_no_nbt, get_nbt, write_buf_no_nbt, write_nbt, write_states, cat_tensor, write_weights
from wrappers import cat
from pref import find_arch
from cifar10.model_loader import load
import torch
from torch.optim import SGD
# ar = 'resnet56_noshort'
import json
from api import t7_to_tensor
from math import prod

# tr, te = load_dataset(threads=1)

if __name__ == '__main__':

    for d in os.listdir('trained'):
        if d.startswith('D121'):
            ar = find_arch(d)
            net = load(ar)
            for di in os.listdir(d1 := cat('trained', d)):
                for dirpath, _, fn in os.walk(d2 := cat(d1, di)):
                    for t7 in fn:
                        if t7.endswith('.t7') and t7.startswith('model_'):
                        # if t7.endswith('.safetensors') and t7.startswith('model_'):

                            # if os.path.exists(t7path + '.net1'):
                            #     os.remove(t7path + '.net1')
                            #     print('del', t7path + '.net1')
                            # if os.path.exists(t7path + '.net2'):
                            #     os.remove(t7path + '.net2')
                            #     print('del', t7path + '.net2')

                            t7path = cat(dirpath, t7)
                            print(t7path)
                            def remove_ckpt():
                                os.remove(t7path)
                                print('remove', t7path)

                            def convert_ckpt():
                                filen = t7path.removesuffix('t7') + 'safetensors'
                                if os.path.exists(filen):
                                    # continue
                                    return

                                t7file = torch.load(t7path)
                                net.load_state_dict(t7file.pop('state_dict'))

                                param = cat_tensor(get_weights(net))
                                buf = cat_tensor(get_buf_no_nbt(net))
                                nbt = cat_tensor(get_nbt(net))
                                # ts, mt = t7_to_tensor(ar, t7path, True, skip_num_batches_tracked=False)
                                # c: dict = torch.load(cat(dirpath, t7))
                                # states = c.pop('state_dict')
                                tensors = {'param': param, 'buf': buf, 'nbt': nbt}

                                with open(t7path.removesuffix('t7') + 'json', 'w') as f:
                                    json.dump(t7file, f)
                                
                                # if os.path.exists(cat(dirpath, opt := 'opt_state' + t7.removeprefix('model'))):
                                    # tensors['opt'] = t7_to_tensor()
                                
                                save_file(tensors, filen)
                                print(t7path, '=>', filen)
                            def json_desc_update():
                                with open(jfile := t7path.removesuffix('safetensors') + 'json', 'r+') as f:
                                    js = json.load(f)
                                    print(js, '=>', end=' ')
                                    if 'train_loss' in js:
                                        js['trl'] = js.pop('train_loss')
                                        js['tra'] = 100. - js.pop('train_err')
                                        js['tel'] = js.pop('test_loss')
                                    if 'acc' in js:
                                        js['tea'] = js.pop('acc')
                                    f.seek(0)
                                    json.dump(js, f)
                                    print(js)
                                    f.truncate()
                                # with open(jfile, 'w') as f:
                                    # js = json.load(f)



                            convert_ckpt()
                            remove_ckpt()
                            # json_desc_update()

                            # def convert_opt():
                            #     oppath = cat(dirpath, 'opt_state'+t7.removeprefix('model'))
                            #     sfpath = oppath.removesuffix('t7') + 'safetensors'
                            #     sd = torch.load(oppath)
                            #     opt = SGD(net.parameters(), 0.1)
                            #     opt.load_state_dict(sd['optimizer'])
                            #     # assert opt.state_dict() == sd['optimizer']
                            #     save_file(opt.state_dict(), sfpath)
                            # convert_opt()

    """
                            # verify
                            with torch.no_grad():
                                net1 = load(ar)
                                net1.load_state_dict(torch.load(t7path)['state_dict'])
                                net2 = load(ar)
                                crit = torch.nn.CrossEntropyLoss()
                                with safe_open(filen, framework="pt", device="cpu") as f:
                                    a = f.get_tensor('param')
                                    b = f.get_tensor('buf')
                                    c = f.get_tensor('nbt')
                                write_weights(net2, a)
                                write_buf_no_nbt(net2, b)
                                write_nbt(net2, c)
                                # for (q, w), (e, r) in zip(net1.state_dict().items(), net2.state_dict().items()):
                                #     # print(q, q==e, w==r)
                                #     res = (w == r)
                                #     if torch.sum(res) != prod(w.shape):
                                #         print(q, e, w, r)
                                #     if len(w.shape) == 0:
                                #         print(q, e, w, r)
                                l1, a1 = eval_loss(net1, crit, tr)
                                l2, a2 = eval_loss(net2, crit, tr)
                                print(l1, l2, a1, a2)
                                assert l1 == l2
                                assert a1 == a2

                            # write_states(net2, tt, False)
                            # torch.save(net1.state_dict(), t7path + '.net1')
                            # torch.save(net2.state_dict(), t7path + '.net2')
    """
