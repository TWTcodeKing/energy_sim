# specific count method for different ops, which is user defined
import numpy as np
import sys
import time
import os
logger_file = "mac_ac.dat"

E_mac = 4.6
E_ac = 0.9


def tensor_pow(ops1,ops2,attrs):
    mac = attrs['mac']
    ops = ops1 if ops1 is not None else ops2
    total_f = np.cumprod(ops,axis=0)[-1]
    layer_cnt = total_f * total_f
    if mac:
        return layer_cnt,0,None,ops
    else:
        return 0,layer_cnt,None,ops
    pass


def fft_r2c(ops1,ops2,attrs):
    if ops1 is not None and ops2 is not None:
        ops = ops1 if len(ops1) > len(ops2) else ops2
    else:
        ops = ops1 if ops1 is not None else ops2
    dim = attrs['dim']
    temp_ops = [*ops]
    temp_ops[dim] = 1
    rest_f = np.cumprod(temp_ops, axis=0)[-1]
    #  ops[dim] must be even!!!
    x_n = ops[dim]
    x_k = ops[dim] // 2 + 1
    mac = x_n * x_k * rest_f
    ops[dim] = x_k
    return mac,0,None,ops


def fft_c2r(ops1,ops2,attrs):
    if ops1 is not None and ops2 is not None:
        ops = ops1 if len(ops1) > len(ops2) else ops2
    else:
        ops = ops1 if ops1 is not None else ops2
    dim = attrs['dim']
    temp_ops = ops
    temp_ops[dim] = 1
    rest_f = np.cumprod(ops,axis=0)[-1]
    x_k = ops[dim]
    x_n = (ops[dim] - 1) * 2
    mac = x_n * x_k * rest_f
    ops[dim] = x_n
    return mac, 0, None, ops


def matrix_mul(ops1,ops2,attrs):
    mac = attrs['mac']
    # ops1 : weight shape [in_f,out_f]
    # ops2 : input tensor shape [x,x,x,in_f]
    if ops1 is None or ops2 is None:
        return 0,0,ops1,ops2
    ops_temp = ops1
    if len(ops1) > len(ops2):
        ops1 = ops2
        ops2 = ops_temp
    _unused_f = np.cumprod(ops2, axis=0)[-2]  # after Batch dim(including T)
    in_f = ops1[1]
    out_f = ops1[0]
    layer_cnt = 1.0
    layer_cnt *= _unused_f
    layer_cnt *= _unused_f
    layer_cnt *= in_f
    layer_cnt *= out_f
    layer_cnt /= ops2[0]
    ops2[-1] = ops1[0]
    if mac:
        return layer_cnt,0,None,ops2
    else:
        return 0,layer_cnt,None,ops2


def batch_matrix_mul(ops1,ops2,attrs):
    mac = attrs['mac']
    if ops1 is None or ops2 is None:
        return 0,0,ops1,ops2
    ops_temp = ops1
    if len(ops1) > len(ops2):
        ops1 = ops2
        ops2 = ops_temp
    _unused_f = np.cumprod(ops2, axis=0)[-2]  # after Batch dim(including T)
    in_f = ops1[1]
    out_f = ops1[0]
    layer_cnt = 1.0
    layer_cnt *= _unused_f
    layer_cnt *= _unused_f
    layer_cnt *= in_f
    layer_cnt *= out_f
    layer_cnt /= ops2[0]
    ops2[-1] = ops1[0]
    if mac:
        return layer_cnt,0,None,ops2
    else:
        return 0,layer_cnt,None,ops2
    pass


def tensor_add(ops1,ops2,attrs):
    # bias add, already counted
    if ops1 is None or ops2 is None:
        return 0,0,None,ops1 if ops1 is not None else ops2
    else:
        total_f1 = np.cumprod(ops1, axis=0)[-1]
        total_f2 = np.cumprod(ops2,axis=0)[-1]
        if total_f1 > total_f2:
            total_f = total_f1
            ops = ops1
        else:
            total_f = total_f2
            ops = ops2
        return 0, total_f, None,ops
    pass


def matrix_dot(ops1,ops2,attrs):
    mac = attrs['mac']
    if ops1 is None or ops2 is None:
        return 0,0,None,ops1 if ops1 is not None else ops2
    total_f1 = np.cumprod(ops1, axis=0)[-1]
    total_f2 = np.cumprod(ops2, axis=0)[-1]
    if total_f1 > total_f2:
        total_f = total_f1
        ops = ops1
    else:
        total_f = total_f2
        ops = ops2
    if mac:
        return total_f,0,None,ops
    else:
        return 0,total_f,None,ops
    pass


def embedding(ops1,ops2,attrs):
    # shape change ops
    new_ops = [*ops2,ops1[-1]]
    return 0,0,None,new_ops
    pass


def unsqueeze(ops,attrs):
    dim = attrs['dim']
    ops.insert(dim,1)
    return ops
    pass


def squeeze(ops,attrs):
    dim = attrs['dim']
    ops[dim] = 0
    ops.remove(0)
    return ops
    pass


def sigmoid(ops1,ops2=None,mac=False):
    # not included
    pass


def tanh(ops1,ops2=None,mac=False):
    # not included
    pass


def conv2d(ops1,ops2,attrs):
    # ops1 [out_chan,in_chan,kn1,kn2,stride,padding]
    padding = ops1[-1]
    stride = ops1[-2]
    mac = attrs['mac']
    in_chan = ops1[1]
    out_chan = ops1[0]
    kn1 = ops1[2]
    kn2 = ops1[3]
    hn, wn = ops2[-2:]
    layer_cnt = kn1 * kn2 * hn * wn * in_chan * out_chan * ops2[0]
    new_hn = (hn-kn1+2*padding[0]) // stride[0] + 1
    new_wn = (wn-kn2+2*padding[1]) // stride[1] + 1
    if new_wn <= 0 or new_wn <= 0:
        padding = ops1.pop(-1)
        stride = ops1.pop(-1)
        ops2.append(stride)
        ops2.append(padding)
        return conv2d(ops2,ops1,attrs)
    if len(ops2) == 5:
        new_ops2 = [ops2[0],ops2[1],out_chan,new_hn,new_wn]  # at least it is like this, but T,B,C,L,D
    else:
        new_ops2 = [ops2[0],out_chan,new_hn,new_wn]
    if mac:
        return layer_cnt,0,None,new_ops2
    else:
        return 0,layer_cnt,None,new_ops2
    # change shape of ops2
    pass


def conv1d(ops1,ops2,attrs):
    # ops1 [out_chan,in_chane,kn,stride,padding]
    mac = attrs['mac']
    in_chan = ops1[1]
    out_chan = ops1[0]
    kn = ops1[2]
    stride = ops1[-2]
    padding = ops1[-1]
    ln = ops2[-1]
    layer_cnt = kn * ln * in_chan * out_chan * ops2[0]
    new_ln = (ln - kn + 2 * padding) / stride + 1
    if len(ops2) == 4:
        new_ops2 = [ops2[0], ops2[1], out_chan, new_ln]  # at least it is like this, but T,B,C,L,D
    else:
        new_ops2 = [ops2[0], out_chan,new_ln]
    if mac:
        return layer_cnt, 0, None, new_ops2
    else:
        return 0, layer_cnt, None, new_ops2
    pass


def expand(ops,dim):
    pass


def cat(ops,attrs):
    dim = attrs['dim']
    num_par_edges = attrs['num_par_edges']
    ops[dim] = ops[dim] * num_par_edges
    return ops
    pass


def mean_sum(ops,attrs):
    dim = attrs['dim']
    if dim > 10:
        # dim = -1
        dim = len(ops) - 1
    keep_dim = attrs['keepdim']
    if keep_dim:
        ops[dim] = 1
    else:
        ops = squeeze(ops, attrs)
    return ops
    pass


def stack(ops,attrs):
    dim = attrs['dim']
    num = attrs['num_par_edges']
    ops.insert(dim,num)
    return ops
    pass


def maxpool_2d(ops,attrs):
    kn1,kn2 = attrs['kernel_size']
    padding1,padding2 = attrs['padding']
    stride1,stride2 = attrs['stride']
    hn,wn = ops[-2:]
    hn = (hn + padding1 * 2 - kn1) // stride1 + 1
    wn = (wn + padding2 * 2 - kn2) // stride2 + 1
    ops[-2] = hn
    ops[-1] = wn
    return ops


def norm():
    return 1.0,True


def atan():
    return 0.1,False


def lnorm(ops):
    mac = 0
    ac = 0
    mean_mac = np.cumprod(ops, axis=0)  # u = x.mean(-1, keepdim=True) add and mul
    mac += mean_mac[-1]
    ac += mean_mac[-1]  # (x - u)
    mac += mean_mac[-1]  # (x - u).pow(2)
    mac += mean_mac[-1]  # (x - u).pow(2).mean(-1, keepdim=True)
    ac += mean_mac[-1]  # (x - u)
    ac += mean_mac[-2]  # s + self.variance_epsilon s shape [B,...,1]
    mac += 2 * mean_mac[-1]  # (x - u) / torch.sqrt(s + self.variance_epsilon)
    mac += mean_mac[-1]
    return mac,ac
    pass


def asynchronous_logger(q):
    # init logger
    def sum_mac_ac_over_log():
        total_mac = 0
        total_ac = 0
        with open(logger_file,"r") as lf:
            while True:
                ops_energy = lf.readline()
                if not ops_energy:
                    break
                ops_energy = ops_energy.strip()
                str_mac,str_ac = ops_energy.split(",")
                int_mac = float(str_mac.split(":")[1])
                int_ac = float(str_ac.split(":")[1])
                total_mac += int_mac
                total_ac += int_ac
        lf.close()
        return total_mac,total_ac
        pass
    import logging
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logger_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    dead_pc = 0
    while True:
        info = q.get()
        status, info1, info2 = info
        if status == 0:
            logger.info(f"mac:{info1},ac:{info2}")
        elif status == 1:
            dead_pc += 1
            progress_bar(dead_pc,info1)
        # else:
        #     logger.info(f"node:{status},ops1:{info1},ops2:{info2}")
        if status == 2:  # finish all
            dead_pc += 1
            progress_bar(dead_pc, info1)
        if dead_pc == info1 and dead_pc != 0:
            progress_bar(dead_pc, info1)
            break
    mac, ac = sum_mac_ac_over_log()
    logger.info(f"total mac:{mac},ac:{ac}")
    energy = (mac * E_mac + ac * E_ac) * 1e-6
    print(f"\n simulated total energy consumption(μJ):{energy}\n")
    fh.close()
    os.remove(logger_file)
    pass


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    sys.stdout.write(f'\r[{bar}] {percent:.2f}%')
    sys.stdout.flush()


if __name__ == "__main__":
    # import sys
    # import time
    import torch
    from torch import nn
    # a = torch.randn(1,20,100)
    # b = torch.randn(1,20,100)
    # c = torch.stack([a,b],dim=1)
    # print(c.shape)
    # a = [1,20,100]
    # a.insert(1,2)
    # print(a)
    a = torch.randn(1,3,5,21)
    conv = nn.Conv2d(3,3,kernel_size=(3,3),padding=(1,1),stride=(1,1))
    pool = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
    print(conv(pool(a)).shape)
    pass
