import argparse
import random
import torch
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def merge_config(conf1, conf2):
    if isinstance(conf1, dict) and isinstance(conf2, dict):
        new_config = {}
        key_list = list(set(conf1.keys()).union(set(conf2.keys())))

        for key in key_list:
            if (key in conf1) and (key in conf2): # union of c1 & c2
                new_config[key] = merge_config(conf1.get(key), conf2.get(key))
            else:
                new_config[key] = conf1.get(key) if key in conf1 else conf2.get(key)
        return new_config
    else:
        return conf1 if conf2 is None else conf2


def add_config(args, name, config):
    if isinstance(config, dict):
        for key in config.keys():
            add_config(args, key, config[key])
    else:
        setattr(args, name, config)
