"""
Extract prediction score after training
By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import argparse
import os
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils.util import merge_config, add_config

from utils.logger import setup_logger

import numpy as np

from helpers import get_eva_loader, build_model


def parse_option():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--config_file', type=str, required=True, help='path of config file (yaml)')
    parser.add_argument('--crop_idx', type=int, default=0, help='the place index [0,1,2]')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    # load config file, default + base + exp
    config_default = yaml.load(open('./base_config/default.yml', 'r'), Loader=yaml.Loader)
    config_exp = yaml.load(open(args.config_file, 'r'), Loader=yaml.Loader)
    if 'base' in config_exp:
        config_base = yaml.load(open(config_exp['base'], 'r'), Loader=yaml.Loader)
    else:
        config_base = None
    config = merge_config(merge_config(config_default, config_base), config_exp)
    args.C = config
    add_config(args, 'root', config)

    if isinstance(args.epochs, list):
        args.batch_size = args.batch_size[-1]
        args.iter_size = args.iter_size[-1]
        args.base_learning_rate = args.base_learning_rate[-1]
        args.epochs = args.epochs[-1]
        args.num_segments = args.num_segments[-1]
        args.clip_length = args.clip_length[-1]
        args.num_steps = args.num_steps[-1]
        args.frozen_bn = args.frozen_bn[-1]

    # change parameters for evaluation
    args.pretrained_model = args.eva_model
    args.transfer_weights = False
    args.remove_fc = False
    args.crop_size = args.eva_crop_size

    return args


def main(args):
    test_loader = get_eva_loader(args)
    n_data = len(test_loader.dataset)
    logger.info("length of testing dataset: {}".format(n_data))

    model = build_model(args)
    model.eval()

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # routine
    all_scores = np.zeros([len(test_loader) * args.batch_size, args.num_classes], dtype=np.float)
    top_idx = 0
    with torch.no_grad():
        for idx, (x, cls) in enumerate(test_loader):
            if (idx % 100 == 0) or (idx == len(test_loader) - 1):
                logger.info('{}/{}'.format(idx, len(test_loader)))
            bsz = x.size(0)
            score = model(x.cuda())
            # torch.distributed.barrier()
            if isinstance(score, list):
                if len(score) == 2:
                    score_numpy = (score[0].data.cpu().numpy() + score[1].data.cpu().numpy()) / 2
                else:
                    score_numpy = sum([score_i.data.cpu().numpy() for score_i in score]) / len(score)
            else:
                score_numpy = score.data.cpu().numpy()
            all_scores[top_idx: top_idx + bsz, :] = score_numpy
            top_idx += bsz
    all_scores = all_scores[:top_idx, :]
    np.save(os.path.join(args.output_dir, 'all_scores_{}.npy'.format(
        torch.distributed.get_world_size() * args.crop_idx + args.local_rank)), all_scores)


if __name__ == '__main__':
    opt = parse_option()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="p3d")
    opt.logger = logger
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "extract_score_3d.config.yml")
        with open(path, 'w') as f:
            yaml.dump(opt.C, f, default_flow_style=False)
        logger.info("Full config saved to {}".format(path))

    main(opt)
