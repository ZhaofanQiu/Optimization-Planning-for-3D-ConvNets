"""
Train 3d architecture with optimization planning (searching strategy)
By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import argparse
import os
import time
import yaml
import numpy as np

import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from utils.util import merge_config, add_config
from utils.logger import setup_logger
from layer.LSR import *
import itertools

from helpers import get_train_loader, build_model
from train_3d import eva_inter, train

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import shutil

# optimization planning (sparse sampling)
op_params = {
    'num_segments_list': [4, 8, 16],
    'clip_length_list': [4, 4, 4],
    'num_steps_list': [1, 1, 1],
    'base_lr_list': [0.01, 0.01, 0.001],
    'batch_size_list': [16, 8, 4],
    'iter_size_list': [4, 8, 16],
    'frozen_bn_list': [False, False, True],

    'visualize_figure': False,  # If visualize the performance-epoch curve in the pdf file
    'attempt_epochs': 10,  # The number T of attempt epochs

    # params to define the candidate transition graph
    # ext graph
    'start_node_list': [0, 0, 1, 1, 4, 4, 2, 4, 7, 2, 5, 5, 7, 3, 5, 8, 6, 8],    # Start node id for each edge
    'end_node_list':   [1, 4, 4, 2, 2, 7, 5, 5, 5, 3, 3, 8, 8, 6, 6, 6, 9, 9],    # End node id for each edge
    'setting_list': [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2],                             # The assigned input clip length (index) for each node
    'lr_mult_list': [-1, 1, 0.1, 0.01, 1, 0.1, 0.01, 1, 0.1, 0.01],             # The assigned learning rate factor for each node
}


def parse_option():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--config_file', type=str, required=True, help='path of config file (yaml)')
    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')

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

    # base_graph
    edge_select = [0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 16, 17]
    op_params['start_node_list'] = [op_params['start_node_list'][i] for i in edge_select]
    op_params['end_node_list'] = [op_params['end_node_list'][i] for i in edge_select]
    return args


def get_op_train_loaders(args):
    train_loaders = []
    # train_loader_0
    if 0 in op_params['setting_list']:
        args.num_segments = op_params['num_segments_list'][0]
        args.clip_length = op_params['clip_length_list'][0]
        args.num_steps = op_params['num_steps_list'][0]
        args.batch_size = op_params['batch_size_list'][0]
        train_loaders.append(get_train_loader(args))
    else:
        train_loaders.append(None)

    # train_loader_1
    if 1 in op_params['setting_list']:
        args.num_segments = op_params['num_segments_list'][1]
        args.clip_length = op_params['clip_length_list'][1]
        args.num_steps = op_params['num_steps_list'][1]
        args.batch_size = op_params['batch_size_list'][1]
        train_loaders.append(get_train_loader(args))
    else:
        train_loaders.append(None)

    # train_loader_2
    if 2 in op_params['setting_list']:
        args.num_segments = op_params['num_segments_list'][2]
        args.clip_length = op_params['clip_length_list'][2]
        args.num_steps = op_params['num_steps_list'][2]
        args.batch_size = op_params['batch_size_list'][2]
        train_loaders.append(get_train_loader(args))
    else:
        train_loaders.append(None)

    return train_loaders


def save_checkpoint(args, epoch, model, optimizer, name):
    logger.info('==> Saving...')
    state = {
        'opt': args.C,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.output_dir, name + '.pth'))


def func(x, a, b, c, d, e):
    return - a * np.exp(- b * x) - c * x - d * x * x + e


def main(args):
    train_loader_list = get_op_train_loaders(args)

    n_data = len(train_loader_list[0].dataset)
    logger.info("length of training dataset: {}".format(n_data))

    if args.label_smooth:
        criterion = LSR(e=0.1).cuda()
    elif args.mixup_prob > 0:
        criterion = SoftTargetCrossEntropy().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    global_epoch = 0
    history_father_node = [0] * (max(op_params['end_node_list']) + 1)
    history_node_epochs = [0] * (max(op_params['end_node_list']) + 1)
    history_edge_epochs = [0] * len(op_params['start_node_list'])
    for edge_id in range(len(op_params['end_node_list'])):
        start_node = op_params['start_node_list'][edge_id]
        end_node = op_params['end_node_list'][edge_id]

        setting_id = op_params['setting_list'][end_node]
        lr_mult = op_params['lr_mult_list'][end_node]

        args.frozen_bn = op_params['frozen_bn_list'][setting_id]
        args.iter_size = op_params['iter_size_list'][setting_id]
        args.base_learning_rate = op_params['base_lr_list'][setting_id] * lr_mult

        if start_node == 0:
            args.pretrained_model = args.C['network']['pretrained_model']
            args.remove_fc = args.C['network']['remove_fc']
            args.transfer_weights = args.C['network']['transfer_weights']
        else:
            args.pretrained_model = os.path.join(args.output_dir, 'node_' + str(start_node) + '.pth')
            args.remove_fc = False
            args.transfer_weights = False

        origin_model = args.pretrained_model
        
        logger.info('edge_id: ' + str(edge_id) + ', [' + str(start_node) + ', ' + str(end_node) + ']')

        # load_pretrained(args, model)
        model = build_model(args)

        if args.fc_higher_lr:
            if hasattr(model, 'fc_dual'):
                fc_parameters_iter = list(itertools.chain(model.fc.parameters(), model.fc_dual.parameters()))
            else:
                fc_parameters_iter = list(model.fc.parameters())
            fc_parameters = list(map(id, fc_parameters_iter))
            other_parameters = (p for p in model.parameters() if (id(p) not in fc_parameters) and p.requires_grad)
            parameters = [
                {'params': other_parameters},
                {'params': fc_parameters_iter, 'lr': args.base_learning_rate * 10}
            ]
        else:
            parameters = filter(lambda p: p.requires_grad, model.parameters())

        if args.adamw_solver:
            optimizer = torch.optim.AdamW(parameters,
                                          lr=args.base_learning_rate,
                                          weight_decay=args.weight_decay,
                                          )
        else:
            optimizer = torch.optim.SGD(parameters,
                                        lr=args.base_learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=args.nesterov)

        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=True,
                                    find_unused_parameters=True)

        acc_list = []
        if start_node == 0:
            acc_list.append(0.)
        else:
            acc_list.append(eva_inter(args, None, model))

        local_epoch = 0
        while True:
            train_loader = train_loader_list[setting_id]

            global_epoch += 1
            local_epoch += 1
            train_loader.sampler.set_epoch(global_epoch)
            tic = time.time()

            loss = train(global_epoch, train_loader, model, criterion, optimizer, None, args)
            logger.info('epoch {}, total time {:.2f}'.format(global_epoch, time.time() - tic))
            if dist.get_rank() == 0:
                save_checkpoint(args, local_epoch, model, optimizer, 'edge_' + str(edge_id) + '_' + str(local_epoch))
            dist.barrier()

            if summary_writer is not None:
                # tensorboard logger
                summary_writer.add_scalar('ins_loss', loss, global_epoch)
                summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_epoch)

            acc_list.append(eva_inter(args, None, model))

            logger.info('--epoch: ' + str(local_epoch))
            logger.info('--' + str(acc_list))

            if local_epoch >= op_params['attempt_epochs']:
                try:
                    input_x = np.array(range(len(acc_list)))
                    input_y = np.array(acc_list)
                    popt, pcov = curve_fit(func, input_x, input_y,
                                           bounds=([0, 0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]))

                    # plot results
                    if op_params['visualize_figure']:
                        yvals = func(input_x, popt[0], popt[1], popt[2], popt[3], popt[4])
                        plot1 = plt.plot(input_x, input_y, 's', label='original values')
                        plot2 = plt.plot(input_x, yvals, 'r', label='polyfit values')
                        plt.xlabel('Number of Epoches')
                        plt.ylabel('Clip Accuracy')
                        plt.legend(loc=4)
                        plt.title('curve_fit')
                        plt.savefig(os.path.join(args.output_dir, 'edge_' + str(edge_id) + '.pdf'))
                        plt.cla()

                    yvals = func(np.array(range(1000)), popt[0], popt[1], popt[2], popt[3], popt[4])
                    yvals_list = yvals.tolist()
                    y_max = max(yvals_list)
                    x_max = yvals_list.index(max(yvals_list))

                    logger.info('--best epoch: ' + str(x_max) + ', acc: ' + str(y_max))
                    if local_epoch >= x_max + op_params['attempt_epochs']:
                        logger.info('--select epoch: ' + str(x_max) + ', acc: ' + str(y_max))
                        
                        if dist.get_rank() == 0:
                            if x_max == 0:
                                shutil.copyfile(origin_model, os.path.join(args.output_dir, 'edge_' + str(edge_id) + '.pth'))
                                history_edge_epochs[edge_id] = 0
                            else:
                                shutil.copyfile(os.path.join(args.output_dir, 'edge_' + str(edge_id) + '_' + str(x_max) + '.pth'), os.path.join(args.output_dir, 'edge_' + str(edge_id) + '.pth'))
                                history_edge_epochs[edge_id] = x_max

                            for model_id in range(1, len(acc_list)):
                                os.remove(os.path.join(args.output_dir, 'edge_' + str(edge_id) + '_' + str(model_id) + '.pth'))

                            with open(os.path.join(args.output_dir, 'edge_' + str(edge_id) + '.txt'), 'w') as f:
                                f.write(str(y_max))
                        dist.barrier()
                        break
                except RuntimeError:
                    logger.info('--cannot fit curve.')

        if dist.get_rank() == 0:
            flag = True
            max_node_score = y_max
            max_edge_id = edge_id
            for i in range(len(op_params['end_node_list'])):
                if op_params['end_node_list'][i] == end_node:
                    if not os.path.exists(os.path.join(args.output_dir, 'edge_' + str(i) + '.txt')):
                        flag = False
                        break
                    else:
                        with open(os.path.join(args.output_dir, 'edge_' + str(i) + '.txt'), 'r') as f:
                            score = float(f.readline())
                        if score > max_node_score:
                            max_node_score = score
                            max_edge_id = i
            if flag:
                shutil.copyfile(os.path.join(args.output_dir, 'edge_' + str(max_edge_id) + '.pth'),
                                os.path.join(args.output_dir, 'node_' + str(end_node) + '.pth'))
                history_father_node[end_node] = op_params['start_node_list'][max_edge_id]
                history_node_epochs[end_node] = history_edge_epochs[max_edge_id]
                with open(os.path.join(args.output_dir, 'node_' + str(end_node) + '.txt'), 'w') as f:
                    f.write(str(max_node_score))
        dist.barrier()
    if dist.get_rank() == 0:
        shutil.copyfile(os.path.join(args.output_dir, 'node_' + str(end_node) + '.pth'),
                        os.path.join(args.output_dir, 'current.pth'))
        best_node = []
        best_epochs = []
        node_id = max(op_params['end_node_list'])
        while node_id != 0:
            best_node.append(node_id)
            best_epochs.append(history_node_epochs[node_id])
            node_id = history_father_node[node_id]
        best_node = list(reversed(best_node))
        best_epochs = list(reversed(best_epochs))

        num_segments = []
        clip_length = []
        num_steps = []
        batch_size = []
        iter_size = []
        frozen_bn = []
        base_learning_rate = []
        epochs = []
        for idx in range(len(best_node)):
            if best_epochs[idx] == 0:
                continue
            setting_id = op_params['setting_list'][best_node[idx]]
            num_segments.append(op_params['num_segments_list'][setting_id])
            clip_length.append(op_params['clip_length_list'][setting_id])
            num_steps.append(op_params['num_steps_list'][setting_id])
            batch_size.append(op_params['batch_size_list'][setting_id])
            iter_size.append(op_params['iter_size_list'][setting_id])
            frozen_bn.append(op_params['frozen_bn_list'][setting_id])
            base_learning_rate.append(op_params['base_lr_list'][setting_id] * op_params['lr_mult_list'][best_node[idx]])
            epochs.append(best_epochs[idx])
        logger.info('num_segments: ' + str(num_segments))
        logger.info('clip_length: ' + str(clip_length))
        logger.info('num_steps: ' + str(num_steps))
        logger.info('batch_size: ' + str(batch_size))
        logger.info('iter_size: ' + str(iter_size))
        logger.info('frozen_bn: ' + str(frozen_bn))
        logger.info('base_learning_rate: ' + str(base_learning_rate))
        logger.info('epochs: ' + str(epochs))


if __name__ == '__main__':
    opt = parse_option()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="p3d")
    opt.logger = logger
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "train_3d.config.yml")
        with open(path, 'w') as f:
            yaml.dump(opt.C, f, default_flow_style=False)
        logger.info("Full config saved to {}".format(path))

    main(opt)
