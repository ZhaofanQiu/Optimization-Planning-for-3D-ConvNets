"""
Train 3d architecture
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

from utils.util import AverageMeter, reduce_tensor, merge_config, add_config
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from layer.LSR import *

from torch.cuda.amp import GradScaler
from utils.mvit_transform.mixup import MixUp
from utils.metrics import accuracy
import itertools

from helpers import get_train_loader, get_eva_loader, build_model


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
    return args


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {
        'opt': args.C,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


def load_checkpoint(args, model, optimizer, scheduler):
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    args.logger.info('Missing keys: {}'.format(misskeys))
    args.logger.info('Unexpect keys: {}'.format(unexpkeys))
    args.logger.info("==> loaded checkpoint '{}'".format(args.checkpoint))

    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])

    return ckpt['epoch']


def eva_inter(args, test_loader, model):
    model.eval()

    if test_loader is None:
        # single-center-crop fast evaluation
        args.crop_idx = 0
        args.eva_num_clips = 1
        args.eva_crop_size = args.crop_size
        test_loader = get_eva_loader(args)

    n_data = len(test_loader.dataset)
    args.logger.info("length of testing dataset: {}".format(n_data))

    # routine
    all_scores = np.zeros([len(test_loader) * args.batch_size, args.num_classes], dtype=np.float)
    all_labels = np.zeros([len(test_loader) * args.batch_size], dtype=np.float)
    top_idx = 0
    with torch.no_grad():
        for idx, (x, label) in enumerate(test_loader):
            if (idx % 100 == 0) or (idx == len(test_loader) - 1):
                args.logger.info('{}/{}'.format(idx, len(test_loader)))
            bsz = x.size(0)
            score = model(x.cuda())
            if isinstance(score, list):
                if len(score) == 2:
                    score_numpy = (score[0].data.cpu().numpy() + score[1].data.cpu().numpy()) / 2
                else:
                    score_numpy = sum([score_i.data.cpu().numpy() for score_i in score]) / len(score)
            else:
                score_numpy = score.data.cpu().numpy()
            label_numpy = label.data.cpu().numpy()
            all_scores[top_idx: top_idx + bsz, :] = score_numpy
            all_labels[top_idx: top_idx + bsz] = label_numpy
            top_idx += bsz

        all_scores = all_scores[:top_idx, :]
        all_labels = all_labels[:top_idx]
        eva_accuracy = accuracy(all_scores, all_labels, topk=(1, 3, 5))
        top1_accuracy = eva_accuracy[0].cuda()
        top3_accuracy = eva_accuracy[1].cuda()
        top5_accuracy = eva_accuracy[2].cuda()
        top1_accuracy = reduce_tensor(top1_accuracy.cuda())
        top3_accuracy = reduce_tensor(top3_accuracy.cuda())
        top5_accuracy = reduce_tensor(top5_accuracy.cuda())
        t1 = top1_accuracy.data.cpu().item()
        t3 = top3_accuracy.data.cpu().item()
        t5 = top5_accuracy.data.cpu().item()
        if dist.get_rank() == 0:
            args.logger.info('eva accuracy top1: {:.4f}, top3: {:.4f}, top5: {:.4f}, ave: {:.4f}'.format(t1, t3, t5, (t1 + t3 + t5) / 3))
        return (t1 + t3 + t5) / 3


def main_multi_step(args):
    list_batch_size = args.batch_size
    list_iter_size = args.iter_size
    list_base_learning_rate = args.base_learning_rate
    list_epochs = args.epochs
    list_num_segments = args.num_segments
    list_clip_length = args.clip_length
    list_num_steps = args.num_steps
    list_frozen_bn = args.frozen_bn

    assert len(list_batch_size) == len(list_iter_size) == len(list_base_learning_rate) == len(list_epochs) == len(list_num_segments) == len(list_clip_length) == len(list_num_steps) == len(list_frozen_bn)

    global_epochs = 1
    for step_id in range(len(list_epochs)):
        args.batch_size = list_batch_size[step_id]
        args.iter_size = list_iter_size[step_id]
        args.base_learning_rate = list_base_learning_rate[step_id]
        args.epochs = list_epochs[step_id]
        args.num_segments = list_num_segments[step_id]
        args.clip_length = list_clip_length[step_id]
        args.num_steps = list_num_steps[step_id]
        args.frozen_bn = list_frozen_bn[step_id]

        train_loader = get_train_loader(args)
        n_data = len(train_loader.dataset)
        logger.info("length of training dataset: {}".format(n_data))

        model = build_model(args)

        # print network architecture
        if dist.get_rank() == 0:
            # try:
            #     from torchsummary import summary
            #     summary(model, (3, args.num_segments * args.clip_length, args.crop_size, args.crop_size))
            # except ImportError:
            logger.info(model)

        if args.label_smooth:
            criterion = LSR(e=0.1).cuda()
        elif args.use_mixup:
            criterion = SoftTargetCrossEntropy().cuda()
        else:
            criterion = torch.nn.CrossEntropyLoss().cuda()

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

        scheduler = get_scheduler(optimizer, len(train_loader), args)

        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=True,
                                        find_unused_parameters=True)

        # tensorboard
        if dist.get_rank() == 0:
            summary_writer = SummaryWriter(log_dir=args.output_dir)
        else:
            summary_writer = None

        # routine
        for epoch in range(global_epochs, global_epochs + args.epochs):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()
            loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, args)
            logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))
            if summary_writer is not None:
                # tensorboard logger
                summary_writer.add_scalar('ins_loss', loss, epoch)
                summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            if dist.get_rank() == 0:
                # save model
                save_checkpoint(args, epoch, model, optimizer, scheduler)
            if args.eva_inter and (epoch % args.eva_inter_freq == 0):
                eva_inter(args, None, model)

        global_epochs += args.epochs
        if step_id == 0:
            args.pretrained_model = os.path.join(args.output_dir, 'current.pth')
            args.transfer_weights = False
            args.remove_fc = False


def main(args):
    if isinstance(args.epochs, list):
        main_multi_step(args)
        return

    train_loader = get_train_loader(args)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))

    model = build_model(args)

    # print network architecture
    if dist.get_rank() == 0:
        # try:
        #     from torchsummary import summary
        #     summary(model, (3, args.num_segments * args.clip_length, args.crop_size, args.crop_size))
        # except ImportError:
        logger.info(model)

    if args.label_smooth:
        criterion = LSR(e=0.1).cuda()
    elif args.use_mixup:
        criterion = SoftTargetCrossEntropy().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

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
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    start_epoch = 0
    if hasattr(args, 'checkpoint'):
        start_epoch = load_checkpoint(args, model, optimizer, scheduler)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=True,
                                    find_unused_parameters=True)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, args)
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))
        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, optimizer, scheduler)
        if args.eva_inter and (epoch % args.eva_inter_freq == 0):
            eva_inter(args, None, model)


def train(epoch, train_loader, model, criterion, optimizer, scheduler, args):
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    end = time.time()

    optimizer.zero_grad()
    scaler = GradScaler()
    bnorm = 0

    if args.use_mixup:
        mixup_fn = MixUp(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=1.0, switch_prob=0.5,
            label_smoothing=0.1, num_classes=args.num_classes)

    for idx, train_data in enumerate(train_loader):
        x = train_data[0]
        label = train_data[1]

        bsz = x.size(0)

        # forward
        x = x.cuda(non_blocking=True)  # clip
        label = label.cuda(non_blocking=True)  # label
        if args.use_mixup:
            x, label = mixup_fn(x, label)

        # with torch.cuda.amp.autocast():
        # forward and get the predict score
        score = model(x)
        # get crossentropy loss
        if isinstance(score, list):
            if len(score) == 2:
                loss = criterion(score[0], label) + criterion(score[1], label)
            else:
                loss = sum([criterion(score_i, label) for score_i in score])
        else:
            loss = criterion(score, label)

        # backward
        scaler.scale(loss / args.iter_size * args.loss_weight).backward()

        if (idx + 1) % args.iter_size == 0:
            scaler.unscale_(optimizer)
            bnorm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   args.clip_gradient)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        norm_meter.update(bnorm, bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            args.logger.info(
                'Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} LR={:>0.5f} Loss={:>0.3f}/{:>0.3f} GradNorm={:>0.3f}/{:>0.3f}'.format(
                    epoch, idx, len(train_loader),
                    batch_time.val, batch_time.avg,
                    next(iter(optimizer.param_groups))['lr'],
                    loss.item(), loss_meter.avg,
                    bnorm, norm_meter.avg
                ))

    return loss_meter.avg


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
