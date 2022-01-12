"""
Training/evaluation helpers
By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import torch

from torchvision import transforms
from utils import clip_transforms
from utils.clip_rand_augment import ClipRandAugment

from dataset.video_dataset import VideoRGBTrainDataset, VideoFlowTrainDataset
from dataset.video_dataset import VideoRGBTestDataset, VideoFlowTestDataset

import model as model_factory
from layer.pooling_factory import get_pooling_by_name
from layer.frozen_bn import FrozenBatchNorm
from utils.mvit_transform.transform import mvit_transform


transform_type_list = ['default',
                       'rand_augment',
                       'mvit_default']


def get_train_loader(args, dist=True):
    if not args.use_flow:
        if args.transform_type == 'default':
            train_transform = transforms.Compose([
                clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.2, 1.),
                                                      ratio=(0.75, 1.3333333333333333)),
                transforms.RandomApply([
                    clip_transforms.ClipColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                clip_transforms.ClipRandomGrayscale(p=0.2),
                transforms.RandomApply([clip_transforms.ClipGaussianBlur([.1, 2.])], p=0.5),
                clip_transforms.ClipRandomHorizontalFlip(p=0.0 if args.no_horizontal_flip else 0.5),
                clip_transforms.ToClipTensor(),
                clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
                    lambda clip: torch.cat(clip, dim=0))
            ])
        elif args.transform_type == 'rand_augment':
            train_transform = transforms.Compose([
                clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.2, 1.),
                                                      ratio=(0.75, 1.3333333333333333)),
                ClipRandAugment(n=args.ra_n, m=args.ra_m),  # N = [1, 2, 3], M = [5, 7, 9, 11, 13, 15]
                clip_transforms.ClipRandomHorizontalFlip(p=0.0 if args.no_horizontal_flip else 0.5),
                clip_transforms.ToClipTensor(),
                clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
                    lambda clip: torch.cat(clip, dim=0))
            ])
        elif args.transform_type == 'mvit_default':
            train_transform = mvit_transform(args)
        else:
            raise NotImplementedError
        if args.use_fore:
            if args.transform_type in ('default', 'rand_augment'):
                train_transform.transforms.insert(len(train_transform.transforms) - 1, clip_transforms.ClipForeground())
            elif args.transform_type == 'mvit_default':
                train_transform.transforms.insert(len(train_transform.transforms) - 3, clip_transforms.ClipForeground())
            else:
                raise NotImplementedError
    else:
        if args.transform_type == 'default':
            train_transform = transforms.Compose([
                clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.5, 1.),
                                                      ratio=(0.75, 1.3333333333333333)),
                clip_transforms.FlowClipRandomHorizontalFlip(p=0.0 if args.no_horizontal_flip else 0.5),
                clip_transforms.FlowToClipTensor(),
                clip_transforms.ClipNormalize(mean=[0.5, 0.5], std=[0.229, 0.229]),
                transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
                    lambda clip: torch.cat(clip, dim=0))
            ])
        else:
            raise NotImplementedError
        assert not args.use_fore

    # flow only support video_dataset
    if args.use_flow:
        assert args.dataset_class == 'video_dataset'

    if args.dataset_class == 'video_dataset':
        assert (args.list_file is not None and args.root_path is not None)
        if not args.use_flow:
            train_dataset = VideoRGBTrainDataset(list_file=args.list_file, root_path=args.root_path,
                                                 transform=train_transform, clip_length=args.clip_length,
                                                 num_steps=args.num_steps, num_segments=args.num_segments,
                                                 format=args.format)
        else:
            train_dataset = VideoFlowTrainDataset(list_file=args.list_file, root_path=args.root_path,
                                                  transform=train_transform, clip_length=args.clip_length,
                                                  num_steps=args.num_steps, num_segments=args.num_segments,
                                                  format=args.format)
    else:
        raise NotImplementedError

    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return train_loader


def get_eva_loader(args, dist=True):
    if args.crop_idx == 0:
        crop = clip_transforms.ClipCenterCrop
    elif args.crop_idx == 1:
        crop = clip_transforms.ClipFirstCrop
    elif args.crop_idx == 2:
        crop = clip_transforms.ClipThirdCrop
    else:
        raise NotImplementedError

    if not args.use_flow:
        test_transform = transforms.Compose([
            clip_transforms.ClipResize(size=args.eva_crop_size),
            crop(size=args.eva_crop_size),
            clip_transforms.ToClipTensor(),
            clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
                lambda clip: torch.cat(clip, dim=0))
        ])

        if args.use_fore:
            test_transform.transforms.insert(len(test_transform.transforms) - 1, clip_transforms.ClipForeground())
    else:
        test_transform = transforms.Compose([
            clip_transforms.ClipResize(size=args.eva_crop_size),
            crop(size=args.eva_crop_size),
            clip_transforms.FlowToClipTensor(),
            clip_transforms.ClipNormalize(mean=[0.5, 0.5], std=[0.229, 0.229]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
                lambda clip: torch.cat(clip, dim=0))
        ])

        assert not args.use_fore

    if not args.use_flow:
        test_dataset = VideoRGBTestDataset(args.eva_list_file, num_clips=args.eva_num_clips,
                                           transform=test_transform, root_path=args.eva_root_path,
                                           clip_length=args.clip_length, num_steps=args.num_steps,
                                           num_segments=args.eva_num_segments,
                                           format=args.format)
    else:
        test_dataset = VideoFlowTestDataset(args.eva_list_file, num_clips=args.eva_num_clips,
                                            transform=test_transform, root_path=args.eva_root_path,
                                            clip_length=args.clip_length, num_steps=args.num_steps,
                                            num_segments=args.eva_num_segments,
                                            format=args.format)

    if dist:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = torch.utils.data.Sampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=test_sampler, drop_last=False)
    return test_loader


def build_model(args):
    model = model_factory.get_model_by_name(net_name=args.net_name, pooling_arch=get_pooling_by_name(args.pooling_name),
                                            num_classes=args.num_classes, dropout_ratio=args.dropout_ratio,
                                            image_size=args.crop_size, early_stride=args.early_stride).cuda()
    if args.pretrained_model:
        load_pretrained(args, model)

    if args.frozen_bn:
        frozen_bn(model)
    return model


def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    if 'model' in ckpt:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    else:
        state_dict = ckpt

    # convert initial weights
    if args.transfer_weights:
        state_dict = model_factory.transfer_weights(args.net_name, state_dict, args.early_stride)
    if args.remove_fc:
        state_dict = model_factory.remove_fc(args.net_name, state_dict)

    if 'vit' in args.net_name:
        from model.vit import checkpoint_filter_fn
        state_dict = checkpoint_filter_fn(state_dict, model)

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    args.logger.info('Missing keys: {}'.format(misskeys))
    args.logger.info('Unexpect keys: {}'.format(unexpkeys))
    args.logger.info("==> loaded checkpoint '{}'".format(args.pretrained_model))


def frozen_bn(module, full_name='root'):

    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = FrozenBatchNorm(module.num_features,
                                        module.eps, module.momentum,
                                        module.affine,
                                        module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    skip_first_bn = (full_name == 'root')
    for name, child in module.named_children():
        if skip_first_bn and isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            skip_first_bn = False
            if torch.distributed.get_rank() == 0:
                print('skip frozen bn: ' + full_name + '.' + name)
            continue
        module_output.add_module(name, frozen_bn(child, full_name + '.' + name))
    del module
    return module_output

