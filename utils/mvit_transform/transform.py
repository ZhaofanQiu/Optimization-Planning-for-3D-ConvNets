# Kinetics default transform in MViT
# from https://github.com/facebookresearch/SlowFast/
# https://arxiv.org/pdf/2104.11227.pdf
import torch
from torchvision import transforms

from utils import clip_transforms
from .rand_augment import rand_augment_transform
from .random_erasing import RandomErasing
from PIL import Image


def mvit_transform(args):
    auto_augment = "rand-m7-n4-mstd0.5-inc1"

    img_size_min = args.crop_size
    aa_params = {"translate_const": int(img_size_min * 0.45)}
    aa_params["interpolation"] = Image.BICUBIC

    aug_transform = transforms.Compose([
        clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.08, 1.), ratio=(0.75, 1.3333333333333333)),
        rand_augment_transform(auto_augment, aa_params),
        clip_transforms.ClipRandomHorizontalFlip(p=0.0 if args.no_horizontal_flip else 0.5),
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda clip: torch.stack(clip, dim=0)),  # T, C, H, W
        RandomErasing(0.25, mode="pixel", max_count=1, num_splits=False, device="cpu"),
        transforms.Lambda(lambda clip: torch.transpose(clip, 0, 1))  # C, T, H, W
        ])

    return aug_transform