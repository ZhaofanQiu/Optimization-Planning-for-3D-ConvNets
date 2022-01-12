"""
C2D-VIT: only change patch-embedding to tubelet-embedding
This implementation is based on https://github.com/rwightman/pytorch-image-models
Modified by Zhaofan Qiu
zhaofanqiu@gmail.com
"""

import torch
import torch.nn as nn
import numpy as np

from utils.vit_helpers import to_2tuple

from .vit import VisionTransformer
from .model_factory import register_model


class TubeletEmbed(nn.Module):
    """ Video to Tubelet Embedding
    """
    def __init__(self, img_size=224, early_stride=4, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # fixed with time-length=time-stride=4
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.early_stride = early_stride

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(early_stride,) + patch_size, stride=(early_stride,) + patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert T == self.early_stride and H == self.img_size[0] and W == self.img_size[1],  \
            f"Input image size ({T}*{H}*{W}) doesn't match model ({self.early_stride}*{self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class C2D_ViT(VisionTransformer):
    def __init__(self, img_size=224, early_stride=4, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, dropout_ratio=0.):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, representation_size,
                 drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, dropout_ratio)

        self.early_stride = early_stride
        self.patch_embed = TubeletEmbed(img_size=img_size, early_stride=early_stride, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        bsz = x.size(0)
        chn = x.size(1)
        lgt = x.size(2)
        hig = x.size(3)
        wid = x.size(4)
        x = x.view(bsz, chn, lgt // self.early_stride, self.early_stride, hig, wid).transpose(1, 2)
        x = x.reshape(bsz * lgt // self.early_stride, chn, self.early_stride, hig, wid)

        x = self.forward_features(x)
        x = self.drop(x)
        x = self.fc(x)
        return x.view(bsz, lgt // self.early_stride, -1).mean(dim=1)

        
def transfer_weights(state_dict, early_stride):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if k == 'patch_embed.proj.weight':
            shape = v.shape
            v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
            if early_stride != 1:
                s1 = early_stride // 2
                s2 = early_stride - early_stride // 2 - 1
                v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])), v, np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
        new_state_dict[k] = torch.from_numpy(v)
    return new_state_dict


@register_model
def c2d_vit_t_p16(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = C2D_ViT(img_size=image_size, early_stride=early_stride, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=192, depth=12,
                      num_heads=3, dropout_ratio=dropout_ratio)
    return model


@register_model
def c2d_vit_s_p16(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = C2D_ViT(img_size=image_size, early_stride=early_stride, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=12,
                      num_heads=6, dropout_ratio=dropout_ratio)

    return model


@register_model
def c2d_vit_b_p16(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = C2D_ViT(img_size=image_size, early_stride=early_stride, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=768, depth=12,
                      num_heads=12, dropout_ratio=dropout_ratio)
    return model


@register_model
def c2d_vit_b_p32(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model = C2D_ViT(img_size=image_size, early_stride=early_stride, patch_size=32, in_chans=3, num_classes=num_classes, embed_dim=768, depth=12,
                      num_heads=12, dropout_ratio=dropout_ratio)
    return model


@register_model
def c2d_vit_l_p16(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = C2D_ViT(img_size=image_size, early_stride=early_stride, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=1024, depth=24,
                      num_heads=16, dropout_ratio=dropout_ratio)
    return model


@register_model
def c2d_vit_l_p32(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = C2D_ViT(img_size=image_size, early_stride=early_stride, patch_size=32, in_chans=3, num_classes=num_classes, embed_dim=1024, depth=24,
                      num_heads=16, dropout_ratio=dropout_ratio)
    return model
