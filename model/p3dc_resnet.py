"""
P3D-C ResNet
A PyTorch implement of P3D-C ResNet as described in
'Learning spatio-temporal representation with pseudo-3d residual networks' - ICCV 2017

By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from .model_factory import register_model

__all__ = ['P3DC_ResNet', 'p3dc_resnet18', 'p3dc_resnet34', 'p3dc_resnet50', 'p3dc_resnet101', 'p3dc_resnet103',
           'p3dc_resnet152', 'p3dc_resnext50_32x4d', 'p3dc_resnext101_32x8d',
           'p3dc_wide_resnet50_2', 'p3dc_wide_resnet101_2']


def conv1x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[1, 3, 3], stride=[1, stride, stride],
                     padding=[0, dilation, dilation], groups=groups, bias=False, dilation=[1, dilation, dilation])


def conv3x1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[3, 1, 1], stride=[1, stride, stride],
                     padding=[dilation, 0, 0], groups=groups, bias=False, dilation=[dilation, 1, 1])


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=[1, stride, stride], bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.conv1_t = conv3x1x1(planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.conv2_t = conv3x1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out_t = self.conv1_t(out)
        out = self.relu(out + out_t)

        out = self.conv2(out)
        out = self.bn2(out)
        out_t = self.conv2_t(out)
        out = out + out_t

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3x3(width, width, stride, groups, dilation)
        self.conv2_t = conv3x1x1(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out_t = self.conv2_t(out)
        out = self.relu(out + out_t)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class P3DC_ResNet(nn.Module):

    def __init__(self, block, layers, pooling_arch, early_stride=4, num_classes=400, dropout_ratio=0.5, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, deep_stem=False):
        super(P3DC_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d if not deep_stem else partial(nn.BatchNorm3d, eps=2e-5)
        self._norm_layer = norm_layer
        self._deep_stem = deep_stem

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if not deep_stem:
            self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=[early_stride, 7, 7], stride=[early_stride, 2, 2], padding=[0, 3, 3],
                                    bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=[early_stride, 3, 3], stride=[early_stride, 2, 2], padding=[0, 1, 1],
                                    bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
                                    bias=False)
            self.bn2 = norm_layer(self.inplanes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv3d(self.inplanes, self.inplanes * 2, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
                                    bias=False)
            self.bn3 = norm_layer(self.inplanes * 2)
            self.relu3 = nn.ReLU(inplace=True)
            
            self.inplanes *= 2
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.pool = pooling_arch(input_dim=512 * block.expansion)

        self.drop = nn.Dropout(dropout_ratio)

        self.fc = nn.Linear(self.pool.output_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, layer):
        # See note [TorchScript super()]
        if not self._deep_stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if layer == 5:
            return x
        x_g = self.pool(x)
        if layer == 6:
            return x_g

        x_g = self.drop(x_g)

        x1 = self.fc(x_g)
        return x1

    def forward(self, x, layer=7):
        return self._forward_impl(x, layer)


def transfer_weights(state_dict, early_stride):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if ('conv' in k) or ('downsample.0' in k):
            shape = v.shape
            v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
            if (shape[2] == 3) and (shape[3] == 3) and ('layer' in k):  # basic conv3x3 layer
                kernel = np.zeros(shape=(shape[0], shape[0], 3, 1, 1))
                ss = k.split('.')
                new_key = k[:-len(ss[-1]) - 1] + '_t.' + ss[-1]
                new_state_dict[new_key] = torch.from_numpy(kernel)
            if (not ('layer' in k)) and ('conv1' in k):  # first conv7x7 layer
                if early_stride != 1:
                    s1 = early_stride // 2
                    s2 = early_stride - early_stride // 2 - 1
                    v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])), v, np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
        new_state_dict[k] = torch.from_numpy(v)
    return new_state_dict


def _p3dc_resnet(arch, block, layers, pooling_arch, image_size=None, **kwargs):
    model = P3DC_ResNet(block, layers, pooling_arch, **kwargs)
    return model


@register_model
def p3dc_resnet18(pooling_arch, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    return _p3dc_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pooling_arch,
                   **kwargs)


@register_model
def p3dc_resnet34(pooling_arch, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    return _p3dc_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pooling_arch,
                   **kwargs)


@register_model
def p3dc_resnet50(pooling_arch, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    return _p3dc_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pooling_arch,
                   **kwargs)


@register_model
def p3dc_resnet101(pooling_arch, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    return _p3dc_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pooling_arch,
                   **kwargs)


@register_model
def p3dc_resnet103(pooling_arch, **kwargs):
    r"""ResNet-103 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    return _p3dc_resnet('resnet103', Bottleneck, [3, 4, 23, 3], pooling_arch, deep_stem=True,
                   **kwargs)


@register_model
def p3dc_resnet152(pooling_arch, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    return _p3dc_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pooling_arch,
                   **kwargs)


@register_model
def p3dc_resnext50_32x4d(pooling_arch, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _p3dc_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pooling_arch, **kwargs)


@register_model
def p3dc_resnext101_32x8d(pooling_arch, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _p3dc_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pooling_arch, **kwargs)


@register_model
def p3dc_wide_resnet50_2(pooling_arch, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    kwargs['width_per_group'] = 64 * 2
    return _p3dc_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pooling_arch, **kwargs)


@register_model
def p3dc_wide_resnet101_2(pooling_arch, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    kwargs['width_per_group'] = 64 * 2
    return _p3dc_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pooling_arch, **kwargs)
