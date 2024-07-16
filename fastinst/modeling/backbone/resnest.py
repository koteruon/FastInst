# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
from detectron2.layers import NaiveSyncBatchNorm, DeformConv
from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, GroupNorm
from timm.models.resnet import BasicBlock, Bottleneck

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectiveKernel, ConvNormAct, create_attn, SplitAttn
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model, generate_default_cfgs
from .resnet import ResNet


######## CBAM
# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

########

######## BilinearAttnTransform
# https://github.com/huggingface/pytorch-image-models/blob/c28ee2e904b75ac59958192ff941b73e2a7fce31/timm/layers/non_local_attn.py#L24

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class BilinearAttnTransform(nn.Module):

    def __init__(self, in_channels, block_size, groups, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BilinearAttnTransform, self).__init__()

        self.conv1 = ConvNormAct(in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_p = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(block_size, 1))
        self.conv_q = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(1, block_size))
        self.conv2 = ConvNormAct(in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        B, C, block_size, block_size1 = x.shape
        _assert(block_size == block_size1, '')
        if t <= 1:
            return x
        x = x.view(B * C, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(B * C, block_size, block_size, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(B, C, block_size * t, block_size * t)
        return x

    def forward(self, x):
        _assert(x.shape[-1] % self.block_size == 0, '')
        _assert(x.shape[-2] % self.block_size == 0, '')
        B, C, H, W = x.shape
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.block_size, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.block_size))
        p = self.conv_p(rp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        q = self.conv_q(cp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        p = p.view(B, C, self.block_size, self.block_size)
        q = q.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        q = q.view(B, C, self.block_size, self.block_size)
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y
    
########

class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            radix=1,
            cardinality=1,
            base_width=64,
            avd=False,
            avd_first=False,
            is_first=False,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            # act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            # norm_layer= nn.GroupNorm,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_layer=drop_block)
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

        ### Updated

        self.ca = ChannelAttention(planes * 4)
        # self.sa = SpatialAttention()
        
        ###
        # rd_ratio, rd_divisor, block_size, groups = 0.25, 8, 7, 2
        # if rd_channels is None:
        #     rd_channels = make_divisible(inplanes * rd_ratio, divisor=rd_divisor)
        
        # self.ba = BilinearAttnTransform(rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer)

        ###

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)

        ### BATransform
        # out = self.ba(out)
        ###

        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        ### CBAM
        out = self.ca(out) * out
        # out = self.sa(out) * out
        ###

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.act3(out)
        return out



# @register_model
# def resnest14d(pretrained=False, **kwargs) -> ResNet:
#     """ ResNeSt-14d model. Weights ported from GluonCV.
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[1, 1, 1, 1],
#         stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False))
#     return _create_resnest('resnest14d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


# @register_model
# def resnest26d(pretrained=False, **kwargs) -> ResNet:
#     """ ResNeSt-26d model. Weights ported from GluonCV.
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[2, 2, 2, 2],
#         stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False))
#     return _create_resnest('resnest26d', pretrained=pretrained, **dict(model_kwargs, **kwargs))

# @register_model
# def resnest50d(pretrained=False, **kwargs) -> ResNet:
#     """ ResNeSt-50d model. Matches paper ResNeSt-50 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'd' for deep stem, stem_width 32, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[3, 4, 6, 3],
#         stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False))
#     return _create_resnest('resnest50d', pretrained=pretrained, **dict(model_kwargs, **kwargs))

# @register_model
# def resnest101e(pretrained=False, **kwargs) -> ResNet:
#     """ ResNeSt-101e model. Matches paper ResNeSt-101 model, https://arxiv.org/abs/2004.08955
#      Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[3, 4, 23, 3],
#         stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False))
#     return _create_resnest('resnest101e', pretrained=pretrained, **dict(model_kwargs, **kwargs))

# @register_model
# def resnest200e(pretrained=False, **kwargs) -> ResNet:
#     """ ResNeSt-200e model. Matches paper ResNeSt-200 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[3, 24, 36, 3],
#         stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False))
#     return _create_resnest('resnest200e', pretrained=pretrained, **dict(model_kwargs, **kwargs))

# @register_model
# def resnest269e(pretrained=False, **kwargs) -> ResNet:
#     """ ResNeSt-269e model. Matches paper ResNeSt-269 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[3, 30, 48, 8],
#         stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False))
#     return _create_resnest('resnest269e', pretrained=pretrained, **dict(model_kwargs, **kwargs))

# @register_model
# def resnest50d_4s2x40d(pretrained=False, **kwargs) -> ResNet:
#     """ResNeSt-50 4s2x40d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[3, 4, 6, 3],
#         stem_type='deep', stem_width=32, avg_down=True, base_width=40, cardinality=2,
#         block_args=dict(radix=4, avd=True, avd_first=True))
#     return _create_resnest('resnest50d_4s2x40d', pretrained=pretrained, **dict(model_kwargs, **kwargs))

# @register_model
# def resnest50d_1s4x24d(pretrained=False, **kwargs) -> ResNet:
    
#     """ResNeSt-50 1s4x24d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
#     """
#     model_kwargs = dict(
#         block=ResNestBottleneck, layers=[3, 4, 6, 3],
#         stem_type='deep', stem_width=32, avg_down=True, base_width=24, cardinality=4,
#         block_args=dict(radix=1, avd=True, avd_first=True))
#     return _create_resnest('resnest50d_1s4x24d', pretrained=pretrained, **dict(model_kwargs, **kwargs))

@BACKBONE_REGISTRY.register()
def build_resnest_backbone(cfg, input_shape):
    depth = cfg.MODEL.RESNETS.DEPTH
    norm_name = cfg.MODEL.RESNETS.NORM
    # # norm = GroupNorm
    if norm_name == "FrozenBN":
        norm = FrozenBatchNorm2d
    elif norm_name == "SyncBN":
        norm = NaiveSyncBatchNorm
    else:
        norm = NaiveSyncBatchNorm

    if depth == 50:
        layers = [3, 4, 6, 3]
    elif depth == 101:
        layers = [3, 4, 23, 3]
    elif depth == 200:
        layers = [3, 24, 36, 3]
    else:
        raise NotImplementedError()

    stage_blocks = []
    # use_deformable = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    for idx in range(4):
        stage_blocks.append("ResNestBottleneck")
    # resnest 101d
    model = ResNet(stage_blocks, layers, stem_type='deep', stem_width=64, avg_down=True, base_width=64, 
                #    norm_layer=norm,
                   cardinality=1,block_args=dict(radix=2, avd=True, avd_first=False))
    return model