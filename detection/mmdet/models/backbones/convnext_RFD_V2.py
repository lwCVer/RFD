# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from mmcv_custom_convnext import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES


class SRFD(nn.Module):  # x = [B, 3, 224, 224] -->  [B, embed_dim, 224/4, 224/4]
    def __init__(self, in_channels=3, out_channels=96, norm_layer=None):
        super().__init__()
        out_channels14 = int(out_channels / 4)  # embed_dim/4
        out_channels12 = int(out_channels / 2)  # embed_dim/2
        # 第一个下采样
        self.conv_init = nn.Conv2d(in_channels, out_channels14, kernel_size=7, stride=1, padding=3)
        self.conv_1 = nn.Conv2d(out_channels14, out_channels12, kernel_size=3, stride=1, padding=1, groups=out_channels14)
        # 分zu卷积下采样
        self.conv_x1 = nn.Conv2d(out_channels12, out_channels12, kernel_size=3, stride=2, padding=1, groups=out_channels12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_channels12)
        # 切片下采样
        self.cut_h = Cut(out_channels14, out_channels12)
        # 融合
        self.fusion1 = nn.Conv2d(out_channels, out_channels12, kernel_size=1, stride=1)
        # 第二个下采样
        self.conv_2 = nn.Conv2d(out_channels12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels12)
        # 分zu卷积下采样
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        # 最大池化下采样
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        # 切片下采样
        self.cut_r = Cut(out_channels12, out_channels)
        # 融合
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 先1*1卷积成embed_dim/4通道
        x = self.conv_init(x)  # x = [B, embed_dim/4, H, W]
        # 第一个下采样仅有卷积和切片
        h = x  # h = [B, embed_dim/4, H, W]
        x = self.conv_1(x)  # x = [B, embed_dim/2, H/2, W/2]
        # 切片下采样
        h = self.cut_h(h)  # h = [B, embed_dim, H/2, W/2] --> [B, embed_dim/2, H/2, W/2]
        # 卷积下采样
        x = self.conv_x1(x)  # x = [B, embed_dim/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # 拼接
        x = torch.cat([x, h], dim=1)  # x = [B, embed_dim, H/2, W/2]
        x = self.fusion1(x)  # x = [B, embed_dim/2, H/2, W/2]

        # 第二个下采样有卷积、池化和切片
        r = x  # r = [B, embed_dim/2, H/2, W/2]
        x = self.conv_2(x)  # x = [B, embed_dim, H/2, W/2]
        m = x  # m = [B, embed_dim, H/2, W/2]
        # 分zu卷积下采样
        x = self.conv_x2(x)  # x = [B, embed_dim, H/4, W/4]
        x = self.batch_norm_x2(x)
        # 最大池化下采样
        m = self.max_m(m)  # m = [B, embed_dim, H/4, W/4]
        m = self.batch_norm_m(m)
        # 切片下采样
        r = self.cut_r(r)  # r = [B, embed_dim, H/4, W/4]
        # 拼接
        x = torch.cat([x, r, m], dim=1)  # x = [B, embed_dim*3, H/4, W/4]
        x = self.fusion2(x)  # x = [B, embed_dim, H/4, W/4]

        return x


class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)
        x = self.batch_norm(x)
        return x


class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_h = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_h = nn.GELU()
        self.cut_x = Cut(in_channels=in_channels, out_channels=out_channels)
        self.batch_norm_h = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]
        h = x  # h = [B, H, W, C]
        h = self.conv(h)  # h = [B, 2C, H, W]
        m = h  # m = [B, 2C, H, W]

        # 切片下采样
        x = self.cut_x(x)  # x = [B, 4C, H/2, W/2] --> x = [B, 2C, H/2, W/2]

        # 卷积下采样
        h = self.conv_h(h)  # h = [B, 2C, H/2, W/2]
        h = self.act_h(h)
        h = self.batch_norm_h(h)  # h = [B, 2C, H/2, W/2]

        # 最大池化下采样
        m = self.max_m(m)  # m = [B, C, H/2, W/2]
        m = self.batch_norm_m(m)

        # 拼接
        x = torch.cat([h, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 2C, H/2, W/2]

        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


@BACKBONES.register_module()
class ConvNeXt_RFD_V2(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()    # stem and 3 intermediate downsampling conv layers
        # stem = LWPE(in_c=in_chans, embed_dim=dims[0])
        stem = SRFD(in_channels=in_chans, out_channels=dims[0])
        self.downsample_layers.append(stem)
        for i in range(3):
            # downsample_layer = nn.Sequential(
            #     LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
            #     # nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            # )
            # downsample_layer = LWRD(dims[i], dims[i+1])
            downsample_layer = DRFD(dims[i], dims[i+1])
            self.downsample_layers.append(downsample_layer)
            self.stages = nn.ModuleList()    # 4 feature resolution stages, each consisting of multiple residual blocks
            dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
