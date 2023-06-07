import torch
import torch.nn as nn
import torch.nn.functional as F


class SRFD(nn.Module):  # x = [B, 3, 224, 224] -->  [B, embed_dim, 224/4, 224/4]
    def __init__(self, in_channels=3, out_channels=96, norm_layer=None):
        super().__init__()
        out_channels14 = int(out_channels / 4)  # embed_dim/4
        out_channels12 = int(out_channels / 2)  # embed_dim/2

        num_patches = (224 // 4) * (224 // 4)
        self.patches_resolution = [
            224 // 4, 224 // 4]
        self.num_patches = num_patches

        self.conv_init = nn.Conv2d(in_channels, out_channels14, kernel_size=7, stride=1, padding=3)
        self.conv_1 = nn.Conv2d(out_channels14, out_channels12, kernel_size=3, stride=1, padding=1, groups=out_channels14)
        self.conv_x1 = nn.Conv2d(out_channels12, out_channels12, kernel_size=3, stride=2, padding=1, groups=out_channels12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_channels12)
        self.cut_h = Cut(out_channels14, out_channels12)
        self.fusion1 = nn.Conv2d(out_channels, out_channels12, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(out_channels12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_channels12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_init(x)  # x = [B, embed_dim/4, H, W]
        h = x  # h = [B, embed_dim/4, H, W]
        x = self.conv_1(x)  # x = [B, embed_dim/2, H/2, W/2]
        h = self.cut_h(h)  # h = [B, embed_dim, H/2, W/2] --> [B, embed_dim/2, H/2, W/2]
        x = self.conv_x1(x)  # x = [B, embed_dim/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        x = torch.cat([x, h], dim=1)  # x = [B, embed_dim, H/2, W/2]
        x = self.fusion1(x)  # x = [B, embed_dim/2, H/2, W/2]

        r = x  # r = [B, embed_dim/2, H/2, W/2]
        x = self.conv_2(x)  # x = [B, embed_dim, H/2, W/2]
        m = x  # m = [B, embed_dim, H/2, W/2]
        x = self.conv_x2(x)  # x = [B, embed_dim, H/4, W/4]
        x = self.batch_norm_x2(x)
        m = self.max_m(m)  # m = [B, embed_dim, H/4, W/4]
        m = self.batch_norm_m(m)
        r = self.cut_r(r)  # r = [B, embed_dim, H/4, W/4]
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

    def forward(self, x, H, W):  # x = [B, C, H, W]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)  # x = [B, H, W, C]

        x = x.permute(0, 3, 1, 2)  # x = [B, C, H, W]
        h = x  # h = [B, C, H, W]
        h = self.conv(h)  # h = [B, 2C, H, W]
        m = h  # m = [B, 2C, H, W]

        x = self.cut_x(x)  # x = [B, 4C, H/2, W/2] --> x = [B, 2C, H/2, W/2]

        h = self.conv_h(h)  # h = [B, 2C, H/2, W/2]
        h = self.act_h(h)
        h = self.batch_norm_h(h)  # h = [B, 2C, H/2, W/2]

        m = self.max_m(m)  # m = [B, C, H/2, W/2]
        m = self.batch_norm_m(m)

        x = torch.cat([h, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 2C, H/2, W/2]

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        return x


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_partition2(x, window_size):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, C, H, W)  pytorch的卷积默认tensor格式为(B, C, H, W)
        window_size (tuple[int]): window size(M)
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    # view: -> [B, C, H//Wh, Wh, W//Ww, Ww]
    x = x.view(B, C, H // window_size[0], window_size[1], W // window_size[0], window_size[1])
    # permute: -> [B, H//Wh, W//Ww, Wh, Ww, C]
    # view: -> [B*num_windows, Wh, Ww, C]
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    num_windows = H//Wh * W//Ww
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Wh, Ww, C] -> [B, H//Wh, W//Ww, Wh, Ww, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H//Wh, Wh, W//Ww, Ww, C]
    # view: [B, H//Wh, Wh, W//Ww, Ww, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_reverse2(windows, window_size, H: int, W: int):
    """ Windows reverse to feature map.
    [B * H // win * W // win , win*win , C] --> [B, C, H, W]
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # view: [B*num_windows, N, C] -> [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, C, H//Wh, Wh, W//Ww, Ww]
    # view: [B, C, H//Wh, Wh, W//Ww, Ww] -> [B, C, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module
    with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, dwconv_kernel_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        # get pair-wise relative position index for each token inside the window
        relative_coords = self._get_rel_pos()
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw] 得到最终的相对位置偏置
        self.register_buffer("relative_position_index", relative_position_index)

        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)  # 在Attention分支，通道数减半
        self.proj_attn_norm = nn.LayerNorm(dim // 2)
        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),  # 在Attention分支通道数减半
        )
        self.projection = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm2d(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)  # 在Attention分支通道数减半
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)  # 最终空间信息输出通道为1
        )
        self.attn_norm = nn.LayerNorm(dim // 2)
        # final projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos(self):
        """
            Get pair-wise relative position index for each token inside the window.
            Args:
                window_size (tuple[int]): window size
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw] 建立了一个绝对位置矩阵
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw] 广播机制---前者将第3个维度复制Mh*Mw次，后者将第2个维度复制Mh*Mw次
        # 以当前像素点的绝对位置索引减去其它像素点的绝对位置索引，就能得到这个像素点的相对位置索引
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2] 得到最终的相对位置索引
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """
        # proj_attn(): -> [B*num_windows, N, C/2]  全连接层期待的输入tensor格式为: [B, *, C]
        x_atten = self.proj_attn_norm(self.proj_attn(x))
        # proj_cnn(): -> [B*num_windows, N, C]
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
        # window_reverse2(): -> [B, C, H, W]
        x_cnn = window_reverse2(x_cnn, self.window_size, H, W)

        # conv branch
        # dwconv3×3(): -> [B, C, H, W]
        x_cnn = self.dwconv3x3(x_cnn)
        # AvgPool2d(1): -> [B, C, 1, 1]  输入数据格式要求[B, C, H, W]
        # conv(): -> [B, C/8, 1, 1]
        # conv(): -> [B, C/2, 1, 1]  对应 在Attention分支通道数减半
        channel_interaction = self.channel_interaction(x_cnn)
        # projection(): -> [B, C/2, H, W]
        x_cnn = self.projection(x_cnn)

        # attention branch
        # B_: B*num_windows;  N: Window_size ** 2;  C: C/2  对应 在Attention分支通道数减半
        B_, N, C = x_atten.shape
        # qkv(): -> [B*num_windows, N, 3*C] --- C: C/2
        # reshape: -> [B*num_windows, N, 3, num_heads, C/num_heads] --- C: C/2
        # permute: -> [3, B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        qkv = self.qkv(x_atten).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # unbind(): -> [B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) 分别取出Q、K、V

        # channel interaction
        # reshape -> [B, 1, num_heads, 1, C/num_heads] --- C: C/2
        x_cnn2v = torch.sigmoid(channel_interaction).reshape(-1, 1, self.num_heads, 1, C // self.num_heads)
        # reshape: -> [B, num_heads, num_heads, N, C/num_heads] --- C: C/2
        v = v.reshape(x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads)
        # *: -> [B, num_heads, num_heads, N, C/num_heads] --- C: C/2
        v = v * x_cnn2v
        # reshape: -> [B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        v = v.reshape(-1, self.num_heads, N, C // self.num_heads)

        # transpose: -> [B*num_windows, num_heads, C/num_heads, N] --- C: C/2
        # @: multiply -> [B*num_windows, num_heads, N, N]
        q = q * self.scale  # Q/sqrt(dk)
        attn = (q @ k.transpose(-2, -1))  # Q*K^{T} / sqrt(dk)

        # relative_position_bias_table.view: [win*win*win*win,num_heads] -> [win*win*win*win,num_heads]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        # +: -> [B*num_windows, num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)  # 注意力+相对位置偏置

        # 如果有mask，直接对attn结果的对应部分加上mask的值，进行不连续区域的q*k值的掩盖
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            # softmax: -> [B*num_windows, num_heads, N, N]
            attn = self.softmax(attn)

        # [B*num_windows, num_heads, N, N]
        attn = self.attn_drop(attn)

        # @: multiply -> [B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        # transpose: -> [B*num_windows, N, num_heads, C/num_heads] --- C: C/2
        # reshape: -> [B*num_windows, N, C] --- C: C/2 对应 attention 分支通道数减半
        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        # window_reverse2: -> [B, C, H, W] --- C: C/2 对应 attention 分支通道数减半
        x_spatial = window_reverse2(x_atten, self.window_size, H, W)
        # conv: -> [B, C/8, H, W] --- C: C/2
        # conv: -> [B, 1, H, W]
        spatial_interaction = self.spatial_interaction(x_spatial)
        # sigmoid: -> [B, 1, H, W]
        # * -> [B, C, H, W] --- C: C/2
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn)
        # [B, C, H, W] --> [num_windows*B, N, C] --- C: C/2
        x_cnn = window_partition2(x_cnn, self.window_size)

        # concat
        x_atten = self.attn_norm(x_atten)
        # cat(): -> [num_windows*B, N, C] --- C: C
        x = torch.cat([x_cnn, x_atten], dim=2)
        # proj: -> [num_windows*B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA.
            We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, dwconv_kernel_size=3, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert self.shift_size == 0, "No shift in MixFormer"

        self.norm1 = norm_layer(dim)
        self.attn = MixAttention(
            dim, window_size=(self.window_size, self.window_size), dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        # [B, L, C] --- L: H * W
        shortcut = x
        x = self.norm1(x)
        # reshape(): -> [B, H, W, C]
        x = x.reshape(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            # 将输入数据从高度和宽度方向移动指定行和列
            # roll(): -> [B, H', W', C]
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            # [B, Hp, Wp, C]
            shifted_x = x
            attn_mask = None

        # partition windows 在SwinTransformerBlock部分才划分窗口
        # window_partition: -> [num_windows*B, window_size, window_size, C]
        x_windows = window_partition(shifted_x, self.window_size)
        # view: -> [num_windows*B, N, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # attn(): -> [num_windows*B, N, C]
        attn_windows = self.attn(x_windows, Hp, Wp, mask=attn_mask)

        # merge windows  计算完毕，从窗口变回数据
        # view(): -> [num_windows*B, window_size, window_size, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # window_reverse(): -> [B, Hp, Wp, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            # [B, H', W', C] -> [B, Hp, Wp, C]
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            # [B, Hp, Wp, C]
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉: -> [B, H, W, C]
            x = x[:, :H, :W, :].contiguous()

        # view(): -> [B, H*W, C]
        x = x.view(B, H * W, C)

        # FFN
        # [B, H*W, C]
        x = shortcut + self.drop_path(x)
        # mlp: -> [B, H*W, C]
        # +: -> [B, H*W, C]
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic layer for one stage in MixFormer.
    Modified from Swin Transformer BasicLayer.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end
            of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, dwconv_kernel_size=3, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, out_dim=0):
        super().__init__()
        # dim: [C, 2C, 4C]; out_dim: [2C, 4C, 8C]
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2  # 设置SW-MSA的移动行数和列数

        # build swinTransformer blocks
        self.blocks = nn.ModuleList([
            MixBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                     dwconv_kernel_size=dwconv_kernel_size, shift_size=0, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)])  # i初始值默认为0

        # conv merging layer
        if downsample is not None:
            self.downsample = downsample(in_channels=dim, out_channels=out_dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        :returns:
            stage1-3: H, W, [B, 2C, Wh, Ww], Wh, Ww
            stage4: H, W, [B, L, C], H, W
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # blk(): -> [B, H*W, C]
            x = blk(x, None)
        if self.downsample is not None:
            # downsample(): -> [B, 2C, Wh, Ww] --- Wh:H/2, Ww: W/2
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return H, W, x_down, Wh, Ww
        else:
            return H, W, x, H, W


class MixFormer(nn.Module):
    def __init__(self, img_size=224, class_num=1000, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, dwconv_kernel_size=3,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super(MixFormer, self).__init__()
        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        # embed_dim: [C, 2C, 4C, 8C]
        if isinstance(embed_dim, int):
            embed_dim = [embed_dim * 2 ** i_layer
                         for i_layer in range(self.num_layers)]
        assert isinstance(embed_dim, list) and \
               len(embed_dim) == self.num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim[-1])
        self.mlp_ratio = mlp_ratio

        # split image into patches
        self.patch_embed = SRFD(in_channels=3, out_channels=embed_dim[0])
        # self.patch_embed = LWPE(in_c=3, outdim=embed_dim[0])
        # self.patch_embed = ConvEmbed(img_size=img_size, patch_size=patch_size,
        #                              in_chans=in_chans, embed_dim=embed_dim[0],
        #                              norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        # 未使用绝对位置编码

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers 构建stage1-4
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim[i_layer]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=DRFD
                if (i_layer < self.num_layers - 1) else None,
                out_dim=int(self.embed_dim[i_layer + 1])
                if (i_layer < self.num_layers - 1) else 0)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.last_proj = nn.Linear(self.num_features, 1280)
        self.activate = nn.GELU()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1280, num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        :param x: input feature (B, C, H, W)
        :return: x: (B, 1280)
        """
        # patch_embed(): -> [B, C, Wh, Ww] --- C: embed_dim[0]; Wh: H/4; Ww: W/4
        x = self.patch_embed(x)
        _, _, Wh, Ww = x.shape
        # flatten(): -> [B, C, Wh*Ww] --- C: embed_dim[0]; Wh: H/4; Ww: W/4
        # permute(): -> [B, Wh*Ww, C] --- C: embed_dim[0]; Wh: H/4; Ww: W/4
        x = x.flatten(2).permute(0, 2, 1)
        x = self.pos_drop(x)

        for layer in self.layers:
            # stage1-3: H, W, [B, 2C, Wh, Ww], Wh, Ww --- H: Wh, W: Ww, Wh: H/2, Ww: W/2
            # stage4: H, W, [B, L, 8C], H, W  --- H: Wh/8, W: Ww/8, L: H*W
            H, W, x, Wh, Ww = layer(x, Wh, Ww)

        x = self.norm(x)  # B L C --- L: (Wh/8)*(Ww/8), C: 8C
        # last_proj(): -> [B, L, 1280] --- L: (Wh/8)*(Ww/8)
        x = self.last_proj(x)
        x = self.activate(x)
        # permute: -> [B, 1280, L]
        # avgpool: -> [B, 1280, 1]
        x = self.avgpool(x.permute(0, 2, 1))
        # flatten: -> [B, 1280]
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """
        :param x: input feature (B, C, H, W)
        :return: x : [B, num_classes]
        """
        # forward_features(): -> [B, 1280]
        x = self.forward_features(x)
        # head(): -> [B, num_classes]
        x = self.head(x)
        return x


def MixFormer_B0(num_classes: int = 1000, **kwargs):
    model = MixFormer(
        window_size=7,
        embed_dim=24,
        depths=[1, 2, 6, 6],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.,
        num_classes=num_classes,
        **kwargs)
    return model


def MixFormer_B1(num_classes: int = 1000, **kwargs):
    model = MixFormer(
        embed_dim=32,
        depths=[1, 2, 6, 6],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.,
        num_classes=num_classes,
        **kwargs)
    return model


def MixFormer_B2(num_classes=1000, **kwargs):
    model = MixFormer(num_classes=num_classes,
                      embed_dim=32,
                      depths=[2, 2, 8, 8],
                      num_heads=[2, 4, 8, 16],
                      drop_path_rate=0.05,
                      **kwargs)
    return model


def MixFormer_B3(num_classes=1000, **kwargs):
    model = MixFormer(num_classes=num_classes,
        embed_dim=48,
        depths=[2, 2, 8, 6],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.1,
        **kwargs)
    return model


def MixFormer_B4(num_classes=1000, **kwargs):
    model = MixFormer(num_classes=num_classes,
        embed_dim=64,
        depths=[2, 2, 8, 8],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.2,
        **kwargs)
    return model


def MixFormer_B5(num_classes=1000, **kwargs):
    model = MixFormer(num_classes=num_classes,
        embed_dim=96,
        depths=[1, 2, 8, 6],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.3,
        **kwargs)
    return model


def MixFormer_B6(num_classes=1000, **kwargs):
    model = MixFormer(num_classes=num_classes,
                      embed_dim=96,
                      depths=[2, 4, 16, 12],
                      num_heads=[6, 12, 24, 48],
                      drop_path_rate=0.5,
                      **kwargs)
    return model



