import torch
from torch import nn
import torch.nn.functional as F
from itertools import repeat
import collections


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


import torch
import torch.nn as nn
import collections
from itertools import repeat


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)

        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)

        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.0,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def resize_complex_weight(origin_weight, new_h, new_w):

    h, w, num_heads = origin_weight.shape[0:3]  # (size, w, c, 2)

    origin_weight = origin_weight

    # Reshape and interpolate
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = F.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)

    return new_weight


class YoloTwoChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloTwoChannels, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1x1_fi = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_1x1_fi1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_3x3_fi = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv_5x5_fi = nn.Conv2d(out_channels * 2, out_channels, kernel_size=5, padding=2)

        self.conv_3x3_fi1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv_5x5_fi1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=5, padding=2)

        self.conv_3x3_final = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv_5x5_final = nn.Conv2d(out_channels * 2, out_channels, kernel_size=5, padding=2)

        self.conv_1x1_fi_final = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv_1x1_fi1_final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.output_layer = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        f_i, f_ip1 = x, x  # 这里假设输入 x 是单个张量
        f_i_blk = EMA(self.in_channels).to(x.device)
        f_i = f_i_blk(f_i)
        f_ip1_blk = DynamicFilter(self.in_channels).to(x.device)
        f_ip1 = f_ip1_blk(f_ip1)

        f_i_1x1 = self.conv_1x1_fi(f_i)
        f_ip1_1x1 = self.conv_1x1_fi1(f_ip1)

        concat_1 = torch.cat([f_i_1x1, f_ip1_1x1], dim=1)

        conv_3x3_1 = self.conv_3x3_fi(concat_1)
        conv_5x5_1 = self.conv_5x5_fi(concat_1)

        concat_2 = torch.cat([conv_3x3_1, conv_5x5_1], dim=1)

        conv_3x3_2 = self.conv_3x3_final(concat_2)
        conv_5x5_2 = self.conv_5x5_final(concat_2)

        f_i_1x1_final = self.conv_1x1_fi_final(f_i)
        f_ip1_1x1_final = self.conv_1x1_fi1_final(f_ip1)

        weighted_sum = f_i_1x1_final + f_ip1_1x1_final + conv_3x3_2 * conv_5x5_2

        out = self.output_layer(weighted_sum)

        return out

if __name__ == "__main__":
    run = 0
    if run == 0:
        input_tensor = torch.randn(1, 1024, 8, 8)
        output = YoloTwoChannels(1024, 1024)
        output = output(input_tensor)
        print(output.shape)
    else:
        input_tensor = torch.randn(1, 1024, 8, 8)
        output = DynamicFilter(1024)
        output = output(input_tensor)
        print(output.shape)
