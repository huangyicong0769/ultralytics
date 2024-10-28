# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Module utils."""

import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = "multi_scale_deformable_attn_pytorch", "inverse_sigmoid"


def _get_clones(module, n):
    """Create a list of cloned modules from the given module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multiscale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    # 若设置了警告并且使用 align_corners, 则给出警告
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])  # 获取输入的高宽
            output_h, output_w = tuple(int(x) for x in size)  # 获取输出的高宽
            # 判断插值条件, 给出警告提示
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    # 使用 PyTorch 提供的插值函数
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M: 窗口的行数
    - N: 窗口的列数

    返回：
    - 二维Hamming窗
    """
    hamming_x = np.hamming(M)  # 生成水平 Hamming 窗
    hamming_y = np.hamming(N)  # 生成垂直 Hamming 窗
    hamming_2d = np.outer(hamming_x, hamming_y)  # 通过外积生成二维 Hamming 窗
    return hamming_2d

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量, 形状为[B, C, H, W]
    - k: 范围大小, 表示周围KxK范围内的点
    - dilation: 膨胀系数, 用于调整卷积的感受野
    - sim: 使用的相似度类型, 默认为余弦相似度 'cos'

    返回：
    - 输出张量, 形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape  # 获取批量大小、通道数、高度和宽度
    
    # 使用 unfold 展开张量, 将每个像素周围的 KxK 区域展平
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # B, CxKxK, HW
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)  # 将张量重塑为 [B, C, KxK, H, W] 形状

    # 计算中心像素与周围像素的余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    # 如果使用点积相似度
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError  # 如果提供的相似度计算类型不被支持, 抛出异常

    # 移除中心像素的相似度, 只保留周围的 KxK-1 的相似度值
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将相似度结果调整为 [B, KxK-1, H, W] 的形状
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity  # 返回相似度结果
   
def carafe(x:torch.Tensor, mask:torch.Tensor|None=None, k=5, g=1, s=2)->torch.Tensor:
    b, c, h, w = x.shape
    h_, w_ = h*s, w*s
    assert mask is not None
    # print(f"fCARAFE:x={x.shape}, mask={mask.shape}, s={s}")
    x = F.upsample(x, scale_factor=s, mode='nearest')
    x = F.unfold(x, kernel_size=k, dilation=s, padding=k//2*s).view(b, c, -1, h_, w_)
    # print(f"fCARAFE:x={x.shape}")
    return torch.einsum('bkhw,bckhw->bchw', [mask, x])

def spatial_selective(x:list[torch.Tensor], act:callable=nn.Identity())->list[torch.Tensor]:
    y = torch.cat(x, dim=1)
    y_feat = torch.cat([
        y.mean(dim=1, keepdim=True),
        y.max(dim=1, keepdim=True)[0],
        y.std(dim=1, keepdim=True, unbiased=False).nan_to_num(0.0)
    ], dim=1)
    y_feat = act(y_feat)
    return [xi * y_feat[:,i,:,:].unsqueeze(1) for i, xi in enumerate(x)]
