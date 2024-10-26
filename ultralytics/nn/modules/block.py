# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock, MLPBlock, LayerNorm2d, MLP, DropPath
from .utils import normal_init, constant_init, resize, hamming2D, compute_similarity, carafe

from math import log2
from typing import Literal

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "SEAttention",
    "MCA",
    "C2fMCA",
    "C3kMCA",
    "C3k2MCA",
    "C2fMCAELAN4",
    "LowFAM",
    "LowIFM",
    "Split",
    "HighFAM",
    "HighIFM",
    "LowLAF",
    "HiLAF",
    "Inject",
    "CARAFE"
    "FreqFusion",
    "C2INXB",
    "C2PSSA",
    "ConvHighIFM",
    "C2fLSK",
    "MSSA",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


# class C2fAttn(nn.Module):
#     """C2f module with an additional attn module."""

#     def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
#         """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

#     def forward(self, x, guide):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x, guide):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class SEAttention(nn.Module):
    def __init__(self, c1, c2, r=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(        
            nn.Linear(c1, c1//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1//r, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.avgpool(x).view(b, c)).view(b, c, 1, 1).expand_as(x)
    
class StdPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        b, c, _, _ = x.size()
        x = x.nan_to_num(0.0)
        std = x.view(b, c, -1).std(dim=2, keepdim=True, unbiased=False).nan_to_num(0.0)
        std = std.reshape(b, c, 1, 1)
        return std
        
class MCAGate(nn.Module):
    def __init__(self, k=3, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super().__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
            
        # self.conv = nn.Conv2d(1, 1, kernel_size=(1, k), stride=1, padding=(0, (k-1)//2), bias=False)
        self.conv = Conv(1, 1, (1, k), 1, (0, (k-1)//2))
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        f = [pool(x) for pool in self.pools]

        if len(f) == 1:
            out = f[0]
        elif len(f) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1/2*(f[0]+f[1]) + weight[0]*f[0]+weight[1]*f[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x*out

class MCA(nn.Module):
    def __init__(self, c1, no_spatial=False):
        """Constructs a MCA module.
        Args:
            c1: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super().__init__()

        self.h_cw = MCAGate()
        self.w_hc = MCAGate()
        self.no_spatial = no_spatial

        if not no_spatial:
            l = 1.5
            g = 1
            temp = round(abs((log2(c1) - g) / l))
            k = temp if temp % 2 else temp - 1

            self.c_hw = MCAGate(k=k)

    def transplit(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            return x_h, x_w, x_c
        else:
            return x_h, x_w

    def forward(self, x):
        if not self.no_spatial:
            x_h, x_w, x_c = self.transplit(x)
            return 1/3*(x_h+x_w+x_c)
        else:
            x_h, x_w = self.transplit(x)
            return 1/2*(x_h+x_w)

class pMCA(MCA):
    def __init__(self, c, no_spatial=False):
        super().__init__(c, no_spatial)
        self.weights = [nn.Parameter(torch.tensor(1., requires_grad=True)) for _ in range(2 if no_spatial else 3)]
    
    def forward(self, x):
        if not self.no_spatial:
            x_h, x_w, x_c = self.transplit(x)
            return 1/3*(x_h*self.weights[0]+x_w*self.weights[1]+x_c*self.weights[2])
        else:
            x_h, x_w = self.transplit(x)
            return 1/2*(x_h*self.weights[0]+x_w*self.weights[1])

class MSSA(MCA):
    '''Multidimensional Spatial Selective Attention Module'''
    def __init__(self, c1, no_spatial=False):
        assert no_spatial == False
        super().__init__(c1, no_spatial)
        self.cv1 = Conv(3, 3, 7)
        # self.cv2 = Conv(c1, c1, 1)

    def forward(self, x):
        x_h, x_w, x_c = self.transplit(x)
        
        y = torch.cat([x_h, x_w, x_c], dim=1).nan_to_num(0.0)
        y_avg = y.mean(dim=1, keepdim=True)
        y_max, _ = y.max(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True, unbiased=False).nan_to_num(0.0)
        y = torch.cat([y_avg, y_max, y_std], dim=1)
        y = self.cv1(y)

        y = 1/3*(x_h * y[:,0,:,:].unsqueeze(1) + x_w * y[:,1,:,:].unsqueeze(1) + x_c * y[:,2,:,:].unsqueeze(1))
        # y = self.cv2(y)
        return y


class BottleneckAttn(Bottleneck):
    """Bottleneck with Attention."""

    def __init__(self, c1, c2, attn=nn.Identity, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a MCA bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        self.attn = attn(c2)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        y = self.cv2(self.cv1(x))
        y = self.attn(y)
        return x + y if self.add else y
    
class C2fAttn(C2f):
    """C2f with Attention modified Bottlesneck."""

    def __init__(self, c1, c2, n=1, attn=nn.Identity, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BottleneckAttn(self.c, self.c, attn, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

class C3kAttn(C3k):
    """C2f with Attention modified Bottlesneck."""

    def __init__(self, c1, c2, n=1, attn=nn.Identity, shortcut=False, g=1, e=0.5, k=3):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(BottleneckAttn(c_, c_, attn, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2Attn(C3k2):
    """C2f with Attention modified Bottlesneck."""

    def __init__(self, c1, c2, n=1, attn=nn.Identity, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3kAttn(self.c, self.c, 2, attn, shortcut, g) if c3k else BottleneckAttn(self.c, self.c, attn, shortcut, g) for _ in range(n)
        )


class C2fAttnELAN4(RepNCSPELAN4):
    """C2f-Attention-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1, attn=nn.Identity, shortcut=False):
        super().__init__(c1, c2, c3, c4, n)
        self.cv2 = nn.Sequential(C2fAttn(c3 // 2, c4, n, attn, shortcut), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(C2fAttn(c4, c4, n, attn, shortcut), Conv(c4, c4, 3, 1))

# class mBottleneck(nn.Module):
#     def __init__(self, c1, c2, shortcut=False, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2*e)
#         self.cv1 = RepConv(c1, c_)
#         self.cv2 = RepVGGDW(c_)
#         self.cv3 = RepConv(c_, c2)
#         self.attn = MCA(c2)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.attn(self.cv3(self.cv2(self.cv1(x)))) if self.add else self.attn(self.cv3(self.cv2(self.cv1(x))))

class LowFAM(nn.Module):
    """Low-stage feature alignment module."""

    def __init__(self, c_u=None, sample:Literal['bilinear', 'carafe', 'FreqFusion']='bilinear'):
        super().__init__()
        self.sample = sample
        if self.sample == 'bilinear':
            self.us = F.interpolate
        elif self.sample == 'carafe':
            assert c_u is not None
            self.us = CARAFE(c_u, 2)
        elif self.sample == 'FreqFusion':
            assert c_u is not None
            self.us = FreqFusion(c_u, c_u//2)
        else:
            raise NotImplementedError
        self.ds = F.adaptive_avg_pool2d

    def forward(self, x:list):
        _, _, H, W = x[1].shape
        if self.sample == 'bilinear':
            y = self.us(x[0], (H, W), mode='bilinear', align_corners=False)
        elif self.sample == 'carafe':
            y = self.us(x[0])
        elif self.sample == 'FreqFusion':
            y, x[1] = self.us(x_l=x[0], x_h=x[1])
        else:
            raise NotImplementedError
        return torch.cat([y, x[1]] + [self.ds(x_, [H, W]) for x_ in x[2:]], dim=1)
        
class LowIFM(nn.Module):
    """Low-stage information fusion module."""

    def __init__(self, c1, c2, n=1, e=0.5, k=(1, 3)):
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, k=1)
        # self.m = nn.Sequential(*(RepVGGDW(c_) for _ in range(n)))
        # self.m = nn.Sequential(*(mBottleneck(c2, c2, e=e) for _ in range(n)))
        self.conv2 = Conv(c_, c2, k=1)
        # self.attn = MCA(c_)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('RepVGGDW', RepVGGDW(c_))
            self.m.add_module('MCA', MCA(c_))

    def forward(self, x):
        # y = self.m(self.conv1(x))
        y = self.conv2(self.m(self.conv1(x)))
        return y
    
class Split(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x:torch.Tensor):
        return x.split(self.c, dim=1)

class HighFAM(nn.Module):
    """High-stage feature alignment module."""
    
    def __init__(self):
        super().__init__()
        self.ds = F.adaptive_avg_pool2d

    def forward(self, x:list[torch.Tensor]):
        _, _, H, W = x[0].shape
        return torch.cat([x[0]] + [self.ds(x_, [H, W]) for x_ in x[1:]], dim=1)

# class MLP(nn.Module):
#     def __init__(self, c1, c2, h=None, d=0.):
#         super().__init__()
#         c_ = h or c1
#         self.fc1 =Conv(c1, c_, act=False)
#         self.conv = DWConv(c_, c_, 3, 1, 1)
#         self.fc2 = Conv(c_, c2, act=False)
#         self.drop = nn.Dropout(d)

#     def forward(self, x):
#         return self.drop(self.fc2(self.drop(self.act(self.conv(self.fc1(x))))))
        
# class topBlock(nn.Module):
#     def __init__(self, dim, n, attn_r = 2., mlp_r = 4., d = 0.):
#         super().__init__()
#         self.attn = Attention(dim, n, attn_r)
#         mlp_ = int(dim * mlp_r)
#         self.mlp = MLP(dim, dim, mlp_, d)

#     def forward(self, x):
#         y = x + self.attn(x)
#         y = y + self.mlp(y)
#         return y

class ConvHighIFM(nn.Module):
    """ High-stage information fusion module.
        Replace ViT to Conv
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_)
        self.conv2 = Conv(c_, c2)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('RepVGGDW', RepVGGDW(c_))
            self.m.add_module('MCA', MCA(c_))

    def forward(self, x):
        # y = self.m(self.conv1(x)) + self.conv3(x)
        y = self.conv2(self.m(self.conv1(x)))
        return y

class HighIFM(ConvHighIFM):
    def __init__(self, c1, c2, n=1, e=0.5, num_head=4):
        super().__init__(c1, c2, n, e)
        c_ = int(c2*e)
        self.m = nn.Sequential(*(StokenAttentionLayer(c_, 1, (1, 1), num_head, drop=0.1, layerscale=True) for _ in range(n)))

class LowLAF(nn.Module):
    """Low-stage lightweight adjacent layer fusion module"""

    def __init__(self, c1, c2, c_s, c_l, sample:Literal['bilinear', 'carafe', 'FreqFusion']='bilinear'):
        super().__init__()
        self.conv1 = Conv(c1, c1)
        self.conv2 = Conv(c_s, c2)
        self.sample = sample
        if self.sample == 'bilinear':
            self.us = F.interpolate
        elif self.sample == 'carafe':
            self.us = CARAFE(c_l, 2)
        elif self.sample == 'FreqFusion':
            self.us = FreqFusion(c_l, c1)
        else:
            raise NotImplementedError
        self.ds = F.adaptive_avg_pool2d

    def forward(self, x:list[torch.Tensor]):
        _, _, H, W = x[1].shape

        if self.sample == 'bilinear':
            x[0] = self.us(x[0], (H, W), mode='bilinear', align_corners=False)
        elif self.sample == 'carafe':
            x[0] = self.us(x[0])
        elif self.sample == 'FreqFusion':
            x[0], x[1] = self.us(x_l=x[0], x_h=x[1])
        else:
            raise NotImplementedError
        x[1] = self.conv1(x[1])
        x[2] = self.ds(x[2], [H, W])

        return self.conv2(torch.cat(x, 1))
    
class HiLAF(nn.Module):
    """High-stage lightweight adjacent layer fusion module"""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1, c2)
        self.ds = F.adaptive_avg_pool2d

    def forward(self, x:list[torch.Tensor]):
        _, _, H, W = x[0].shape
        x[1] = self.ds(x[1], [H, W])
        return self.conv(torch.cat(x, 1))
    
class Inject(nn.Module):
    """Information injection module"""

    def __init__(self, c1, c2, index, n=1, sample:Literal['avgpool', 'bilinear', 'carafe', 'FreqFusion']='bilinear'):
        super().__init__()
        
        self.index = index
        self.conv1 = Conv(c1, c2, act=False)
        self.conv2 = Conv(c1, c2, act=False)
        self.conv3 = Conv(c1, c2, act=False)
        self.conv4 = nn.Sequential(*[RepVGGDW(c2) for _ in range(n)])
        self.act = nn.Sigmoid()
        self.sample = sample
        if self.sample == 'carafe':
            self.us1 = CARAFE(c2, 2)
            self.us2 = CARAFE(c2, 2)
        elif self.sample == 'FreqFusion':
            self.us1 = FreqFusion(c2, c2)
            self.us2 = FreqFusion(c2, c2)
            self.w1, self.w2 = torch.tensor(1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)

    def forward(self, x:list[torch.Tensor]):

        # print(f"{x[0].shape}, {x[1][self.index].shape}")

        x1 = self.conv1(x[0])
        x2 = x[1][self.index]

        _, _, H, W = x1.shape
        x_ = self.act(self.conv2(x2))
        x2 = self.conv3(x2)

        if self.sample == 'avgpool':
            y = F.adaptive_avg_pool2d(x2, [H, W]) + x1 * F.adaptive_avg_pool2d(x_, [H, W])
        elif self.sample == 'bilinear':
            y = F.interpolate(x2, (H, W), mode='bilinear', align_corners=False) + x1 * F.interpolate(x_, (H, W), mode='bilinear', align_corners=False)
        elif self.sample == 'carafe':
            y = self.us1(x2) + x1 * self.us2(x_)
        elif self.sample == 'FreqFusion':
            x2, x11 = self.us1(x_l=x2, x_h=x1)
            x_, x12 = self.us2(x_l=x_, x_h=x1)
            x1 = (x11*self.w1+x12*self.w2)*0.5
            y = x2 + x1 * x_
        else:
            raise NotImplementedError

        return self.conv4(y)

class CARAFE(nn.Module):
    '''Content-Aware ReAssembly of FEatures'''

    def __init__(self, c1, s=2, c_=64, k=5):
        super().__init__()
        self.s = s
        self.compressor = Conv(c1, c_)
        self.encoder = Conv(c_, s**2 * k**2, k=k-2, act=False)
        self.shuffle = nn.PixelShuffle(s)
        self.normalizer = nn.Softmax(dim=1)

        self.us = nn.Upsample(scale_factor=s, mode='nearest')
        self.unfold = nn.Unfold(k, dilation=s, padding=k//2*s)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor|None=None):
        b, c, h, w = x.shape
        h_, w_ = h*self.s, w*self.s

        if mask is None:
            mask = self.normalizer(self.shuffle(self.encoder(self.compressor(x))))

        # print(f"CARAFE:x={x.shape}, mask={mask.shape}, s={self.s}")

        x = self.us(x)
        # print(f"CARAFE:x={x.shape}")
        x = self.unfold(x)
        # print(f"CARAFE:x={x.shape}")
        x = x.view(b, c, -1, h_, w_)
        # print(f"CARAFE:x={x.shape}")
        return torch.einsum('bkhw,bckhw->bchw', [mask, x])

class FreqFusion(nn.Module):
    '''See https://github.com/Linwei-Chen/FreqFusion'''

    def __init__(self, c_l, c_h, s=1, k_l=5, k_h=3, g_u=1, k_e=3, d_e=1, c_=64, align_corners=False, upsample_mode='nearest', feature_resample=False, g_feature_resample=4, comp_feat_upsample=True, use_high_pass=True, use_low_pass=True, residual_h=True, semi_conv=True, hamming_window=True, feature_resample_norm=True):
        super().__init__()
        self.s = s
        self.k_l = k_l
        self.k_h = k_h
        self.g_u = g_u
        self.k_e = k_e
        self.d_e = d_e
        self.c_ = c_

        self.compressor_h = Conv(c_h, c_, 1, act=False)
        self.compressor_l = Conv(c_l, c_, 1, act=False)

        self.alpf = Conv(c_, k_l**2 * g_u * s**2, k_e, p=int((k_e-1)*d_e/2), d=d_e, g=1, act=False) #Adaptive Low-Pass Filte

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.residual_h = residual_h
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample

        if self.feature_resample:
            self.dysampler = LocalSimGuideSampler(c_, 2, 'lp', g_feature_resample, True, k_e, norm=feature_resample_norm)

        if self.use_high_pass: #Adaptive High-Pass Filte
            self.ahpf = Conv(c_, k_h**2 * g_u * s**2, k_e, p=int((k_e-1)*d_e/2), d=d_e, g=1, act=False)

        self.hamming_window = hamming_window
        p_l = 0
        p_h = 0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(hamming2D(k_l+2*p_l, k_l+2*p_l))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(hamming2D(k_h+2*p_h, k_h+2*p_h))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))

        self.init_weights()

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, Conv):
        #         nn.init.xavier_uniform_(m)
        normal_init(self.alpf, std=0.001)
        if self.use_high_pass:
            normal_init(self.ahpf, std=0.001)

    def kernel_normalize(self, mask, k, s=None, hamming = 1):
        if s is not None:
            mask = F.pixel_shuffle(mask, s)
        n, mask_c, h, w = mask.size()
        c_mask = int(mask_c/float(k**2))
        mask = mask.view(n, c_mask, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, c_mask, k, k, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).contiguous().view(n, -1, k, k)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdim=True)
        mask = mask.view(n, c_mask, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).contiguous().view(n, -1, h, w)
        return mask

    def forward(self, x_l:torch.Tensor|list[torch.Tensor], x_h:torch.Tensor|None=None):
        if x_h is None:
            x_h = x_l[1]
            x_l = x_l[0]

        xcl = self.compressor_l(x_l)
        xch = self.compressor_h(x_h)

        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    #AHPF
                    mask_hh = self.ahpf(xch)
                    mask_hi = self.kernel_normalize(mask_hh, self.k_h, hamming=self.hamming_highpass)
                    xch = xch + xch - carafe(xch, mask_hi, self.k_h, self.g_u, 1)

                    #ALPF
                    mask_lh = self.alpf(xch)
                    mask_li = self.kernel_normalize(mask_lh, self.k_l, hamming=self.hamming_lowpass)
                    mask_lll = self.alpf(xcl)
                    mask_ll = F.interpolate(carafe(mask_lll, mask_li, self.k_l, self.g_u, 2), size=xch.shape[-2:], mode='nearest')
                    mask_l = mask_lh + mask_ll

                    mask_li = self.kernel_normalize(mask_l, self.k_l, hamming=self.hamming_lowpass)
                    mask_hl = F.interpolate(carafe(self.ahpf(xcl), mask_li, self.k_l, self.g_u, 2), size=xch.shape[-2:], mode='nearest')

                    mask_h = mask_hh+mask_hl
                else:
                    raise NotImplementedError
            else:
                mask_l = self.alpf(xch) + F.interpolate(self.alpf(xcl), size=xch.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_h = self.ahpf(xch) + F.interpolate(self.ahpf(xcl), size=xch.shape[-2:], mode='nearest')
        else:
            xc = F.interpolate(xcl, size=xch.shape[-2:], mode='nearest') + xch
            mask_l = self.alpf(xc)
            if self.use_high_pass:
                mask_h = self.ahpf(xc)

        mask_l = self.kernel_normalize(mask_l, self.k_l, hamming=self.hamming_lowpass)
        
        if self.semi_conv:
            x_l = carafe(x_l, mask_l, self.k_l, self.g_u, 2)
        else:
            x_l = resize(input=x_l, size=x_h.shape[2:], mode=self.upsample_mode, align_corners=None if self.upsample_mode=='nearest' else self.align_corners)
            x_l = carafe(x_l, mask_l, self.k_l, self.g_u, 1)

        if self.use_high_pass:
            mask_h = self.kernel_normalize(mask_h, self.k_h, hamming=self.hamming_highpass)
            x_hh = x_h - carafe(x_h, mask_h, self.k_h, self.g_u, 1)
            x_h = (x_hh + x_h) if self.residual_h else x_hh

        if self.feature_resample:
            x_l = self.dysampler(x_l=xcl, x_h=xch, feat2sample=x_l)

        return x_l, x_h
    
class LocalSimGuideSampler(nn.Module):
    '''Offset Generator in FreqFusion'''
    def __init__(self, c1, s, style='lp', g=4, use_direct_scale=True, k=1, local_window=3, sim_type='cos', norm=True, direction_feat='sim_concat'):
        super().__init__()
        assert s == 2
        assert style == 'lp'

        self.s = s
        self.style = style
        self.g = g
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        assert c1 >= g and c1 % g == 0

        if style == 'lp':
            c1 = c1 // s**2
            c2 = 2*g
        else:
            c2 = 2*g*s**2

        if self.direction_feat == 'sim':
            self.offset = Conv(local_window**2-1, c2, k, p=k//2, act=False)
        elif self.direction_feat == 'sim_concat':
            self.offset = Conv(c1 + local_window**2 - 1, c2, k, p = k//2, act=False)
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale = Conv(c1, c2, k, p=k//2, act=False)
            elif self.direction_feat == 'sim_concat':
                self.direct_scale = Conv(c1 + local_window**2 - 1, c2, k, p=k//2, act=False)
            else:
                raise NotImplementedError
            constant_init(self.direct_scale, val=0.)

        if self.direction_feat == 'sim':
            self.offset_h = Conv(local_window**2 - 1, c2, k, p=k//2, act=False)
        elif self.direction_feat == 'sim_concat':
            self.offset_h = Conv(c1 + local_window**2 - 1, c2, k, p=k//2, act=False)
        else:
            raise NotImplementedError
        normal_init(self.offset_h, std=0.001)

        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale_h = Conv(c1, c2, k, p=k//2, act=False)
            elif self.direction_feat == 'sim_concat':
                self.direct_scale_h = Conv(c1 + local_window**2 - 1, c2, k, p=k//2, act=False)
            else:
                raise NotImplementedError
            constant_init(self.direct_scale_h, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_h = nn.GroupNorm(c1//8, c1)
            self.norm_l = nn.GroupNorm(c1//8, c1)
        else:
            self.norm_h = nn.Identity()
            self.norm_l = nn.Identity()

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.s+1)/2, (self.s-1)/2+1)/self.s
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.g, 1).reshape(1, -1, 1, 1)
    
    def sample(self, x:torch.Tensor, offset:torch.Tensor, s=None):
        if s is None:
            s = self.s
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        coord_h = torch.arange(h) + 0.5
        coord_w = torch.arange(w) + 0.5
        coords = torch.stack(torch.meshgrid([coord_w, coord_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(b, -1, h, w), s).view(b, 2, -1, s*h, s*w).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(b*self.g,-1, x.size(-2), x.size(-1)), coords, mode='bilinear', align_corners=False, padding_mode='border').view(b, -1, s*h, s*w)
    
    def forward(self, x_l, x_h, feat2sample):
        x_l = self.norm_l(x_l)
        x_h = self.norm_h(x_h)

        if self.direction_feat == 'sim':
            x_l = compute_similarity(x_l, self.local_window, dilation=2, sim='cos')
            x_h = compute_similarity(x_h, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            sim_l = torch.cat([x_l, compute_similarity(x_l, self.local_window, dilation=2, sim='cos')], dim=1)
            sim_h = torch.cat([x_h, compute_similarity(x_h, self.local_window, dilation=2, sim='cos')], dim=1)
            x_l, x_h = sim_l, sim_h

        offset = self.get_offset_lp(x_l, x_h, sim_l, sim_h)
        return self.sample(feat2sample, offset)

    def get_offset_lp(self, x_l, x_h, sim_l, sim_h):
        if hasattr(self, 'direct_scale'):
            offset = (self.offset(sim_l) + F.pixel_unshuffle(self.offset_h(sim_h), self.s)) * (self.direct_scale(x_l) + F.pixel_unshuffle(self.direct_scale_h(x_h), self.s)).sigmoid() + self.init_pos
        else:
            offset = (self.offset(x_l) + F.pixel_unshuffle(self.offset_h(x_h), self.s)) * 0.25 + self._init_pos
        return offset
    
    def get_offset(self, x_l, x_h): # Abandoned
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(x_l, x_h)

class ConvMLP(nn.Module):
    '''MLP using 1x1 convs that keeps spatial dims'''
    def __init__(self, c1, c_=None, c2=None, act=nn.ReLU, norm=None, bias=True, drop=0.):
        super().__init__()
        c2 = c2 or c1
        c_ = c_ or c1
        if isinstance(bias, tuple) is False:
            bias = (bias, bias)
        
        self.fc1 = nn.Conv2d(c1, c_, 1, bias=bias[0])
        self.norm = LayerNorm2d(c_)
        self.act = act()
        self.drop = DropPath(drop)
        self.fc2 = nn.Conv2d(c_, c2, 1, bias=bias[1])

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.norm(self.fc1(x)))))
        

class tokenMixer(nn.Module):
    def __init__(self, c, k1=3, k2=11, r=0.15):
        super().__init__()
        c_ = int(c*r)
        self.conv1 = DWConv(c_, c_, k1)
        self.conv2 = DWConv(c_, c_, (1, k2))
        self.conv3 = DWConv(c_, c_, (k2, 1))
        self.split_index = (c - 3*c_, c_, c_, c_)
    
    def forward(self, x):
        x0, x1, x2, x3 = torch.split(x, self.split_index, dim=1)
        return torch.cat([x0, self.conv1(x1), self.conv2(x2), self.conv3(x3)], dim=1)

class InceptionNeXtBlock(nn.Module):
    '''http://arxiv.org/abs/2303.16900'''

    def __init__(self, c, mlp_r=4, shortcut=True, ls_init_value=1e-6):
        super().__init__()
        self.tokenMixer = tokenMixer(c)
        self.norm = nn.BatchNorm2d(c)
        self.feedback = ConvMLP(c, int(mlp_r*c))
        self.shortcut = shortcut
        self.gamma = nn.Parameter(ls_init_value * torch.ones(c, requires_grad=True)) if ls_init_value else None

    def forward(self, x:torch.Tensor):
        y = self.tokenMixer(x)
        y = self.norm(y)
        # print(f"INXB:y={y.shape}")
        y = self.feedback(y)
        if self.gamma is not None:
            y = y.mul(self.gamma.reshape(1, -1, 1, 1))
        return y + x if self.shortcut else y

class C2INXB(C2fPSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(InceptionNeXtBlock(self.c) for _ in range(n)))

class Unfold(nn.Module):
    def __init__(self, k=3):
        super().__init__()       
        self.k = k
        w = torch.eye(k**2)
        w = w.reshape(k**2, 1, k, k)
        self.w = nn.Parameter(w, requires_grad=False)  
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.w, stride=1, padding=self.k//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        w = torch.eye(k**2)
        w = w.reshape(k**2, 1, k, k)
        self.w = nn.Parameter(w, requires_grad=False)
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.w, stride=1, padding=self.k//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, c, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.s = c ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            if refine_attention:
                self.stoken_refine = Attention(c, num_heads=num_heads, attn_ratio=1)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(c, c, 1, 1, 0),
                    nn.Conv2d(c, c, 5, 1, 2, groups=c),
                    nn.Conv2d(c, c, 1, 1, 0)
                )
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        hh, ww = H//h, W//w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).contiguous().reshape(B, hh*ww, h*w, C)
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.s # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).contiguous().reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).contiguous().reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)

        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.reshape(B, C, hh*ww).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, hh, ww)
            else:
                stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.flatten(2).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)

class StokenAttentionLayer(nn.Module):
    def __init__(self, c, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., drop=0.,  drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.conv = DWConv(c, c, 3)
                                        
        self.norm1 = LayerNorm2d(c)
        self.attn = StokenAttention(c, stoken_size=stoken_size, n_iter=n_iter, num_heads=num_heads)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(c)
        self.mlp2 = ConvMLP(c1=c, c_=int(c * mlp_ratio), c2=c, act=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, c, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, c, 1, 1),requires_grad=True)
        
    def forward(self, x):
        x = x + self.conv(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x

class C2PSSA(C2PSA):
    def __init__(self, c1, c2, n=1, layerscale=True, e=0.5, stoken_size=1, n_iter=1):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(StokenAttentionLayer(self.c, n_iter=n_iter, stoken_size=(stoken_size, stoken_size), num_heads=self.c//64, layerscale=layerscale) for _ in range(n)))

class LSKblock(nn.Module):
    ''''http://arxiv.org/abs/2303.09030'''

    def __init__(self, c):
        super().__init__()
        self.cv0 = Conv(c, c, 5, p=2, g=c)
        self.cvsp = Conv(c, c, 7, s=1, p=9, g=c, d=3)
        self.cv1 = Conv(c, c//2, 1)
        self.cv2 = Conv(c, c//2, 1)
        self.cvsq = Conv(3, 2, 7, p=3)
        self.cv3 = Conv(c//2, c, 1)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x1 = self.cv0(x)
        x2 = self.cvsp(x1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)

        y = torch.cat([x1, x2], dim=1)
        y_avg = y.mean(dim=1, keepdim=True)
        y_max, _ = y.max(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)
        y = torch.cat([y_avg, y_max, y_std], dim=1)
        y = self.cvsq(y).sigmoid()

        y = x1 * y[:,0,:,:].unsqueeze(1) + x2 * y[:,1,:,:].unsqueeze(1)
        y = self.cv3(y)
        return self.bn(x * y)
    
class C2fLSK(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(LSKblock(self.c) for _ in range(n))