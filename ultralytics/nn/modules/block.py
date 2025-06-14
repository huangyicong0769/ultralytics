# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, WTConv, ConvTranspose, InceptionDWConv2d, SpatialAttention, PConv, SCConv
from .transformer import TransformerBlock, MLPBlock, LayerNorm2d, MLP, DropPath
from .utils import normal_init, constant_init, resize, hamming2D, compute_similarity, carafe, spatial_selective
from .operator import StdPool2d

from math import log2, log
from typing import Literal
from timm.layers import trunc_normal_
from einops import rearrange

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
    "TorchVision",
    "SEAttention",
    "MCA",
    "MCWA",
    "pMCA",
    "MSSA",
    "LSKMCA",
    "C2fMCA",
    "C3kMCA",
    "C3k2MCA",
    "C2fMCAELAN4",
    "LowFAM",
    "LowFSAM",
    "LowIFM",
    "LowLKSIFM",
    "StarLowIFM",
    "Split",
    "HighFAM",
    "HighFSAM",
    "HighIFM",
    "INXBHighIFM",
    "ConvHighIFM",
    "ConvHighLSKIFM",
    "StarHighIFM",
    "LowLAF",
    "HiLAF",
    "Inject",
    "CARAFE"
    "FreqFusion",
    "C2INXB",
    "C2PSSA",
    "LSKblock",
    "C2fLSK",
    "Star",
    "S2f",
    "S2fMCA",
    "FMF",
    "WTC2f",
    "WTEEC2f",
    "WTCC2f",
    "DetailEnhancement",
    "MogaBlock",
    "ConvMogaPB",
    "ConvMogaSB",
    "DySample",
    "CGAFusion",
    "MCAM",
    "EUCB",
    "DFF",
    "CAFF",
    "EFF",
    "CAFM",
    "CAInject",
    "SAFMNPP",
    "SCAM",
    "AFE",
    "C2f_p",
    "EffC2",
    "EffC2f",
    "LAE",
    "LGAE",
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
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initialize the YOLOv8 mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
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
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
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
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (Tuple[int, int, int]): Kernel sizes for max pooling.
        """
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
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
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
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
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
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
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
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c_, k[0], 1)
        self.cv2 = Conv(self.c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
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
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
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
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
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
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
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
        # """
        # Initialize C2f module with attention mechanism.

        # Args:
        #     c1 (int): Input channels.
        #     c2 (int): Output channels.
        #     n (int): Number of Bottleneck blocks.
        #     ec (int): Embedding channels for attention.
        #     nh (int): Number of heads for attention.
        #     gc (int): Guide channels for attention.
        #     shortcut (bool): Whether to use shortcut connections.
        #     g (int): Groups for convolutions.
        #     e (float): Expansion ratio.
        # """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

#     def forward(self, x, guide):
#         """
        # Forward pass through C2f layer with attention.

        # Args:
        #     x (torch.Tensor): Input tensor.
        #     guide (torch.Tensor): Guide tensor for attention.

        # Returns:
        #     (torch.Tensor): Output tensor after processing.
        # """
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x, guide):
#         """
        # Forward pass using split() instead of chunk().

        # Args:
        #     x (torch.Tensor): Input tensor.
        #     guide (torch.Tensor): Guide tensor for attention.

        # Returns:
        #     (torch.Tensor): Output tensor after processing.
        # """
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (Tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
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
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
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
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
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
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
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
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
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
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
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
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

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
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
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
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
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
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
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
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
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
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
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
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
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
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
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
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


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
    def __init__(self, k=3, pool_types=['avg', 'std'], act=nn.Sigmoid()):
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
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
            
        # self.conv = nn.Conv2d(1, 1, kernel_size=(1, k), stride=1, padding=(0, (k-1)//2), bias=False)
        self.conv = Conv(1, 1, (1, k), 1, (0, (k-1)//2), act=False)
        self.act = act
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        return x * self._get_weight(x).expand_as(x)

    def _get_weight(self, x):
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
        out = self.act(out)
        return out

class LSKMCAGate(MCAGate):
    def __init__(self, k1=3, k2=5, pool_types=['avg', 'std']):
        super().__init__(k1, pool_types)
        self.conv2 = DWConv(1, 1, (1, k2), 1, d=k2//2)
        self.cvsq = Conv(3, 2, (1, 7), act=nn.Sigmoid())

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
        x1 = self.conv(out)
        x2 = self.conv2(x1)
        out = 1/2 * sum(spatial_selective([x1, x2], self.cvsq))
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.act(out)
        out = out.expand_as(x)

        return x*out
        

class MCA(nn.Module):
    def __init__(self, c1, res=False, no_spatial=False):
        """Constructs a MCA module.
        Args:
            c1: Number of channels of the input feature maps
            res: Whether to use residual connection
            no_spatial: whether to build channel dimension interactions
        """
        super().__init__()

        self.h_cw = MCAGate()
        self.w_hc = MCAGate()
        self.res = res
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
            return x + 1/3*sum(self.transplit(x)) if self.res else 1/3*sum(self.transplit(x))
        else:
            return x + 1/2*sum(self.transplit(x)) if self.res else 1/2*sum(self.transplit(x))

class MCWA(MCA):
    def __init__(self, c1, no_spatial=False):
        super().__init__(c1, no_spatial)
        assert no_spatial is False
    
    def forward(self, x):
        return x *  self.transplit(x)
    
    def transplit(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw._get_weight(x_h).permute(0, 2, 1, 3).contiguous()
        # print(f"h_cw: x{x.shape}, f{tmp.shape}")

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc._get_weight(x_w).permute(0, 3, 2, 1).contiguous()
        # print(f"w_hc: x{x.shape}, f{tmp.shape}")

        x_c = self.c_hw._get_weight(x)
        # print(f"c_hw: x{x.shape}, f{tmp.shape}")
        return x_h * x_w * x_c

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
    '''Multidimensional Spatial Selective Attention Module
       Useless 
    '''
    def __init__(self, c1, no_spatial=False):
        assert no_spatial == False
        super().__init__(c1, no_spatial)
        self.cv1 = Conv(3, 3, 7)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return self.bn(1/3 * sum(spatial_selective(self.transplit(x), self.cv1)))

class LSKMCA(MCA):
    def __init__(self, c1, no_spatial=False):
        super().__init__(c1, no_spatial)
        self.h_cw = LSKMCAGate()
        self.w_hc = LSKMCAGate()
        if not no_spatial:
            l = 1.5
            g = 1
            temp = round(abs((log2(c1) - g) / l))
            k = temp if temp % 2 else temp - 1

            self.c_hw = LSKMCAGate(k1=k, k2=k+2)

class BottleneckAttn(Bottleneck):
    """Bottleneck with Attention."""

    def __init__(self, c1, c2, attn=nn.Identity, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a MCA bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        if attn is FMF:
            self.attn = attn(c2, c2)
        else:
            self.attn = attn(c2)

    def forward(self, x):
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

    def upsample(self, x:list[torch.Tensor]):
        _, _, H, W = x[1].shape
        if self.sample == 'bilinear':
            x[0] = self.us(x[0], (H, W), mode='bilinear', align_corners=False)
        elif self.sample == 'carafe':
            x[0] = self.us(x[0])
        elif self.sample == 'FreqFusion':
            x[0], x[1] = self.us(x_l=x[0], x_h=x[1])
        else:
            raise NotImplementedError
        return x[0], x[1]

    def align(self, x:list[torch.Tensor])->list[torch.Tensor]:
        _, _, H, W = x[1].shape
        x[0], x[1] = self.upsample(x[:2])
        return x[:2] + [self.ds(t, [H, W]) for t in x[2:]]

    def forward(self, x:list):
        return torch.cat(self.align(x), dim=1)
    
# class LowDFAM(LowFAM):
#     def __init__(self, c, sample = 'bilinear'):
#         super().__init__(c, sample)
#         self.ds = AConv(c//2, c)
        
#     def align(self, x):
#         x[0], x[1] = self.upsample(x[:2])
#         return x[:2] + [self.ds(x[2]), self.ds(self.ds[x[3]])]
    
class LowFSAM(LowFAM):
    def __init__(self, c_u, sample = 'bilinear'):
        super().__init__(c_u, sample)
        self.cvsq = Conv(3, 4, 7, p=3, act=nn.Sigmoid())

    def forward(self, x:list[torch.Tensor]):
        return torch.cat(spatial_selective(self.align(x), self.cvsq), dim=1)
        
class LowIFM(nn.Module):
    """Low-stage information fusion module."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.conv1 = Conv(c1, self.c_, k=1)
        # self.m = nn.Sequential(*(RepVGGDW(c_) for _ in range(n)))
        # self.m = nn.Sequential(*(mBottleneck(c2, c2, e=e) for _ in range(n)))
        self.conv2 = Conv(self.c_, c2, k=1)
        # self.attn = MCA(c_)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('RepVGGDW', RepVGGDW(self.c_))
            self.m.add_module('MCA', MCA(self.c_))

    def forward(self, x):
        # y = self.m(self.conv1(x))
        y = self.conv2(self.m(self.conv1(x)))
        return y
    
    
class LowLKSIFM(LowIFM):
    '''Large kernel selective LIFM'''
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('LSKblock', LSKblock(self.c_))
            self.m.add_module('MCA', MCA(self.c_))

class StarLowIFM(LowIFM):
    def __init__(self, c1, c2, n=1, e=0.25, mlp_r=4):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('Star', Star(self.c_, mlp_r))
            self.m.add_module('MCA', MCA(self.c_))

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

    def align(self, x:list[torch.Tensor])->list[torch.Tensor]:
        _, _, H, W = x[0].shape
        return x[:1] + [self.ds(x_, [H, W]) for x_ in x[1:]]

    def forward(self, x:list[torch.Tensor]):
        return torch.cat(self.align(x), dim=1)

class HighFSAM(HighFAM):
    def __init__(self):
        super().__init__()
        self.cv = Conv(3, 3, 7, p=3, act=nn.Sigmoid)

    def forward(self, x:list[torch.Tensor]):
        return torch.cat(spatial_selective(self.align(x), self.cv), dim=1)

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
        Replace ViT to RepVGGDW
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.conv1 = Conv(c1, self.c_)
        self.conv2 = Conv(self.c_, c2)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('RepVGGDW', RepVGGDW(self.c_))
            self.m.add_module('MCA', MCA(self.c_))

    def forward(self, x):
        # y = self.m(self.conv1(x)) + self.conv3(x)
        y = self.conv2(self.m(self.conv1(x)))
        return y

class INXBHighIFM(ConvHighIFM):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = C2INXB(self.c_, self.c_, n=n, e=1)

class ConvHighLKSIFM(ConvHighIFM):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('LSKblock', LSKblock(self.c_))
            self.m.add_module('MCA', MCA(self.c_))
    
class StarHighIFM(ConvHighIFM):
    def __init__(self, c1, c2, n=1, e=0.25, mlp_r=4):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential()
        for _ in range(n):
            self.m.add_module('Star', Star(self.c_, mlp_r))
            self.m.add_module('MCA', MCA(self.c_))

class HighIFM(ConvHighIFM):
    '''Replace Transformer to Super Token Attention'''
    def __init__(self, c1, c2, n=1, e=0.5, num_head=4):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(StokenAttentionLayer(self.c_, 1, (1, 1), num_head, drop=0.1, layerscale=True) for _ in range(n)))

class LowLAF(nn.Module):
    """Low-stage lightweight adjacent layer fusion module"""

    def __init__(self, c1, c2, c_s, c_l, sample:Literal['bilinear', 'carafe', 'FreqFusion', 'DySample']='bilinear'):
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
        elif self.sample == 'DySample':
            self.us = DySample(c_l)
        else:
            raise NotImplementedError
        self.ds = F.adaptive_avg_pool2d

    def forward(self, x:list[torch.Tensor]):
        _, _, H, W = x[1].shape

        if self.sample == 'bilinear':
            x[0] = self.us(x[0], (H, W), mode='bilinear', align_corners=False)
        elif self.sample in {'carafe', 'DySample'}:
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

    def forward(self, x:list[torch.Tensor])->torch.Tensor:
        _, _, H, W = x[0].shape
        x[1] = self.ds(x[1], [H, W])
        return self.conv(torch.cat(x, 1))

class Inject(nn.Module):
    """Information injection module"""

    def __init__(self, c1, c2, index, n=1, sample:Literal['avgpool', 'bilinear', 'carafe', 'FreqFusion', 'Conv', 'ConvT', 'Same', 'DySample', 'EUCB', 'SAFMNPP']='Same'):
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
            self.cv = Conv(2*c2, c2, act=False)
        elif self.sample == 'Conv':
            self.ds1 = Conv(c2, c2, 3, 2, act=False)
            self.ds2 = Conv(c2, c2, 3, 2, act=False)
        elif self.sample == 'ConvT':
            self.us1 = ConvTranspose(c2, c2, act=False)
            self.us2 = ConvTranspose(c2, c2, act=False)
        elif self.sample == 'DySample':
            self.us1 = DySample(c2)
            self.us2 = DySample(c2)
        elif self.sample == 'EUCB':
            self.us1 = EUCB(c2, c2)
            self.us2 = EUCB(c2, c2)
        elif self.sample == 'SAFMNPP':
            self.us1 = SAFMNPP(c2, c2)
            self.us2 = SAFMNPP(c2, c2)

    def forward(self, x:list[torch.Tensor]):
        x1 = self.conv1(x[0])
        x2 = x[1] if self.index == -1 else x[1][self.index]
        
        x_ = self.act(self.conv2(x2))
        x2 = self.conv3(x2)

        x1, x2, x_ = self.alian(x1, x2, x_)

        y = x2 + x1 * x_
        return self.conv4(y)

    def alian(self, x1, x2, x_):
        _, _, H, W = x1.shape
        if self.sample == 'avgpool':
            x2, x_ = F.adaptive_avg_pool2d(x2, [H, W]), F.adaptive_avg_pool2d(x_, [H, W])
        elif self.sample == 'Conv':
            x2, x_ = self.ds1(x2), self.ds2(x_)
        elif self.sample == 'bilinear':
            x2, x_ = F.interpolate(x2, (H, W), mode='bilinear', align_corners=False), F.interpolate(x_, (H, W), mode='bilinear', align_corners=False)
        elif self.sample in {'carafe', 'ConvT', 'DySample', 'EUCB', 'SAFMNPP'}:
            x2, x_ = self.us1(x2), self.us2(x_)
        elif self.sample == 'FreqFusion':
            x2, x11 = self.us1(x_l=x2, x_h=x1)
            x_, x12 = self.us2(x_l=x_, x_h=x1)
            x1 = self.cv(torch.cat([x11, x12], dim=1))
        else:
            if self.sample != 'Same':
                print(self.sample)
                raise NotImplementedError
        return x1,x2,x_

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

class InceptionNeXtBlock(nn.Module):
    '''http://arxiv.org/abs/2303.16900'''

    def __init__(self, c, mlp_r=4, shortcut=True, ls_init_value=1e-6):
        super().__init__()
        self.tokenMixer = InceptionDWConv2d(c)
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

class C2INXB(C2PSA):
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

    def __init__(self, c, e=0.5):
        super().__init__()
        c_ = int(c * e)
        self.cv0 = Conv(c, c, 5, p=2, g=c)
        self.cvsp = Conv(c, c, 7, s=1, p=9, g=c, d=3)
        self.cv1 = Conv(c, c_, 1)
        self.cv2 = Conv(c, c_, 1)
        self.cvsq = Conv(3, 2, 7, p=3, act=nn.Sigmoid())
        self.cv3 = Conv(c_, c, 1)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x1 = self.cv0(x)
        x2 = self.cvsp(x1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        y = sum(spatial_selective([x1, x2], self.cvsq))
        y = self.cv3(y)
        return self.bn(x * y)
    
class C2fLSK(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(LSKblock(self.c) for _ in range(n))
        
class Star(nn.Module):
    """https://github.com/ma-xu/Rewrite-the-Stars/
    """
    def __init__(self, c, mlp_r=4):
        super().__init__()
        self.cv1 = DWConv(c, c, k=7, act=False)
        self.fc1 = nn.Conv2d(c, c*mlp_r, kernel_size=1)
        self.act = nn.ReLU6()
        self.fc2 = nn.Conv2d(c, c*mlp_r, kernel_size=1)
        self.fc3 = Conv(c* mlp_r, c, k=1, act=False)
        self.cv2 = DWConv(c, c, k=7, act=False)
    
    def forward(self, x):
        res = x
        x = self.cv1(x)
        x1, x2 = self.fc1(x), self.fc2(x)
        x = self.act(x1) * x2
        x = self.fc3(x)
        return res + self.cv2(x)
    
class StarMCA(Star):
    def __init__(self, c, mlp_r=4):
        super().__init__(c, mlp_r)
        self.attn = MCA(c)
        
    def forward(self, x):
        return self.attn(super().forward(x))
    
class C2fS(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.cv2 = DWConv(self.c, c2, 1, act=False)
        self.norm = nn.BatchNorm2d(self.c)
        self.act = nn.ReLU6()
    
    def forward(self, x:torch.Tensor):
        y = [chunk.contiguous() for chunk in self.cv1(x).chunk(2, 1)]
        y.extend(m(y[-1]) for m in self.m)
        rsl = y[0]
        for t in y[1:]:
            rsl = self.norm(rsl * self.act(t.clone()))
        return self.cv2(y[0] + rsl)
    
class C2fSAttn(C2fS):
    def __init__(self, c1, c2, n=1, attn=nn.Identity, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BottleneckAttn(self.c, self.c, attn, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
class S2f(C2f):
    def __init__(self, c1, c2, n=1, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        self.m = nn.ModuleList(Star(self.c) for _ in range(n))
        
class S2fMCA(S2f):
    def __init__(self, c1, c2, n=1, g=1, e=0.5):
        super().__init__(c1, c2, n, g, e)
        self.m = nn.ModuleList(StarMCA(self.c) for _ in range(n))
        
class FMF(nn.Module):
    """Feature Focusing Diffusion Model
    http://arxiv.org/abs/2408.00438
    """

    def __init__(self, c1, c2, k=[1, 3, 5, 7], drop=0.2, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.m = nn.ModuleList(DWConv(c1, self.c, kl) for kl in k)
        self.cv1 = Conv(c1, self.c)
        self.cv2 = Conv(self.c, c2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.cv2(self.cv1(x) + torch.stack([cv(x) for cv in self.m], dim=0).mean(dim=0)))
    
class WTBottleneck(nn.Module):
    def __init__(self, c1, c2, attn=nn.Identity, shortcut=True, wt_levels=1, k=3, e=0.5):
        super().__init__()
        self.shortcut = shortcut and c1==c2
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c)
        self.cv2 = WTConv(self.c, self.c, k, wt_levels=wt_levels)
        self.cv3 = Conv(self.c, c2)
        self.attn = attn(c2)
        
    def forward(self, x):
        H, W = x.shape[2:]
        y = self.attn(self.cv3(self.cv2(self.cv1(x))))
        return x +  y if self.shortcut else y
    
class WTC2f(C2fAttn):
    def __init__(self, c1, c2, n=1, attn=nn.Identity, shortcut=False, wt_levels=1, k=3, e=0.5):
        super().__init__(c1, c2, n, attn, shortcut, 1, e)
        self.m = nn.ModuleList(WTBottleneck(self.c, self.c, attn, shortcut, wt_levels, k, e=1.0) for _ in range(n))
        
class WTEEC2f(C2fAttn):
    def __init__(self, c1, c2, n=1, attn=nn.Identity, shortcut=False, wt_levels=1, k=3, e=0.5):
        super().__init__(c1, c2, n, attn, shortcut, 1, e)
        self.m = nn.ModuleList(WTBottleneck(self.c, self.c, attn, shortcut, wt_levels, k, e=1.0) for _ in range(n))
        self.ee = MEEM(self.c)
        self.cv2 = Conv((3 + n) * self.c, c2)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.ee(y[0]))
        return self.cv2(torch.cat(y, 1))
        
class MEEM(nn.Module):
    """Multi-scale Edge Enhancement Module
    http://arxiv.org/abs/2408.04326
    """
    def __init__(self, c1, n = 3, e=0.5):
        super().__init__()
        self.n = n
        self.c = int(c1 * e)
        
        self.cv1 = Conv(c1, self.c, 1, act=nn.Sigmoid())
        self.m = nn.ModuleList(nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1),
                                             Conv(self.c, self.c, 1, act=nn.Sigmoid()),
                                             EdgeEnhancer(self.c)) for _ in range(n))
        self.cv3 = Conv(self.c * (n + 1), c1, 1)
    
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))

class EdgeEnhancer(nn.Module):
    """http://arxiv.org/abs/2408.04326"""
    def __init__(self, c):
        super().__init__()
        self.cv = Conv(c, c, 1, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        return x + self.cv(x - self.pool(x))
    
class DetailEnhancement(nn.Module):
    """http://arxiv.org/abs/2408.04326"""
    def __init__(self, c1, c2, ci, e=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(ci, self.c, 3)
        self.img_er = MEEM(self.c)
        self.feature_upsample = nn.Sequential(
            Conv(c1 * 2, c1, 3),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            Conv(c1, c1, 3),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            Conv(c1, self.c, 3)
        )
        self.cv2 = Conv(self.c * 2, c2, 3)
    
    def forward(self, x:list[torch.Tensor]):
        img, feature, b_feature = x # [3, 640, 640], [c1, 160, 160], [c1, 160, 160]
        
        feature = torch.cat([feature, b_feature], dim = 1) # -> [2*c1]
        feature = self.feature_upsample(feature) # [2*c1, 160, 160] -> [c, 640, 640]

        img_feature = self.cv1(img) # [3, 640, 640] -> [c, 640, 640]
        img_feature = self.img_er(img_feature) + img_feature # -> [c, 640, 640]

        return self.cv2(torch.cat([feature, img_feature], dim = 1)) # [2*c, 640, 640] -> [c2, 640, 640]
    
class WTCC2f(C2fAttn):
    def __init__(self, c1, c2, n=1, attn=nn.Identity, shortcut=False, wt_levels=1, k=3, e=0.5):
        super().__init__(c1, c2, n, attn, shortcut, 1, e)
        self.wt = WTBottleneck(self.c, self.c, attn, shortcut, wt_levels, k, e)
        self.ee = MEEM(self.c)
        self.cv2 = Conv((4 + n) * self.c, c2)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.wt(y[0]))
        y.append(self.ee(y[0]))
        return self.cv2(torch.cat(y, 1))

class ElementScale(nn.Module):
    def __init__(self, c, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, c, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        c (int): The feature dimension. Same as `MultiheadAttention`.
        c_f (int): The hidden dimension of FFNs.
        k (int): The depth-wise conv kernel size as the depth-wise convolution. Defaults to 3.
        act: The type of activation. Defaults to nn.GELU().
        drop (float, optional): Probability of an element to be zeroed in FFN. Default 0.0.
    """

    def __init__(self, c, c_f, k=3, act=nn.GELU(), drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.c = c
        self.c_f = c_f

        self.fc1 = Conv(c, c_f, act=False)
        self.dwconv = DWConv(c_f, c_f, k, act=act)
        self.fc2 = nn.Conv2d(
            in_channels=c_f,
            out_channels=c,
            kernel_size=1)
        self.fc2 = Conv(c_f, c, act=False)
        self.drop = nn.Dropout(drop)
        self.decompose = Conv(c_f, 1, act=act)
        self.sigma = ElementScale(self.c_f, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        return x + self.sigma(x - self.decompose(x)) # x_d: [B, C, H, W] -> [B, 1, H, W]

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        c (int): Number of input channels.
        dw_d (list): Dilations of three DWConv layers.
        c_split (list): The raletive ratio of three splited channels.
    """
    def __init__(self, c, dw_d=[1, 2, 3,], c_split=[1, 3, 4,],):
        super(MultiOrderDWConv, self).__init__()

        self.split_r = [i / sum(c_split) for i in c_split]
        self.c1 = int(self.split_r[1] * c)
        self.c2 = int(self.split_r[2] * c)
        self.c0 = c - self.c1 - self.c2
        self.c = c
        
        assert len(dw_d) == len(c_split) == 3
        assert 1 <= min(dw_d) and max(dw_d) <= 3
        assert c % sum(c_split) == 0

        self.DW_conv0 = DWConv(self.c, self.c, 5, d=dw_d[0], act=False) # basic DW conv
        self.DW_conv1 = DWConv(self.c1, self.c1, 5, d=dw_d[1], act=False) # DW conv 1
        self.DW_conv2 = DWConv(self.c2, self.c2, 7, d=dw_d[2], act=False) # DW conv 2
        self.PW_conv = Conv(self.c, self.c, act=False) # a channel convolution

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.c0: self.c0+self.c1, ...])
        x_2 = self.DW_conv2(x_0[:, self.c-self.c2:, ...])
        x = torch.cat([x_0[:, :self.c0, ...], x_1, x_2], dim=1)
        return self.PW_conv(x)

class MultiOrderGatedAggregation(nn.Module):
    '''Spatial Block with Multi-order Gated Aggregation.[https://arxiv.org/pdf/2211.03295]
       https://github.com/Westlake-AI/MogaNet

    Args:
        c (int): Number of input channels.
        attn_dw_d(list): Dilations of three DWConv layers.
        attn_c_split (list): The raletive ratio of splited channels.
        attn_act: The activation for Spatial Block. Defaults to nn.SiLU().
    '''
    def __init__(self, c, attn_dw_d=[1, 2, 3], attn_c_split=[1, 3, 4], attn_act=nn.SiLU(), attn_force_fp32=False,):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = c
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1)
        self.gate = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1)
        self.value = MultiOrderDWConv(c=c, dw_d=attn_dw_d, c_split=attn_c_split,)
        self.proj_2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1)

        # activation for gating and value
        self.act_value = attn_act
        self.act_gate = attn_act

        # decompose
        self.sigma = ElementScale(c, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1) # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x = x + self.sigma(x - x_d)
        return self.act_value(x)

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        x = self.feat_decompose(x) # proj 1x1
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        return x + shortcut
    
class MogaBlock(nn.Module):
    """A block of MogaNet.[https://arxiv.org/pdf/2211.03295]
       https://github.com/Westlake-AI/MogaNet

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self, c, ffn_r=4., drop=0., droppath=0., act=nn.GELU(), init_value=1e-5, attn_dw_d=[1, 2, 3], attn_c_split=[1, 3, 4], attn_act=nn.SiLU(), attn_force_fp32=False,):
        super(MogaBlock, self).__init__()
        self.c2 = c
        self.norm1 = nn.BatchNorm2d(c)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(c, attn_dw_d=attn_dw_d, attn_c_split=attn_c_split, attn_act=attn_act, attn_force_fp32=attn_force_fp32,)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(c)

        # channel MLP
        c_e = int(c * ffn_r)
        self.mlp = ChannelAggregationFFN(c=c, c_f=c_e, act=act, drop=drop,) # DWConv + Channel Aggregation FFN

        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, c, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x

class ConvMogaPB(nn.Module):
    def __init__(self, c1, c2, n=1, ffn_r=4, e=1., drop=0.):
        super().__init__()
        self.c = int(c2 * e) // 2
        self.conv = C2f(self.c, self.c, n, shortcut=True)
        self.moga = MogaBlock(self.c, ffn_r, drop)
        self.pwconv1 = nn.Conv2d(c1, 2 * self.c, 1, padding='same')
        self.pwconv2 = nn.Conv2d(2 * self.c, c2, 1, padding='same')
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x1, x2 = torch.split(self.pwconv1(x), [self.c, self.c], dim=1)
        return self.pwconv2(torch.cat([self.conv(x1), self.moga(x2)], dim=1))
    
class ConvMogaSB(nn.Module):
    def __init__(self, c1, c2, n=1, ffn_r=4, e=0.5, drop=0.):
        super().__init__()
        self.c = int(c2 * e)
        self.conv = C2f(self.c, self.c, n, shortcut=True)
        self.moga = MogaBlock(self.c, ffn_r, drop)
        self.pwconv1 = nn.Conv2d(c1, self.c, 1, padding='same')
        self.pwconv2 = nn.Conv2d(2 * self.c, c2, 1, padding='same')
        
    def forward(self, x):
        x1 = self.conv(self.pwconv1(x))
        x2 = self.moga(x1)
        return self.pwconv2(torch.cat([x1, x2], dim=1))

class DySample(nn.Module):
    def __init__(self, c1, s=2, style='lp', g=4, dyscope=True):
        super().__init__()
        self.s = s
        self.style = style
        self.g = g
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert c1 >= s ** 2 and c1 % s ** 2 == 0
        assert c1 >= g and c1 % g == 0

        if style == 'pl':
            c1 = c1 // s ** 2
            c2 = 2 * g
        else:
            c2 = 2 * g * s ** 2

        self.offset = nn.Conv2d(c1, c2, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(c1, c2, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.s + 1) / 2, (self.s - 1) / 2 + 1) / self.s
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.g, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.s).view(B, 2, -1, self.s * H, self.s * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.g, -1, H, W), coords, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, self.s * H, self.s * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.s)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.s) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.s) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class CGAFusion(nn.Module):
    """ingle image dehazing based on detail enhanced convolution and content-guided attention
    [https://github.com/cecret3350/DEA-Net/tree/main]
    """
    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

        def forward(self, x):
            x_avg = torch.mean(x, dim=1, keepdim=True)
            x_max, _ = torch.max(x, dim=1, keepdim=True)
            x2 = torch.cat([x_avg, x_max], dim=1)
            sattn = self.sa(x2)
            return sattn

    class ChannelAttention(nn.Module):
        def __init__(self, dim, reduction=8):
            super().__init__()
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.ca = nn.Sequential(
                nn.Conv2d(dim, dim // reduction, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // reduction, dim, 1, bias=True),
            )

        def forward(self, x):
            x_gap = self.gap(x)
            cattn = self.ca(x_gap)
            return cattn

    class PixelAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, pattn1):
            B, C, H, W = x.shape
            x = x.unsqueeze(dim=2)  # B, C, 1, H, W
            pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
            x2 = torch.cat([x, pattn1], dim=2).contiguous().view([B, -1, H, W])  # B, C, H, W
            pattn2 = self.pa2(x2)
            pattn2 = self.sigmoid(pattn2)
            return pattn2
        
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = self.SpatialAttention()
        self.ca = self.ChannelAttention(dim, reduction)
        self.pa = self.PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:list[torch.Tensor])->torch.Tensor:
        x, y = x
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

class MCAM(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(MCAM, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g_sar = conv_nd(in_channels=self.in_channels,out_channels=self.inter_channels,kernel_size=1,stride=1,padding=0)

        self.g_opt = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta_sar = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.theta_opt = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi_sar = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.phi_opt = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        if sub_sample:
            self.g_sar = nn.Sequential(self.g_sar, max_pool_layer)
            self.g_opt = nn.Sequential(self.g_opt, max_pool_layer)
            self.phi_sar = nn.Sequential(self.phi_sar, max_pool_layer)
            self.phi_opt = nn.Sequential(self.phi_opt, max_pool_layer)

    def forward(self, x):
        sar, opt = x

        batch_size = sar.size(0)

        g_x = self.g_sar(sar).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta_sar(sar).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi_sar(sar).view(batch_size, self.inter_channels, -1)

        f_x = torch.matmul(theta_x, phi_x)
        f_div_C_x = F.softmax(f_x, dim=-1)

        g_y = self.g_opt(opt).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_y = g_y.permute(0, 2, 1)

        theta_y = self.theta_opt(opt).view(batch_size, self.inter_channels, -1)
        theta_y = theta_y.permute(0, 2, 1)

        phi_y = self.phi_opt(opt).view(batch_size, self.inter_channels, -1)

        f_y = torch.matmul(theta_y, phi_y)
        f_div_C_y = F.softmax(f_y, dim=-1)
        y = torch.einsum('ijk,ijk->ijk', [f_div_C_x, f_div_C_y])
        y_x = torch.matmul(y, g_x)
        y_y = torch.matmul(y, g_y)
        y = y_x * y_y
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *sar.size()[2:])
        y = self.W(y)
        return y

class EUCB(nn.Module):
#   Efficient up-convolution block (EUCB)
    def __init__(self, c1, c2, k=3, s=1, act=nn.ReLU(inplace=True)):
        super(EUCB, self).__init__()

        self.in_channels = c1
        self.out_channels = c2
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=k, stride=s,
                      padding=k // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x
    
class DFF(nn.Module):
    '''[https://arxiv.org/abs/2403.10674]
    '''
    def __init__(self, dim):
        super().__init__()

        self.conv_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x):
        x, skip = x
        output = torch.cat([x, skip], dim=1)
        
        att = self.conv_atten(output)
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output
    
class CAFF(DFF):
    def __init__(self, dim):
        super().__init__(dim)
        self.conv_atten = MCA(dim)

class EFF(nn.Module):
    """SUnet: A multi-organ segmentation network based on multiple attention
    [https://www.sciencedirect.com/science/article/abs/pii/S0010482523010612]
    """
    class Efficient_Attention_Gate(nn.Module):
        def __init__(self, F_g, F_l, F_int, num_groups=32):
            super().__init__()
            self.num_groups = num_groups
            self.grouped_conv_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
                nn.BatchNorm2d(F_int),
                nn.ReLU(inplace=True)
            )

            self.grouped_conv_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
                nn.BatchNorm2d(F_int),
                nn.ReLU(inplace=True)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.grouped_conv_g(g)
            x1 = self.grouped_conv_x(x)
            psi = self.psi(self.relu(x1 + g1))
            out = x * psi
            out += x

            return out

    class EfficientChannelAttention(nn.Module):
        def __init__(self, channels, gamma=2, b=1):
            super().__init__()

            # 设计自适应卷积核，便于后续做1*1卷积
            kernel_size = int(abs((log(channels, 2) + b) / gamma))
            kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

            # 全局平局池化
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

            # 基于1*1卷积学习通道之间的信息
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

            # 激活函数
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
            v = self.avg_pool(x)

            # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
            v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            # 最终，经过sigmoid 激活函数处理
            v = self.sigmoid(v)
            return v

    def __init__(self, in_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.EAG = self.Efficient_Attention_Gate(in_dim, in_dim, in_dim)
        else:
            self.EAG = nn.Identity()
        self.ECA = self.EfficientChannelAttention(in_dim*2)
        self.SA = SpatialAttention()
        self.conv = DWConv(2 * in_dim, in_dim)

    def forward(self, x):
        x, skip = x
        # print(f"pre:{x.shape}, {skip.shape}")
        if not self.is_bottom:
            EAG_skip = self.EAG(x, skip)
            x = torch.cat((EAG_skip, x), dim=1)
            # x = EAG_skip + x
        else:
            x = self.EAG(x)
        x = self.ECA(x) * x
        x = self.SA(x) * x
        x = self.conv(x)
        # print(f"aft:{x.shape}")
        return x
    
class CAFM(nn.Module):  
    """Cross Attention Fusion Module
    Attention Multihop Graph and Multiscale Convolutional Fusion Network for Hyperspectral Image Classification
    [https://ieeexplore.ieee.org/document/10098209]
    """
    def __init__(self, c, e=0.5):
        super(CAFM, self).__init__()
        c_ = int(c * e)

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(c, c_, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(c, c_, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(c, c_, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(c, c_, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(c_, c, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(c_, c, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(c_, c, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(c_, c, 1, stride=1, padding=0)

        self.cv = Conv(2*c, c)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        f1, f2 = x
        f1, f2 = self._forward(f1, f2)
        return f1 + f2

    def _forward(self, f1, f2):
        b, c, h, w = f1.shape
        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2
        return f1.view([b, c, h, w]), f2.view([b, c, h, w])
    
class CAInject(Inject):
    def __init__(self, c1, c2, index, n=1, sample = 'Same'):
        super().__init__(c1, c2, index, n, sample)
        self.ca = CAFM(c2)
        
    def forward(self, x):
        x1 = x[0]
        x2 = x[1] if self.index == -1 else x[1][self.index]
        
        x1, x2, _ = self.alian(x1, x2, x2)
        x1, x2 = self.ca._forward(x1, x2)
        
        return self.conv4(x1 * self.act(x2))

class SAFMNPP(nn.Module):
    """[https://github.com/sunny2109/SAFMN/blob/main/NTIRE2024_ESR/models/team23_safmnpp.py]"""
    class SimpleSAFM(nn.Module):
        def __init__(self, dim, ratio=4):
            super().__init__()
            self.dim = dim
            self.chunk_dim = dim // ratio

            self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
            self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)
            self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
            self.act = nn.GELU()

        def forward(self, x):
            h, w = x.size()[-2:]

            x0, x1 = self.proj(x).split([self.chunk_dim, self.dim-self.chunk_dim], dim=1)

            x2 = F.adaptive_max_pool2d(x0, (h//8, w//8))
            x2 = self.dwconv(x2)
            x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
            x2 = self.act(x2) * x0

            x = torch.cat([x1, x2], dim=1)
            x = self.out(self.act(x))
            return x
    
    # Convolutional Channel Mixer
    class CCM(nn.Module):
        def __init__(self, dim, ffn_scale, use_se=False):
            super().__init__()
            self.use_se = use_se
            hidden_dim = int(dim*ffn_scale)

            self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.conv2(x)
            return x

    class AttBlock(nn.Module):
        def __init__(self, dim, ffn_scale, use_se=False):
            super().__init__()

            self.conv1 = SAFMNPP.SimpleSAFM(dim, ratio=3)
            self.conv2 = SAFMNPP.CCM(dim, ffn_scale, use_se)

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            return out + x

    def __init__(self, c1, dim, n_blocks=1, ffn_scale=2.0, use_se=False, upscaling_factor=2):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(c1, dim, 3, 1, 1, bias=False)

        self.feats = nn.Sequential(*[self.AttBlock(dim, ffn_scale, use_se) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, c1 * upscaling_factor**2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )
        
    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        return self.to_img(x)

class SCAM(nn.Module):
    """[https://github.com/yemu1138178251/FFCA-YOLO/blob/main/models/common.py]
    """
    def __init__(self, in_channels):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, 1), nn.SiLU())
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # avg max: [N, C, 1, 1]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y2:[N, 1, H, W]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat:[N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y

# 自适应特征增强模块
class AFE(nn.Module):
    """[https://arxiv.org/pdf/2407.09379]
    FANET: FEATURE AMPLIFICATION NETWORK FOR SEMANTIC SEGMENTATION IN CLUTTERED BACKGROUND
    """
    class LayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数，初始化为1
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 可学习的偏移参数，初始化为0
            self.eps = eps  # 防止除零的微小值
            self.data_format = data_format  # 数据格式
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError  # 如果数据格式不支持则抛出异常
            self.normalized_shape = (normalized_shape,)

        def forward(self, x):
            if self.data_format == "channels_last":
                # 使用F.layer_norm进行归一化
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                # 手动实现归一化
                u = x.mean(1, keepdim=True)  # 计算均值
                s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
                x = (x - u) / torch.sqrt(s + self.eps)  # 标准化
                x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用缩放和偏移
                return x


    # 特征细化模块，用于特征增强和提取
    class FeatureRefinementModule(nn.Module):
        def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
            super().__init__()

            # 深度可分离卷积层，用于低频和高频特征提取
            self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
            self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
            self.norm1 = AFE.LayerNorm(in_dim, eps=1e-6, data_format="channels_first")  # 用于低频特征的归一化
            self.norm2 = AFE.LayerNorm(in_dim, eps=1e-6, data_format="channels_first")  # 用于高频特征的归一化
            self.act = nn.GELU()  # 激活函数

            # 下采样层，用于减少特征图的尺寸 低频特征通常代表图像中的平滑或全局信息，它们可以通过对输入特征进行下采样、滤波和归一化等操作来提取

            self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                                  groups=in_dim)
            self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)  # 投影层，用于合并低频和高频特征

            self.apply(self._init_weights)  # 初始化权重

        def _init_weights(self, m):
            # 初始化权重的方法
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)  # 使用截断正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0

            elif isinstance(m, (AFE.LayerNorm, nn.LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def forward(self, x):
            B, C, H, W = x.shape
            #首先对输入特征进行下采样（通过卷积操作）。下采样减少了特征图的空间分辨率，保留了图像的粗略结构，这有助于捕捉低频信息。
            dx = self.down(x)  # 下采样特征
            #将下采样的特征图恢复到原始尺寸，近似地保留了原图的低频信息。
            udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)  # 恢复特征图尺寸
            #利用低频特征（udx）与输入特征（x）的乘积进行卷积，进一步提取低频信息。norm1和act操作帮助稳定训练并增强非线性特征。
            lx = self.norm1(self.lconv(self.act(x * udx)))  # 低频特征细化
            hx = self.norm2(self.hconv(self.act(x - udx)))  # 高频特征细化

            out = self.act(self.proj(torch.cat([lx, hx], dim=1)))  # 合并并投影特征

            return out
    
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        # 深度可分离卷积，用于初步特征提取
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, padding=0)  # 将通道数减半
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)  # 恢复通道数

        # 上下文卷积层，用于提取全局上下文信息
        self.ctx_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=4)

        # 归一化层
        self.norm1 = self.LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = self.LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.norm3 = self.LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        # 特征细化模块
        self.enhance = self.FeatureRefinementModule(in_dim=dim // 2, out_dim=dim // 2, down_kernel=3, down_stride=2)

        self.act = nn.GELU()  # 激活函数

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.norm1(self.act(self.dwconv(x)))  # 初步特征增强
        x = self.norm2(self.act(self.proj1(x)))  # 通道数减半后归一化

        ctx = self.norm3(self.act(self.ctx_conv(x)))  # 提取全局上下文信息

        enh_x = self.enhance(x)  # 细化特征
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))  # 合并上下文和细化特征

        return x

class Bottleneck_p(BottleneckAttn):
    def __init__(self, c1, c2, attn=nn.Identity, shortcut=True, e=0.25):
        super().__init__(c1, c2, attn, shortcut, 1, (3, 3), e)
        assert c1 == c2
        self.cv1 = PConv(c1, n=int(1/e))
        self.cv2 = PConv(c2, n=int(1/e))

class C2f_p(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_p(self.c, self.c, shortcut=shortcut, e=1) for _ in range(n))

class SCBottleneck(BottleneckAttn):
    def __init__(self, c1, c2, attn=nn.Identity, shortcut=True, e=(1.0, 0.5), a=0.5, treshold=0.5):
        super().__init__(c1, c2, attn, shortcut, 1, (3, 3), e[0])
        assert c1 == c2
        self.cv1 = SCConv(c1, 3, e[1], a, treshold)
        self.cv2 = SCConv(c1, 3, e[1], a, treshold)

class EffC2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.25):
        super().__init__()
        assert c1 == c2
        self.shortcut = shortcut

        self.m = nn.Sequential(*(SCBottleneck(c1, c1, MCA, shortcut=shortcut, e=e) for _ in range(n)))
        self.cv = Conv(c1, c1, 1)

    def forward(self, x):
        return x + self.cv(self.m(x)) if self.shortcut else self.cv(self.m(x))
    
class EffC2f(EffC2):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.25):
        super().__init__(c1, c2, n, shortcut, e)
        self.m = nn.ModuleList(SCBottleneck(c1, c1, MCA, shortcut=shortcut, e=e) for _ in range(n))
        self.cv = Conv((n + 1)*c1, c2, 1)

    def forward(self, x):
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        return x + self.cv(torch.cat(y, 1)) if self.shortcut else self.cv(torch.cat(y, 1))
    
class LAE(nn.Module):
    '''Light-weight Adaptive Extraction
    https://github.com/VincentYuuuuuu/LSM-YOLO/blob/master/ultralytics/nn/modules/block.py
    '''
    def __init__(self, c, g=16) -> None:
        super().__init__()
        
        self.act = nn.Softmax(dim=-1)
        self.m = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(c, c, k=1)
        )
        
        self.conv = Conv(c, c * 4, k=3, s=2, g=(c // g))
    
    def forward(self, x:Tensor) -> Tensor:
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.m(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.act(att)
        
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x
    
class LGAE(LAE):
    '''Light-weight Gated Adaptive Extraction_i
    '''
    def __init__(self, c, g=16):
        super().__init__(c, g)
        self.m = MCAGate(act=nn.Identity())

class LGAE2(LAE):
    '''Light-weight Gated Adaptive Extraction_i
    '''
    def __init__(self, c, g=16):
        super().__init__(c, g)
        self.m = MCA(c, no_spatial=True)

# class LGAEGate(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.pools = nn.ModuleList([nn.AvgPool2d(3, 1, 1), nn.MaxPool2d(3, 1, 1)])
#         self.weight = nn.Parameter(torch.rand(2), requires_grad=True)
#         # self.weight[-1] = 0.
#         self.conv = Conv(c, c, 1, act=False)

#     def forward(self, x:Tensor)->Tensor:
#         n = len(self.pools)
#         f = torch.stack([pool(x) for pool in self.pools], dim=1) #B N C H W
#         w = self.weight.view(1, n, 1, 1, 1)
#         # w = torch.sigmoid(w)
#         x = torch.sum(f * w, 1)
#         return self.conv(x.squeeze(1))

# class LGAE(LAE):
#     '''Light-weight Gated Adaptive Extraction_ii
#     '''
#     def __init__(self, c, g=16):
#         super().__init__(c, g)
#         self.m = LGAEGate(c)
