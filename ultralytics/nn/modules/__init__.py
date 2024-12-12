# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    SEAttention,
    MCA,
    MCWA,
    pMCA,
    LSKMCA,
    MSSA,
    # C2fMCA,
    C3kAttn,
    C3k2Attn,
    C2fAttnELAN4,
    LowFAM,
    LowFSAM,
    LowIFM,
    LowLKSIFM,
    StarLowIFM,
    Split,
    HighFAM,
    HighFSAM,
    HighIFM,
    INXBHighIFM,
    ConvHighIFM,
    ConvHighLKSIFM,
    StarHighIFM,
    LowLAF,
    HiLAF,
    Inject,
    CARAFE,
    FreqFusion,
    C2INXB,
    C2PSSA,
    LSKblock,
    C2fLSK,
    Star,
    C2fS,
    C2fSAttn,
    S2f,
    S2fMCA,
    FMF,
    WTBottleneck,
    WTC2f,
    WTEEC2f,
    WTCC2f,
    DetailEnhancement,
    MogaBlock,
    ConvMogaPB,
    ConvMogaSB,
    DySample,
    CGAFusion,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    WTConv,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect, ARBDetect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "ARBDetect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SEAttention",
    "MCA",
    "pMCA",
    "LSKMCA",
    "MSSA",
    # "C2fAttn",
    "C3kAttn",
    "C3k2Attn",
    "C2fAttnELAN4",
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
    "ConvHighLKSIFM",
    "StarHighIFM",
    "LowLAF",
    "HiLAF",
    "Inject",
    "CARAFE",
    "FreqFusion",
    "C2INXB",
    "C2PSSA",
    "LSKblock",
    "C2fLSK",
    "Star",
    "C2fS",
    "C2fSAttn",
    "S2f",
    "S2fMCA",
    "FMF",
    "WTConv",
    "WTBottleneck",
    "WTC2f",
    "WTEEC2f",
    "WTCC2f",
    "DetailEnhancement",
    "MogaBlock",
    "ConvMogaPB",
    "ConvMogaSB",
    "DySample",
    "CGAFusion",
)
