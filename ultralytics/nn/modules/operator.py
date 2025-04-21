"""Operator modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from torch import Tensor

__all__ = (
    "StdPool2d",
)

class StdPool2d(nn.Module):
    """
    标准差池化层 (Standard Deviation Pooling)
    
    参数:
        k: 池化窗口的大小
        s: 滑动步长，默认与kernel_size相同
        p: 填充大小，默认为0
        d: 空洞卷积的系数，默认为1
        return_sqrt: 是否返回标准差(True)或方差(False)
        unbiased: 是否使用无偏估计 (n-1) 而不是 (n)，默认为True以匹配torch.std()
    """
    
    def __init__(self, k, s=None, p=0, d=1, return_sqrt=True, unbiased=True):
        super(StdPool2d, self).__init__()
        self.k = k if isinstance(k, tuple) else (k, k)
        self.s = s if s is not None else self.k
        self.s = self.s if isinstance(self.s, tuple) else (self.s, self.s)
        self.p = p if isinstance(p, tuple) else (p, p)
        self.d = d if isinstance(d, tuple) else (d, d)
        self.return_sqrt = return_sqrt
        self.unbiased = unbiased
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 使用unfold获取所有窗口内的值
        unfolded = F.unfold(
            x,
            kernel_size=self.k,
            dilation=self.d,
            padding=self.p,
            stride=self.s
        )
        
        # 正确计算输出的尺寸
        # 公式: output_size = (input_size + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1
        out_h = int((H + 2 * self.p[0] - self.d[0] * (self.k[0] - 1) - 1) / self.s[0] + 1)
        out_w = int((W + 2 * self.p[1] - self.d[1] * (self.k[1] - 1) - 1) / self.s[1] + 1)
        
        # 重新排列张量形状以便于计算
        windows = rearrange(
            unfolded, 
            'b (c k) l -> b c k l', 
            c=C, 
            k=self.k[0] * self.k[1]
        )
        
        # 计算每个窗口的均值
        means = reduce(windows, 'b c k l -> b c 1 l', 'mean')
        
        if self.unbiased:
            # 无偏估计，使用 n-1 作为分母
            n = self.k[0] * self.k[1]
            variances = reduce((windows - means)**2, 'b c k l -> b c l', 'sum') / (n - 1)
        else:
            # 有偏估计，使用 n 作为分母
            variances = reduce((windows - means)**2, 'b c k l -> b c l', 'mean')
        
        # 如果需要，计算标准差
        result = torch.sqrt(variances) if self.return_sqrt else variances
        
        # 重塑结果为常规的输出格式 [B, C, H_out, W_out]
        return result.view(B, C, out_h, out_w)
    
    @staticmethod
    def _test():
        # 创建一个4x4的输入张量
        input_tensor = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度 [1, 1, 4, 4]
        input_tensor = torch.rand((1, 1, 4, 4))

        # 测试常规用例
        print("=== 测试 kernel_size=2, stride=2 ===")
        std_pool = StdPool2d(k=2, s=2, p=0)
        output = std_pool(input_tensor)
        print(f"输入尺寸: {input_tensor.shape}")
        print(f"输出尺寸: {output.shape}")
        print("输入张量:")
        print(input_tensor.squeeze())
        print("输出张量:")
        print(output.squeeze())

        # 测试尺寸保持的情况
        print("\n=== 测试 kernel_size=3, stride=1, padding=1 ===")
        std_pool2 = StdPool2d(k=3, s=1, p=1)
        output2 = std_pool2(input_tensor)
        print(f"输入尺寸: {input_tensor.shape}")
        print(f"输出尺寸: {output2.shape}")
        print("输入张量:")
        print(input_tensor.squeeze())
        print("输出张量:")
        print(output2.squeeze())

        return output


if __name__ == "__main__":
    StdPool2d._test()