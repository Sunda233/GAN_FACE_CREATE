"""
提供支持的函数
"""
import torch
from torch import nn


# functions to generate random data，生成随机数据

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

# modified from https://github.com/pytorch/vision/issues/720
# 生成视图，作用是将尺寸为 (218, 178, 3) 的三维图像张量转换为尺寸为 (218 * 178*3) 的一维张量；
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)