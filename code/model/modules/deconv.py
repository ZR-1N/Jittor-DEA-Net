import math
import jittor as jt
from jittor import nn

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight # Shape: [Co, Ci, H, W]
        Co, Ci, H, W = conv_weight.shape
        
        # 展平最后两个维度: [Co, Ci, 9]
        conv_weight_flat = conv_weight.reshape(Co, Ci, -1)
        
        # 创建新的权重容器
        conv_weight_cd = jt.zeros((Co, Ci, 3 * 3))
        
        # 复制原有权重
        conv_weight_cd[:, :, :] = conv_weight_flat[:, :, :]
        
        # 核心逻辑：中心点 = 原中心点 - 所有点之和
        # index 4 是 3x3 矩阵的中心 (0-8)
        conv_weight_cd[:, :, 4] = conv_weight_flat[:, :, 4] - jt.sum(conv_weight_flat, dim=2)
        
        # 恢复形状: [Co, Ci, H, W]
        conv_weight_cd = conv_weight_cd.reshape(Co, Ci, H, W)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        Co, Ci, H, W = conv_weight.shape
        conv_weight_flat = conv_weight.reshape(Co, Ci, -1)
        
        # 角度差分逻辑：原权重 - theta * 重排后的权重
        # 索引 [3, 0, 1, 6, 4, 2, 7, 8, 5] 对应旋转或重新映射的邻域
        indices = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        conv_weight_ad = conv_weight_flat - self.theta * conv_weight_flat[:, :, indices]
        
        conv_weight_ad = conv_weight_ad.reshape(Co, Ci, H, W)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_rd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def execute(self, x): # PyTorch forward -> Jittor execute
        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            Co, Ci, H, W = conv_weight.shape
            
            # 创建 5x5 的容器 (padding=2 暗示了 effective kernel size 变大了)
            conv_weight_rd = jt.zeros((Co, Ci, 5 * 5))
            conv_weight_flat = conv_weight.reshape(Co, Ci, -1)
            
            # 复杂的索引映射，用于模拟径向差分
            indices_1 = [0, 2, 4, 10, 14, 20, 22, 24]
            # PyTorch: conv_weight[:, :, 1:] 取的是除第一个元素外的所有 (假设 input 是 3x3 展平后的 9 个)
            # 这里需要注意原逻辑：Conv2d_rd 的 kernel_size 是 3，所以 flat 长度是 9
            # indices_1 有 8 个位置，正好对应 flat[1:] (即索引 1-8，共8个)
            
            conv_weight_rd[:, :, indices_1] = conv_weight_flat[:, :, 1:]
            
            indices_2 = [6, 7, 8, 11, 13, 16, 17, 18]
            conv_weight_rd[:, :, indices_2] = -conv_weight_flat[:, :, 1:] * self.theta
            
            # 索引 12 是 5x5 的中心
            conv_weight_rd[:, :, 12] = conv_weight_flat[:, :, 0] * (1 - self.theta)
            
            conv_weight_rd = conv_weight_rd.reshape(Co, Ci, 5, 5)
            
            # 使用手动构造的权重进行卷积
            out_diff = nn.conv2d(x, conv_weight_rd, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__() 
        # 注意：这里用的是 Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight # Shape: [Co, Ci, K]
        Co, Ci, K = conv_weight.shape # K=3
        
        conv_weight_hd = jt.zeros((Co, Ci, 3 * 3))
        
        # 将 1D 权重映射到 2D 矩阵的特定列
        # [0, 3, 6] 是第一列
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        # [2, 5, 8] 是第三列
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        
        # 此时第二列 [1, 4, 7] 保持为 0，这实际上是一个水平 Sobel 算子的变体
        
        conv_weight_hd = conv_weight_hd.reshape(Co, Ci, 3, 3)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        Co, Ci, K = conv_weight.shape
        
        conv_weight_vd = jt.zeros((Co, Ci, 3 * 3))
        
        # 垂直方向差分
        # [0, 1, 2] 第一行
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        # [6, 7, 8] 第三行
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        
        conv_weight_vd = conv_weight_vd.reshape(Co, Ci, 3, 3)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__() 
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def execute(self, x): # PyTorch forward -> Jittor execute
        # 获取所有分量的等效权重和偏置
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        # 核心步骤：直接在参数空间进行相加
        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        
        # 使用合并后的权重进行一次卷积
        # Jittor conv2d 参数：input, weight, bias=None, stride=1, padding=0...
        res = nn.conv2d(x, w, b, stride=1, padding=1, groups=1)

        return res