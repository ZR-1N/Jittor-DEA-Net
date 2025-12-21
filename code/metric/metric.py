import math
from math import exp
import numpy as np
import jittor as jt
from jittor import nn

# 辅助函数：创建高斯分布向量
def gaussian(window_size, sigma):
    # jittor 中创建 tensor 可以用 jt.array
    gauss = jt.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# 辅助函数：创建 2D 高斯窗口
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # Jittor 中矩阵乘法可以用 matmul 或 @，这里 .mm() 也可以用
    # .float() 在 Jittor 中对应 .float32()，通常默认就是 float32
    _2D_window = _1D_window.matmul(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    
    # expand 在 Jittor 中也支持
    # Jittor 内存布局自动优化，不需要 .contiguous()
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # Jittor 的 conv2d 接口：jt.nn.conv2d(input, weight, stride=1, padding=0, dilation=1, groups=1)
    # 注意：Jittor 的 padding 参数同样支持整数
    padding = window_size // 2
    
    mu1 = nn.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = nn.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = nn.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = nn.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        # Jittor 的 mean 支持传入维度元组或多次调用
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    # 确保数值在 [0, 1] 范围内
    img1 = jt.clamp(img1, min_v=0, max_v=1)
    img2 = jt.clamp(img2, min_v=0, max_v=1)
    
    # 获取维度
    (_, channel, _, _) = img1.shape # Jittor 使用 .shape 属性，而不是 .size() 方法
    
    window = create_window(window_size, channel)
    
    # 移除 .cuda() 和 .type_as()，Jittor 会自动处理设备和类型匹配
    # 如果 img1 是 float32，window 也会自动适配计算
    if img1.dtype != window.dtype:
        window = window.cast(img1.dtype)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(pred, gt):
    # 修改：完全使用 Jittor 操作，不依赖 numpy，这样速度更快且支持自动求导（如果需要）
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    
    imdff = pred - gt
    # 使用 jt.mean 计算均方误差
    rmse = jt.sqrt(jt.mean(imdff ** 2))
    
    # 将结果转为 python float 进行判断，避免除零错误
    rmse_val = rmse.item()
    if rmse_val == 0:
        return 100
        
    return 20 * math.log10(1.0 / rmse_val)