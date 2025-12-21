import math
from math import exp
import numpy as np
import jittor as jt
from jittor import nn

def gaussian(window_size, sigma):
    # torch.Tensor -> jt.array
    gauss = jt.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    # unsqueeze 在 Jittor 中完全支持
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # .mm() 矩阵乘法 -> .matmul() 或 @ (为了清晰推荐 matmul)
    # .float() 默认就是 float32，通常不需要显式转
    _2D_window = _1D_window.matmul(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    
    # Jittor 自动处理连续内存，不需要 .contiguous()
    # expand 接口一致
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # F.conv2d -> nn.conv2d
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
        return ssim_map.mean(1).mean(1).mean(1)

def val_ssim(img1, img2, window_size=11, size_average=True):
    # torch.clamp -> jt.clamp
    img1 = jt.clamp(img1, min_v=0, max_v=1)
    img2 = jt.clamp(img2, min_v=0, max_v=1)
    
    # 获取维度: .size() -> .shape
    (_, channel, _, _) = img1.shape
    
    window = create_window(window_size, channel)
    
    # 移除 .cuda() 和 .type_as()，Jittor 自动管理
    if img1.dtype != window.dtype:
        window = window.cast(img1.dtype)
        
    return _ssim(img1, img2, window, window_size, channel, size_average)

def val_psnr(pred, gt):
    # 优化：直接在 Jittor 变量上计算，减少 CPU-GPU 传输
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    
    imdff = pred - gt
    # 使用 jt.mean 计算 MSE
    mse = jt.mean(imdff ** 2)
    
    # 转为 Python float 做除法判断
    mse_val = mse.item()
    if mse_val == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse_val))