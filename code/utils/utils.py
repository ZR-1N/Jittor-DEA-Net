import cv2
import numpy as np
import jittor as jt
from jittor import nn
from PIL import Image

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pad_img(x, patch_size):
    _, _, h, w = x.shape # .size() -> .shape
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    
    # F.pad -> nn.pad
    # Jittor 的 padding 格式与 PyTorch 一致: (left, right, top, bottom)
    x = nn.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
    return x

def norm_zero_to_one(x):
    return (x - jt.min(x)) / (jt.max(x) - jt.min(x))

def save_heat_image(x, save_path, norm=False):
    if norm:
        x = norm_zero_to_one(x)
    
    # squeeze(dim=0) -> squeeze(0)
    x = x.squeeze(0)
    C, H, W = x.shape
    
    # 链式操作：
    # 1. .mul(255).add_(0.5).clamp_(...) -> 算术运算 + clamp
    # 注意：Jittor 变量一般是不可变的(immutable semantics)，没有 add_ 这种原地操作，
    # 但 Python 操作符重载会处理好引用。
    x = x * 255 + 0.5
    x = jt.clamp(x, 0, 255)
    
    # permute(1, 2, 0) -> permute(1, 2, 0) 接口一致
    x = x.permute(1, 2, 0)
    
    # 转为 numpy uint8
    # PyTorch: .to('cpu', torch.uint8).numpy()
    # Jittor: .cast('uint8').numpy() (numpy()会自动把数据拉回CPU)
    x = x.cast('uint8').numpy()
    
    if C == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)[:, :, ::-1]
    x = Image.fromarray(x)
    x.save(save_path)