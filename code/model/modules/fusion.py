import jittor as jt
from jittor import nn

# 假设 cga.py 和 fusion.py 在同一个目录下
from .cga import SpatialAttention, ChannelAttention, PixelAttention


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x, y): # 修改：forward -> execute
        initial = x + y
        
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        
        pattn2 = self.sigmoid(self.pa(initial, pattn1))

        
        # 融合公式：initial + 加权融合
        # Jittor 的张量运算与 PyTorch 完全一致
        result = initial + pattn2 * x + (1 - pattn2) * y
        
        result = self.conv(result)
        return result