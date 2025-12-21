import jittor as jt
from jittor import nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 修正：Jittor Conv2d 不直接支持 padding_mode='reflect'
        # 拆分为 ReflectionPad2d + Conv2d
        self.pad = nn.ReflectionPad2d(3)
        self.sa = nn.Conv2d(2, 1, 7, padding=0, bias=True)

    def execute(self, x): # 修改：forward -> execute
        # Jittor 的 mean 和 max 均支持 keepdims 参数
        x_avg = jt.mean(x, dim=1, keepdims=True)
        
        # 修正：jt.max 默认只返回 value，不需要像 pytorch 那样解包 (values, indices)
        x_max = jt.max(x, dim=1, keepdims=True)
        
        # 修正：torch.concat -> jt.concat
        x2 = jt.concat([x_avg, x_max], dim=1)
        
        # 先填充再卷积
        x2 = self.pad(x2)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        # Jittor 也有 AdaptiveAvgPool2d
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            # inplace操作在 Jittor 中通常不强制要求，普通 ReLU 即可
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def execute(self, x): # 修改：forward -> execute
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # 修正：同样处理 reflect padding
        self.pad = nn.ReflectionPad2d(3)
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=0, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x, pattn1): # 修改：forward -> execute
        # x: [B, C, H, W]
        # pattn1: [B, C, H, W]
        
        # 原逻辑解析：Rearrange('b c t h w -> b (c t) h w')
        # 这意味着它先堆叠两个张量，然后把 C 和 T 维度合并。
        # 结果通道顺序是：[C1_x, C1_p, C2_x, C2_p, ...] (交错)
        
        # 1. 在 dim=2 维度堆叠 (相当于 unsqueeze + concat)
        # stack 后 shape: [B, C, 2, H, W]
        x2 = jt.stack([x, pattn1], dim=2)
        
        B, C, _, H, W = x2.shape
        
        # 2. Reshape 实现 Rearrange
        # [B, C, 2, H, W] -> [B, C*2, H, W]
        # Jittor 的 reshape 默认就是按顺序合并维度，这与 'b c t h w -> b (c t) h w' 逻辑一致
        x2 = x2.reshape(B, C * 2, H, W)
        
        # 3. 卷积 (带 padding)
        x2 = self.pad(x2)
        pattn2 = self.pa2(x2)
        
        pattn2 = self.sigmoid(pattn2)
        return pattn2