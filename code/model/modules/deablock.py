import jittor as jt
from jittor import nn
from .cga import SpatialAttention, ChannelAttention, PixelAttention

class DEABlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlock, self).__init__()
        # conv 参数通常是一个类（如 nn.Conv2d），这里实例化它
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU() # Jittor 中 inplace 参数通常可省略
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def execute(self, x): # 修改: forward -> execute
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        
        res = self.conv2(res)
        
        # 注意力机制计算
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        
        res = res * pattn2
        res = res + x
        return res


class DEBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def execute(self, x): # 修改: forward -> execute
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        
        res = self.conv2(res)
        res = res + x
        return res