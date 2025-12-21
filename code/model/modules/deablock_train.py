import jittor as jt
from jittor import nn

# 引入之前转换好的模块
from .deconv import DEConv
from .cga import SpatialAttention, ChannelAttention, PixelAttention

class DEABlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlockTrain, self).__init__()
        # 强制使用 DEConv 作为第一个卷积
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def execute(self, x): # 修改: forward -> execute
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        
        res = self.conv2(res)
        
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        
        res = res * pattn2
        res = res + x
        return res


class DEBlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlockTrain, self).__init__()
        # 强制使用 DEConv
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def execute(self, x): # 修改: forward -> execute
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        
        res = self.conv2(res)
        res = res + x
        return res