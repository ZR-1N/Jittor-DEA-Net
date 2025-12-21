import jittor as jt
from jittor import nn
from jittor.models.vgg import vgg19

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # 加载 Jittor 的预训练 VGG19
        # 注意：第一次运行会自动下载权重
        vgg_pretrained_features = vgg19(pretrained=True).features
        
        # Jittor 的 Sequential 初始化推荐直接传入 list
        # 利用列表推导式替代原先的 add_module 循环，代码更简洁
        self.slice1 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(12, 21)])
        self.slice5 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(21, 30)])

        # 冻结参数：Jittor 使用 stop_grad()
        if not requires_grad:
            for param in self.parameters():
                param.stop_grad()

    def execute(self, X):  # 修改：forward -> execute
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        # 移除 .cuda()，Jittor 自动处理
        self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        # 权重保持不变
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def execute(self, a, p, n): # 修改：forward -> execute
        # 计算特征
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            # 修改：detach() -> stop_grad()
            # 这里的 stop_grad() 很重要，表示不计算 p 和 n 的梯度，只把它们当作目标值
            d_ap = self.l1(a_vgg[i], p_vgg[i].stop_grad())
            
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].stop_grad())
                # 加一个小常数防止除零
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
            
        return loss