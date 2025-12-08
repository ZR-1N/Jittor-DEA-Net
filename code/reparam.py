import torch
from einops.layers.torch import Rearrange
from collections import OrderedDict  # <-- 我们需要这个
import os

# ----------------------------------------------------
# (reparam.py 原本的 4 个数学函数，原封不动)
# ----------------------------------------------------
def convert_cdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_cd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
    conv_weight_cd[:, :, :] = conv_weight[:, :, :]
    conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
    conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
    return conv_weight_cd

def convert_hdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_hd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
    conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
    conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
    conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
    return conv_weight_hd

def convert_vdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_vd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
    conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
    conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
    conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
    return conv_weight_vd

def convert_adc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_ad = conv_weight - conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
    conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
    return conv_weight_ad
# ----------------------------------------------------
# (下面是我们的“三合一”修复逻辑)
# ----------------------------------------------------

# [步骤 0] 定义输入和输出路径
# (读取您刚复制的 best.pk)
saved_model_path = '../experiment/Dense_Haze/DEA-Net-CR-Dense/saved_model/best.pk' 
# (定义一个新名字，eval.py 将会使用这个)
fused_model_path = '../experiment/Dense_Haze/DEA-Net-CR-Dense/saved_model/best_fused.pk' 

print(f"Loading checkpoint from {saved_model_path}...")
ckp = torch.load(saved_model_path, map_location='cpu')

# [修复 1] 解决“检查点”问题
print("Extracting 'model' state dict...")
ckp = ckp['model']

# [修复 2] 解决 "DataParallel" 的 'module.' 前缀问题
print("Stripping 'module.' prefix...")
stripped_ckp = OrderedDict()
for k, v in ckp.items():
    name = k.replace('module.', '') # 'module.conv1' -> 'conv1'
    stripped_ckp[k.replace('module.', '')] = v

print("Fusing DEConv branches...")
# [修复 3] 在 *干净的* 字典上，执行“融合” (Reparam)
ckp = stripped_ckp  # <--- 关键！
simplified_ckp = {}

# (下面的融合逻辑与原脚本完全一样)
for key in ckp.keys():
    if 'conv1_1' in key:
        if 'weight' in key:
            w_cdc = convert_cdc(ckp[key])
        elif 'bias' in key:
            b_cdc = ckp[key]
    elif 'conv1_2' in key:
        if 'weight' in key:
            w_hdc = convert_hdc(ckp[key])
        elif 'bias' in key:
            b_hdc = ckp[key]
    elif 'conv1_3' in key:
        if 'weight' in key:
            w_vdc = convert_vdc(ckp[key])
        elif 'bias' in key:
            b_vdc = ckp[key]
    elif 'conv1_4' in key:
        if 'weight' in key:
            w_adc = convert_adc(ckp[key])
        elif 'bias' in key:
            b_adc = ckp[key]
    elif 'conv1_5' in key:
        if 'weight' in key:
            w_vc = ckp[key]
        elif 'bias' in key:
            b_vc = ckp[key]
            
            w = w_cdc + w_hdc + w_vdc + w_adc + w_vc
            b = b_cdc + b_hdc + b_vdc + b_adc + b_vc
            
            # (注意：这里的 split 是为了把 'conv1_5' 换成 'conv1')
            simplified_ckp[key.split('conv1_5')[0] + 'weight'] = w
            simplified_ckp[key.split('conv1_5')[0] + 'bias'] = b
    else:
        # 其他的键 (比如 mix1, up1, ...) 直接复制
        simplified_ckp[key] = ckp[key]

# 保存最终的、"纯净"的、"融合后"的模型
torch.save(simplified_ckp, fused_model_path)
print(f"Success! Fused model saved to {fused_model_path}")