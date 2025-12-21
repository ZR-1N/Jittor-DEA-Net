import jittor as jt
import os
from collections import OrderedDict

# ----------------------------------------------------
# Jittor 版数学转换函数
# 移除了 einops 依赖，使用 reshape 实现相同逻辑
# ----------------------------------------------------

def convert_cdc(w):
    # w shape: [C_out, C_in, K, K]
    out_c, in_c, k1, k2 = w.shape
    
    # 1. Flatten spatial dims: [C_out, C_in, 9]
    # 对应原代码: Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')
    w_flat = w.reshape(out_c, in_c, -1)
    
    # 2. 创建容器
    w_cd = jt.zeros((out_c, in_c, 3 * 3))
    w_cd[:] = w_flat[:] # 复制权重
    
    # 3. 核心逻辑: 中心点 = 原中心 - 所有点之和
    # Index 4 是 3x3 的中心
    w_cd[:, :, 4] = w_flat[:, :, 4] - jt.sum(w_flat, dim=2)
    
    # 4. Reshape 回去
    # 对应原代码: Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2')
    return w_cd.reshape(out_c, in_c, k1, k2)

def convert_hdc(w):
    out_c, in_c, k1, k2 = w.shape
    w_flat = w.reshape(out_c, in_c, -1) # Flatten
    
    w_hd = jt.zeros((out_c, in_c, 3 * 3))
    
    # 水平差分逻辑
    # indices: [0, 3, 6] (第一列) -> 赋值正
    w_hd[:, :, [0, 3, 6]] = w_flat[:, :, :]
    # indices: [2, 5, 8] (第三列) -> 赋值负
    w_hd[:, :, [2, 5, 8]] = -w_flat[:, :, :]
    
    return w_hd.reshape(out_c, in_c, k1, k2)

def convert_vdc(w):
    out_c, in_c, k1, k2 = w.shape
    w_flat = w.reshape(out_c, in_c, -1)
    
    w_vd = jt.zeros((out_c, in_c, 3 * 3))
    
    # 垂直差分逻辑
    # indices: [0, 1, 2] (第一行) -> 赋值正
    w_vd[:, :, [0, 1, 2]] = w_flat[:, :, :]
    # indices: [6, 7, 8] (第三行) -> 赋值负
    w_vd[:, :, [6, 7, 8]] = -w_flat[:, :, :]
    
    return w_vd.reshape(out_c, in_c, k1, k2)

def convert_adc(w):
    out_c, in_c, k1, k2 = w.shape
    w_flat = w.reshape(out_c, in_c, -1)
    
    # 角度差分逻辑
    # 原权重 - 旋转后的权重
    indices = [3, 0, 1, 6, 4, 2, 7, 8, 5]
    w_ad = w_flat - w_flat[:, :, indices]
    
    return w_ad.reshape(out_c, in_c, k1, k2)

# ----------------------------------------------------
# 主逻辑
# ----------------------------------------------------

# [步骤 0] 定义输入和输出路径
# 请确保这个路径指向的是你实际训练出来的权重文件
saved_model_path = '../experiment/Dense_Haze/DEA-Net-CR-Dense/saved_model/best.pk' 
fused_model_path = '../experiment/Dense_Haze/DEA-Net-CR-Dense/saved_model/best_fused.pk' 

# 确保输出目录存在
output_dir = os.path.dirname(fused_model_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print(f"Loading checkpoint from {saved_model_path}...")
# Jittor 的 load 可以处理 PyTorch 保存的字典 (如果是标准 pickle 格式)
ckp = jt.load(saved_model_path)

# [修复 1] 解决“检查点”问题
if 'model' in ckp:
    print("Extracting 'model' state dict...")
    ckp = ckp['model']

# [修复 2] 解决 "DataParallel" 的 'module.' 前缀问题
print("Stripping 'module.' prefix...")
stripped_ckp = OrderedDict()
for k, v in ckp.items():
    name = k.replace('module.', '') # 'module.conv1' -> 'conv1'
    stripped_ckp[name] = v

print("Fusing DEConv branches...")
# [修复 3] 在 *干净的* 字典上，执行“融合” (Reparam)
ckp = stripped_ckp
simplified_ckp = {}

# 遍历所有键进行处理
for key in ckp.keys():
    # 1. 如果是 DEConv 的 5 个分支之一
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
            
            # 当遍历到最后一个分支 (conv1_5) 时，执行加法融合
            # Jittor 支持直接张量相加
            w = w_cdc + w_hdc + w_vdc + w_adc + w_vc
            b = b_cdc + b_hdc + b_vdc + b_adc + b_vc
            
            # 将键名从 '...conv1_5.weight' 改为 '...conv1.weight'
            base_key = key.split('conv1_5')[0]
            simplified_ckp[base_key + 'weight'] = w
            simplified_ckp[base_key + 'bias'] = b
            
            print(f"Fused: {base_key}conv1")
            
    # 2. 如果是其他层的参数 (例如 mix1, up1 等)，直接复制
    else:
        simplified_ckp[key] = ckp[key]

# 保存最终的、"纯净"的、"融合后"的模型
jt.save(simplified_ckp, fused_model_path)
print(f"Success! Fused model saved to {fused_model_path}")