import torch
import jittor as jt
import os
import argparse
from collections import OrderedDict

# -------------------------------------------------------------------------
# 数学融合逻辑 (适配 PyTorch Tensor 输入)
# -------------------------------------------------------------------------
# 尝试导入 einops，如果没有则报错提示安装
try:
    from einops.layers.torch import Rearrange
except ImportError:
    print("Error: Please install einops first: pip install einops")
    exit(1)

def convert_cdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
    conv_weight_cd[:, :, :] = conv_weight[:, :, :]
    conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
    conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
    return conv_weight_cd

def convert_hdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_hd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
    conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
    conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
    conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
    return conv_weight_hd

def convert_vdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_vd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
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

# -------------------------------------------------------------------------
# 主转换逻辑
# -------------------------------------------------------------------------
def convert(pth_path, save_path):
    print(f"Loading PyTorch checkpoint: {pth_path}")
    if not os.path.exists(pth_path):
        print(f"Error: File not found: {pth_path}")
        return

    # 加载 PyTorch 权重到 CPU
    try:
        ckp = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {pth_path}: {e}")
        return

    # 1. 提取 state_dict
    if 'model' in ckp:
        ckp = ckp['model']
    
    # 2. 去除 module. 前缀
    stripped_ckp = OrderedDict()
    for k, v in ckp.items():
        name = k.replace('module.', '')
        stripped_ckp[name] = v
    
    ckp = stripped_ckp
    final_dict = {}
    
    # 3. 遍历并融合权重
    keys_to_process = list(ckp.keys())
    processed_bases = set()

    print("Starting fusion process...")
    for key in keys_to_process:
        if 'conv1_5' in key: # 以最后一个分支为触发点
            base_key = key.split('conv1_5')[0]
            if base_key in processed_bases: continue
            
            # 找到5个分支的权重
            w_cdc = convert_cdc(ckp[base_key + 'conv1_1.weight'])
            w_hdc = convert_hdc(ckp[base_key + 'conv1_2.weight'])
            w_vdc = convert_vdc(ckp[base_key + 'conv1_3.weight'])
            w_adc = convert_adc(ckp[base_key + 'conv1_4.weight'])
            w_vc  = ckp[base_key + 'conv1_5.weight']
            
            w_final = w_cdc + w_hdc + w_vdc + w_adc + w_vc
            
            b_cdc = ckp[base_key + 'conv1_1.bias']
            b_hdc = ckp[base_key + 'conv1_2.bias']
            b_vdc = ckp[base_key + 'conv1_3.bias']
            b_adc = ckp[base_key + 'conv1_4.bias']
            b_vc  = ckp[base_key + 'conv1_5.bias']
            
            b_final = b_cdc + b_hdc + b_vdc + b_adc + b_vc
            
            # 存入新字典，改名为 standard conv1
            final_dict[base_key + 'conv1.weight'] = w_final.numpy() # 转为 numpy 传给 Jittor
            final_dict[base_key + 'conv1.bias'] = b_final.numpy()
            
            processed_bases.add(base_key)
            # print(f"Fused branch: {base_key}") # 减少刷屏

        elif any(x in key for x in ['conv1_1', 'conv1_2', 'conv1_3', 'conv1_4']):
            # 这些是中间分支，已经被合并了，跳过
            continue
        else:
            # 普通层，直接复制 (注意转 numpy)
            if isinstance(ckp[key], torch.Tensor):
                final_dict[key] = ckp[key].numpy()
            else:
                final_dict[key] = ckp[key]

    # 4. 保存为 Jittor 格式
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jt.save(final_dict, save_path)
    print(f"Success! Saved Jittor model to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input .pth file')
    parser.add_argument('--output', type=str, required=True, help='Output .pkl file')
    args = parser.parse_args()
    
    convert(args.input, args.output)
