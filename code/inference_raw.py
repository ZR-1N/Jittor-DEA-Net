import os
import argparse
import numpy as np
from PIL import Image

import jittor as jt
from jittor import nn
import jittor.transform as transform

from model.backbone_train import DEANet 

jt.flags.use_cuda = 1

# 1.set parameter

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='输入图片文件夹')
parser.add_argument('--output_dir', type=str, required=True, help='输出保存文件夹')
parser.add_argument('--model_path', type=str, required=True, help='您的 best.pk 路径 (训练出的原始权重)')

parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 2.load model
print(f"Loading RAW model from {args.model_path}...")

net = DEANet(base_dim=32)

try:
    ckpt = jt.load(args.model_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

if isinstance(ckpt, dict) and 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v

try:
    net.load_parameters(new_state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print("\n[ERROR] 权重加载失败！")
    print("请确认：")
    print("1. 您使用的是 'best.pk' (训练生成的未融合权重)。")
    print("2. 这里的代码使用的是 DEANet (训练版架构)，而不是 Backbone (推理版)。")
    raise e

net.eval()


# 3. inference function
def inference_image(img_path, save_path):
    try:
        original_img = Image.open(img_path).convert('RGB')
    except Exception:
        return # 跳过非图片文件

    w, h = original_img.size
    
    # 图片转 Tensor (Jittor 会自动归一化到 0-1)
    img_tensor = transform.ToTensor()(original_img)
    # 增加 Batch 维度: (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Padding (确保是 16 的倍数)
    # 训练版网络因为有多层下采样，对尺寸对齐要求较高，建议用 16
    factor = 16
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    
    # Jittor Pad: (left, right, top, bottom)
    img_tensor_padded = nn.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    print(f"Processing {os.path.basename(img_path)} | Original: {w}x{h} -> Padded: {img_tensor_padded.shape[3]}x{img_tensor_padded.shape[2]}")

    with jt.no_grad():
        # 推理
        output_tensor = net(img_tensor_padded)
        
        # 裁剪回原尺寸 (Un-padding)
        output_tensor = output_tensor[:, :, :h, :w]
        
        # 限制范围
        output_tensor = output_tensor.clamp(0, 1)
        
        # --- Jittor Tensor 转图片保存 ---
        # 1. 去掉 Batch 维度: (C, H, W)
        output_tensor = output_tensor.squeeze(0)
        # 2. 维度置换: (H, W, C)
        output_tensor = output_tensor.permute(1, 2, 0)
        # 3. 转 numpy 并放大到 0-255
        out_np = (output_tensor.numpy() * 255).astype(np.uint8)
        
        # 4. 保存
        output_img = Image.fromarray(out_np)
        output_img.save(save_path)

# 4. run main
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
if not os.path.exists(args.input_dir):
    print(f"Error: Input directory {args.input_dir} not found.")
    exit(1)

files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in img_extensions]

print(f"Found {len(files)} images.")

for file_name in files:
    input_path = os.path.join(args.input_dir, file_name)
    output_path = os.path.join(args.output_dir, file_name)
    
    inference_image(input_path, output_path)
    
    # 显存清理 (处理大图或多图时很重要)
    jt.gc()

print(f"Done! Results saved to {args.output_dir}")