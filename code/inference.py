import os
import argparse
import jittor as jt
from jittor import nn
import jittor.transform as transform
from PIL import Image
import numpy as np

from model.backbone import Backbone 

# 开启 CUDA
jt.flags.use_cuda = 1

# ---------------------------------------------------------
# 1. 设置参数
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='存放您自己照片的文件夹路径')
parser.add_argument('--output_dir', type=str, required=True, help='结果保存路径')
# Jittor 模型通常后缀为 .pkl，但也兼容 .pth
parser.add_argument('--model_path', type=str, required=True, help='您的模型路径 (必须是融合后的权重)')
# device 参数在 Jittor 中其实不需要，但为了兼容命令行习惯保留，内部通过 jt.flags 控制
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu (Jittor自动管理，此参数仅做占位)')
args = parser.parse_args()

# 如果强制指定 cpu
if args.device == 'cpu':
    jt.flags.use_cuda = 0

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. 加载模型
# ---------------------------------------------------------
print(f"Loading model from {args.model_path}...")

# 初始化模型 (Jittor 不需要 .to(device))
net = Backbone()

# 加载权重字典
try:
    # jt.load 可以加载 .pkl 和 .pth
    ckpt = jt.load(args.model_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 兼容性处理：有些 ckpt 直接是字典，有些包含 'model' 键
if isinstance(ckpt, dict) and 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

# 自动剥离 'module.' 前缀 (处理多卡训练遗留的命名问题)
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v

# 加载权重到模型
try:
    net.load_parameters(new_state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print("\n[ERROR] 权重加载失败！")
    print("可能的原因：")
    print("1. 权重文件路径错误。")
    print("2. 模型结构不匹配（请确保使用的是 inference 阶段的 Backbone，配合融合后的权重）。")
    print(f"详细错误: {e}\n")
    exit(1)

net.eval()

# ---------------------------------------------------------
# 3. 推理函数
# ---------------------------------------------------------
def inference_image(img_path, save_path):
    # 读取图片
    original_img = Image.open(img_path).convert('RGB')
    w, h = original_img.size
    
    # 转为 Tensor
    # Jittor 的 ToTensor 也会归一化到 [0, 1] 并转为 (C, H, W)
    img_tensor = transform.ToTensor()(original_img)
    # 增加 batch 维度: (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Padding: 保持与 eval.py 一致
    factor = 4
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    
    # 使用 reflect padding
    # Jittor nn.pad 参数顺序：(left, right, top, bottom)
    img_tensor_padded = nn.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    print(f"Processing {os.path.basename(img_path)} | Original: {w}x{h} | Padded: {img_tensor_padded.shape[3]}x{img_tensor_padded.shape[2]}")

    # 推理
    with jt.no_grad():
        try:
            # 核心去雾
            output_tensor = net(img_tensor_padded)
            
            # 截取有效区域 (Un-padding)
            output_tensor = output_tensor[:, :, :h, :w]
            
            # 限制范围在 [0, 1]
            output_tensor = output_tensor.clamp(0, 1)
            
            # --- 保存结果 (Jittor 手动处理) ---
            # 1. 去掉 batch 维度: (C, H, W)
            output_tensor = output_tensor.squeeze(0)
            # 2. 转换维度: (H, W, C)
            output_tensor = output_tensor.permute(1, 2, 0)
            # 3. 反归一化并转为 uint8 numpy
            out_np = (output_tensor.numpy() * 255).astype(np.uint8)
            
            output_img = Image.fromarray(out_np)
            output_img.save(save_path)
            print(f"Saved to {save_path}")
            
        except Exception as e:
            # Jittor 的 OOM 错误通常包含 'Out of Memory'
            if 'Memory' in str(e):
                print(f"ERROR: 显存不足 (OOM) 处理 {os.path.basename(img_path)}。请尝试缩小图片尺寸。")
                jt.gc() # 清理显存
            else:
                raise e

# ---------------------------------------------------------
# 4. 主循环
# ---------------------------------------------------------
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
if not os.path.exists(args.input_dir):
    print(f"Error: Input directory '{args.input_dir}' not found.")
    exit()

files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in img_extensions]

print(f"Found {len(files)} images.")

for file_name in files:
    input_path = os.path.join(args.input_dir, file_name)
    output_path = os.path.join(args.output_dir, file_name)
    inference_image(input_path, output_path)
    # 手动调用垃圾回收，确保显存及时释放
    jt.gc()

print("Done!")