import os
import argparse
import numpy as np
from PIL import Image

import jittor as jt
from jittor import nn
import jittor.transform as transform

# 【关键修改】这里导入的是推理版架构 Backbone，而不是训练版 DEANet
from model.backbone import Backbone 

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='输入图片文件夹')
parser.add_argument('--output_dir', type=str, required=True, help='输出保存文件夹')
parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径 (.pth 或已转好的 .pk)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 2. 加载模型
print(f"Loading FUSED model from {args.model_path}...")

# 实例化推理版模型
net = Backbone(base_dim=32)

# 加载参数
try:
    ckpt = jt.load(args.model_path)
    # 处理不同来源权重的嵌套问题
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # 剥离可能存在的 module. 前缀 (如果您自己 reparam 出来的模型)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v

    net.load_parameters(new_state_dict)
    print("Pre-trained Model (Fused) loaded successfully!")
except Exception as e:
    print(f"[ERROR] 加载失败。请确保该文件是重参数化后的权重（1路卷积架构）。")
    raise e

net.eval()

# 3. 推理函数 (完全保留您的 Padding 逻辑，支持任意尺寸)
def inference_image(img_path, save_path):
    try:
        original_img = Image.open(img_path).convert('RGB')
    except Exception:
        return 

    w, h = original_img.size
    img_array = transform.ToTensor()(original_img)
    img_tensor = jt.array(img_array).unsqueeze(0)
    
    # Padding 逻辑：模型有 2 层下采样，通常要求是 2^2=4 的倍数
    # 为了保险，Padding 到 16 的倍数是完全可以的
    factor = 16
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    
    img_tensor_padded = nn.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    with jt.no_grad():
        output_tensor = net(img_tensor_padded)
        # 裁剪回原尺寸 (Un-padding)
        output_tensor = output_tensor[:, :, :h, :w]
        output_tensor = output_tensor.clamp(0, 1)
        
        # 保存图片
        output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
        out_np = (output_tensor.numpy() * 255).astype(np.uint8)
        Image.fromarray(out_np).save(save_path)

# 4. 运行
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in img_extensions]

for file_name in files:
    inference_image(os.path.join(args.input_dir, file_name), os.path.join(args.output_dir, file_name))
    jt.gc()

print(f"Done! Results saved to {args.output_dir}")