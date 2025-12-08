import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
# [关键修改] 导入 DEANet (训练版架构)，而不是 Backbone
from model import DEANet 

# ---------------------------------------------------------
# 1. 设置参数
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='输入图片文件夹')
parser.add_argument('--output_dir', type=str, required=True, help='输出保存文件夹')
parser.add_argument('--model_path', type=str, required=True, help='您的 best.pk 路径')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. 加载模型 (训练版架构)
# ---------------------------------------------------------
print(f"Loading RAW model from {args.model_path}...")

# 初始化复杂架构 (与 train.py 一致)
net = DEANet(base_dim=32).to(args.device)

# 加载权重
ckpt = torch.load(args.model_path, map_location=args.device)

# 处理检查点字典
if 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

# 处理 DataParallel 的 'module.' 前缀
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v

# 加载
try:
    net.load_state_dict(new_state_dict)
    print("Model loaded successfully!")
except RuntimeError as e:
    print("\n[ERROR] 权重不匹配！请确认您使用的是 'best.pk' (未融合版本)。")
    raise e

net.eval()

# ---------------------------------------------------------
# 3. 推理函数
# ---------------------------------------------------------
def inference_image(img_path, save_path):
    try:
        original_img = Image.open(img_path).convert('RGB')
    except Exception:
        return # 跳过非图片文件

    w, h = original_img.size
    img_tensor = ToTensor()(original_img).unsqueeze(0).to(args.device)
    
    # Padding (确保是 16 的倍数，对齐训练时的下采样)
    factor = 16
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    img_tensor_padded = F.pad(img_tensor, (0, w_pad, 0, h_pad), 'reflect')

    print(f"Processing {os.path.basename(img_path)}...")

    with torch.no_grad():
        # 推理
        output_tensor = net(img_tensor_padded)
        
        # 裁剪回原尺寸
        output_tensor = output_tensor[:, :, :h, :w]
        
        # 保存
        output_tensor = output_tensor.clamp(0, 1)
        output_img = ToPILImage()(output_tensor.squeeze(0).cpu())
        output_img.save(save_path)

# ---------------------------------------------------------
# 4. 运行
# ---------------------------------------------------------
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in img_extensions]

for file_name in files:
    inference_image(os.path.join(args.input_dir, file_name), 
                    os.path.join(args.output_dir, file_name))

print("Done! Check your output directory.")