import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
# [修正 1] 导入 Backbone 而不是 DEANet，因为我们要加载融合后的权重
from model import Backbone 

# ---------------------------------------------------------
# 1. 设置参数
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='存放您自己照片的文件夹路径')
parser.add_argument('--output_dir', type=str, required=True, help='结果保存路径')
parser.add_argument('--model_path', type=str, required=True, help='您的 .pk 模型路径 (必须是 best_fused.pk)')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. 加载模型
# ---------------------------------------------------------
print(f"Loading model from {args.model_path}...")

# [修正 2] 使用 Backbone() 初始化，不带参数
# 这里的 Backbone 对应 eval.py 中的结构，也就是融合后的单分支结构
net = Backbone().to(args.device)

# 加载权重
ckpt = torch.load(args.model_path, map_location=args.device)

# 兼容性处理：有些 ckpt 直接是字典，有些包含 'model' 键
if 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

# 自动剥离 'module.' 前缀 (为了兼容性保留这个逻辑)
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v

# 加载权重到模型
try:
    net.load_state_dict(new_state_dict)
    print("Model loaded successfully!")
except RuntimeError as e:
    print("\n[ERROR] 权重加载失败！")
    print("请确认您使用的是 'best_fused.pk' (融合后权重)，而不是 'best.pk' (训练时权重)。")
    print("推理脚本 (inference.py) 使用的是 Backbone 结构，必须配合融合后的权重使用。\n")
    raise e

net.eval()

# ---------------------------------------------------------
# 3. 推理函数
# ---------------------------------------------------------
def inference_image(img_path, save_path):
    # 读取图片
    original_img = Image.open(img_path).convert('RGB')
    w, h = original_img.size
    
    # 转为 Tensor
    img_tensor = ToTensor()(original_img).unsqueeze(0).to(args.device)
    
    # [修正 3] Padding: eval.py 使用的是 4，我们也改成 4
    factor = 4
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    
    # 使用 reflect padding
    img_tensor_padded = F.pad(img_tensor, (0, w_pad, 0, h_pad), 'reflect')

    print(f"Processing {os.path.basename(img_path)} | Original: {w}x{h} | Padded: {img_tensor_padded.shape[3]}x{img_tensor_padded.shape[2]}")

    # 推理
    with torch.no_grad():
        try:
            # 核心去雾
            output_tensor = net(img_tensor_padded)
            
            # 截取有效区域 (Un-padding: 去掉之前补的边)
            output_tensor = output_tensor[:, :, :h, :w]
            
            # 限制范围在 [0, 1]
            output_tensor = output_tensor.clamp(0, 1)
            
            # 保存结果
            output_img = ToPILImage()(output_tensor.squeeze(0).cpu())
            output_img.save(save_path)
            print(f"Saved to {save_path}")
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"ERROR: 显存不足 (OOM) 处理 {os.path.basename(img_path)}。请尝试缩小图片尺寸。")
                torch.cuda.empty_cache() # 清理显存
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

print("Done!")