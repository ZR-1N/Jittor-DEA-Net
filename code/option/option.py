import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='ITS')
parser.add_argument('--val_dataset_dir', type=str)
parser.add_argument('--model_name', type=str, default='DEA-Net', help='experiment name')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

# 仅用于评估/推理
# 注意：Jittor 模型通常保存为 .pkl，但我保留了兼容性
parser.add_argument('--pre_trained_model', type=str, default='null', help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
opt = parser.parse_args()

# 路径配置
opt.val_dataset_dir = os.path.join('../dataset/', opt.dataset, 'test')
exp_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
exp_model_dir = os.path.join(exp_dataset_dir, opt.model_name)

# 1. 统一创建实验根目录和数据集目录
os.makedirs(opt.exp_dir, exist_ok=True)
os.makedirs(exp_dataset_dir, exist_ok=True)

# 2. 处理 infer 目录名称 (修正了你提到的路径包含 .. 的问题)
if opt.pre_trained_model != 'null':
    # os.path.basename: 无论传入 '../model.pkl' 还是 '/abs/path/model.pkl'，只取 'model.pkl'
    file_name = os.path.basename(opt.pre_trained_model)
    # os.path.splitext: 自动识别并去除 .pth 或 .pkl 后缀，取出文件名 'model'
    model_name_pure = os.path.splitext(file_name)[0]
    
    opt.saved_infer_dir = os.path.join(exp_model_dir, model_name_pure)
else:
    # 如果没有预训练模型，给一个默认文件夹名，避免报错
    opt.saved_infer_dir = os.path.join(exp_model_dir, 'inference_results')

# 3. 统一创建模型目录和推理结果目录
os.makedirs(exp_model_dir, exist_ok=True)
os.makedirs(opt.saved_infer_dir, exist_ok=True)

# 保存参数配置
with open(os.path.join(exp_model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)

print(f"Option saved. Inference dir: {opt.saved_infer_dir}")