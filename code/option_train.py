import argparse
import os
import jittor as jt

parser = argparse.ArgumentParser()

# Jittor 自动管理设备，use_cuda 标志在 main 中设置，这里保留参数是为了兼容性，但不再用于逻辑判断
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--iters_per_epoch', type=int, default=5000)
parser.add_argument('--finer_eval_step', type=int, default=400000)
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--start_lr', default=0.0004, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')

parser.add_argument('--w_loss_L1', default=1., type=float, help='weight of loss L1')
parser.add_argument('--w_loss_CR', default=0.1, type=float, help='weight of loss CR')

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--model_name', type=str, default='MDCTDN')
# 这些目录将自动基于 model_name 生成
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--saved_plot_dir', type=str, default='saved_plot')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

parser.add_argument('--dataset', type=str, default='ITS')

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--pre_trained_model', type=str, default='null')

opt = parser.parse_args()

# 路径配置
dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)

# 自动递归创建目录
opt.saved_model_dir = os.path.join(model_dir, 'saved_model')
opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
opt.saved_plot_dir = os.path.join(model_dir, 'saved_plot')
opt.saved_infer_dir = os.path.join(model_dir, 'saved_infer')

os.makedirs(opt.saved_model_dir, exist_ok=True)
os.makedirs(opt.saved_data_dir, exist_ok=True)
os.makedirs(opt.saved_plot_dir, exist_ok=True)
os.makedirs(opt.saved_infer_dir, exist_ok=True)

import json
with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)

print(f"Options saved to {model_dir}")