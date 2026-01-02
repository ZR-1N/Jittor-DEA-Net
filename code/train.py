import os
import time
import math
import numpy as np
import jittor as jt
from jittor import nn, optim

# ----------------- 修正导入路径 -----------------

# 1. Logger (在 code 目录下)
from logger import plot_loss_log, plot_psnr_log

# 2. Metric 和 Utils (在 utils 文件夹下)
# 我们用 utils.metric 而不是根目录的 metric，因为之前为了适配 Jittor 修改过 utils/metric.py
from utils.metric import val_psnr, val_ssim
from utils.utils import pad_img

# 3. Model (在 model 文件夹下)
from model.backbone_train import DEANet 

# 4. Loss (在 loss 文件夹下)
# 尝试从 loss.cr 导入，如果失败则尝试直接从 loss 包导入
try:
    from loss.cr import ContrastLoss
except ImportError:
    from loss import ContrastLoss

# 5. 其他
from option_train import opt
from data.data_loader import TrainDataset, TestDataset

# ------------------------------------------------

# 全局开启 CUDA
jt.flags.use_cuda = 1

start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps

def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr

def train(net, loader_train, loader_test, optimizer, criterion):
    losses = []
    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    # Jittor Dataset 迭代器
    loader_train_iter = iter(loader_train)

    for step in range(start_step + 1, steps + 1):
        net.train()
        
        # 1. 学习率调度
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # 2. 获取数据
        try:
            x, y = next(loader_train_iter)
        except StopIteration:
            loader_train_iter = iter(loader_train)
            x, y = next(loader_train_iter)

        # 3. 前向传播
        out = net(x)
        
        loss_L1 = 0
        loss_CR = 0
        
        if opt.w_loss_L1 > 0:
            loss_L1 = criterion[0](out, y)
        if opt.w_loss_CR > 0:
            loss_CR = criterion[1](out, y, x)
            
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
        
        # 4. 反向传播
        optimizer.step(loss)
        
        # 记录日志
        losses.append(loss.item())
        loss_val = loss.item()
        l1_val = loss_L1.item() if hasattr(loss_L1, 'item') else loss_L1
        cr_val = loss_CR.item() if hasattr(loss_CR, 'item') else loss_CR

        loss_log_tmp['L1'].append(l1_val)
        loss_log_tmp['CR'].append(cr_val)
        loss_log_tmp['total'].append(loss_val)

        print(
            f'\rloss:{loss_val:.5f} | L1:{l1_val:.5f} | '
            f'CR:{opt.w_loss_CR * cr_val:.5f} | '
            f'step :{step}/{steps} | lr :{lr :.7f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        # 绘图逻辑
        if step % opt.iters_per_epoch == 0:
            epoch_idx = int(step / opt.iters_per_epoch)
            for key in loss_log.keys():
                if len(loss_log_tmp[key]) > 0:
                    loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                    loss_log_tmp[key] = []
            
            plot_loss_log(loss_log, epoch_idx, opt.saved_plot_dir)
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)

        # 评估逻辑
        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or \
           (step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train)) == 0):
            
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (5 * len(loader_train))
            else:
                epoch = int(step / opt.iters_per_epoch)
            
            with jt.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir)

            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pk')
                jt.save({
                    'epoch': epoch,
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict() 
                }, saved_best_model_path)
            
            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pk')
            jt.save({
                'epoch': epoch,
                'step': step,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, saved_single_model_path)
            
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
            
            net.train()

def test(net, loader_test):
    net.eval()
    jt.gc()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        with jt.no_grad():
            H, W = inputs.shape[2:]
            
            inputs = pad_img(inputs, 4)
            # 保持和 eval.py 一致，如果 eval.py 没加 (x-0.5)/0.5，这里也不加
            
            pred = net(inputs)
            pred = pred.clamp(0, 1)
            pred = pred[:, :, :H, :W]
            
            # 使用适配 Jittor 的指标函数
            ssim_tmp = val_ssim(pred, targets)
            psnr_tmp = val_psnr(pred, targets)
            
            if hasattr(ssim_tmp, 'item'): ssim_tmp = ssim_tmp.item()
            if hasattr(psnr_tmp, 'item'): psnr_tmp = psnr_tmp.item()
                
            ssims.append(ssim_tmp)
            psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)

def set_seed(seed=2018):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

if __name__ == "__main__":
    set_seed(666)

    # 路径配置：动态读取 opt.dataset
    train_dir = os.path.join('../dataset', opt.dataset, 'train')
    test_dir = os.path.join('../dataset', opt.dataset, 'test')
    
    print(f"Dataset: {opt.dataset}")
    print(f"Train dir: {train_dir}")
    print(f"Test dir: {test_dir}")
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found.")
        exit(1)

    train_set = TrainDataset(os.path.join(train_dir, 'hazy'), os.path.join(train_dir, 'clear'))
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))
    
    # 保持原版配置
    train_set.set_attrs(batch_size=opt.bs, shuffle=True, num_workers=8)
    test_set.set_attrs(batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Train set size: {len(train_set)}")

    net = DEANet(base_dim=32)
    
    total_params = sum(p.numel() for p in net.parameters())
    print("Total_params: ==> {}".format(total_params))

    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(net.parameters(), lr=opt.start_lr, betas=(0.9, 0.999), eps=1e-08)

    train(net, train_set, test_set, optimizer, criterion)