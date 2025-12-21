import os
import jittor as jt
from tqdm import tqdm

# Jittor 替代 torchvision.utils.save_image
from jittor.misc import save_image

# 保持你源代码的引用结构
from utils.utils import AverageMeter, pad_img
from utils.metric import val_psnr, val_ssim
from data.data_loader import ValDataset
from option import opt

# [修正] 严格保持原来的 model (单数)
from model.backbone import Backbone 

# 开启 CUDA
jt.flags.use_cuda = 1

def eval(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    # Jittor 显存回收
    jt.gc()

    network.eval()

    for batch in tqdm(val_loader, desc='evaluation'):
        # Jittor 数据自动流向 GPU，不需要 .cuda()
        hazy_img = batch['hazy']
        clear_img = batch['clear']
        
        # 兼容处理：如果是 list (batch_size=1)，取第一个元素作为文件名
        file_name = batch['filename']

        with jt.no_grad():
            H, W = hazy_img.shape[2:]
            
            # Padding
            hazy_img = pad_img(hazy_img, 4)
            
            output = network(hazy_img)
            
            output = output.clamp(0, 1)
            
            # Un-padding
            output = output[:, :, :H, :W]

            if opt.save_infer_results:
                # 确保保存目录存在
                if not os.path.exists(opt.saved_infer_dir):
                    os.makedirs(opt.saved_infer_dir, exist_ok=True)
                
                # Jittor save_image
                save_path = os.path.join(opt.saved_infer_dir, file_name[0])
                save_image(output.squeeze(0), save_path)

        # 计算指标
        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img)
        
        # 类型转换
        if isinstance(ssim_tmp, jt.Var):
            ssim_tmp = ssim_tmp.item()
        if isinstance(psnr_tmp, jt.Var):
            psnr_tmp = psnr_tmp.item()

        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    # 初始化网络 (不需要 .cuda())
    network = Backbone()

    # 初始化数据集
    # 路径拼接逻辑保持源代码习惯
    val_dataset = ValDataset(os.path.join(opt.val_dataset_dir, 'hazy'), os.path.join(opt.val_dataset_dir, 'clear'))
    
    # Jittor 替代 DataLoader
    val_dataset.set_attrs(
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    # 加载模型
    # 这里加一个文件名提取，防止路径重复拼接报错
    model_filename = os.path.basename(opt.pre_trained_model)
    model_path = os.path.join('../trained_models', opt.dataset, model_filename)

    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        # 备用逻辑：如果上面那个路径找不到，试试直接用传入的路径
        if os.path.exists(opt.pre_trained_model):
            model_path = opt.pre_trained_model
            print(f"Found model at: {model_path}")
        else:
            print(f"Error: Model file not found at {model_path}")
            exit(1)

    # Jittor 加载权重
    network.load(model_path)

    # 开始评估
    avg_psnr, avg_ssim = eval(val_dataset, network) 
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))