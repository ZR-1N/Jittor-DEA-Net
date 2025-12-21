import os
import random
from PIL import Image
from jittor.dataset import Dataset
import jittor.transform as transform

# ---------------------------------------------------------
# [新增] 辅助函数：自动寻找对应的 clear 图片 (兼容 png 和 jpg)
# ---------------------------------------------------------
def get_clear_image_path(hazy_name, clear_dir):
    # 1. 提取 ID (例如 6369_0.95.jpg -> 6369)
    if '_' in hazy_name:
        image_id = hazy_name.split('_')[0]
    else:
        # 如果没有下划线，尝试去掉后缀作为 ID
        image_id = os.path.splitext(hazy_name)[0]

    # 2. 依次尝试常见后缀
    # 优先找 .jpg (OTS 常见)，然后找 .png (ITS 常见)
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate_name = image_id + ext
        candidate_path = os.path.join(clear_dir, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path

    # 3. 如果 ID 匹配失败，尝试直接用原名 (有些数据集 hazy 和 clear 同名)
    if os.path.exists(os.path.join(clear_dir, hazy_name)):
        return os.path.join(clear_dir, hazy_name)

    # 4. 如果都找不到，返回一个默认猜测 (虽然之后 open 会报错，但至少路径逻辑完整)
    # OTS 绝大多数是 jpg，所以默认回落到 jpg
    return os.path.join(clear_dir, image_id + '.jpg')


class TrainDataset(Dataset):
    def __init__(self, hazy_path, clear_path):
        super().__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        
        # 过滤图片文件
        self.hazy_image_list = [f for f in os.listdir(hazy_path) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        # Jittor Dataset 必须设置 total_len
        self.set_attrs(total_len=len(self.hazy_image_list))

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        
        # [关键修复] 使用辅助函数动态寻找 clear 路径
        clear_image_path = get_clear_image_path(hazy_image_name, self.clear_path)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        # -------------------------------------------------------------------
        # Random Crop (保持之前的修复)
        # -------------------------------------------------------------------
        crop_width, crop_height = 256, 256
        w, h = hazy.size
        
        # 保护机制
        if w < crop_width or h < crop_height:
             hazy = hazy.resize((max(w, crop_width), max(h, crop_height)))
             clear = clear.resize((max(w, crop_width), max(h, crop_height)))
             w, h = hazy.size
        
        # 随机坐标
        i = random.randint(0, h - crop_height)
        j = random.randint(0, w - crop_width)
        
        hazy = hazy.crop((j, i, j + crop_width, i + crop_height))
        clear = clear.crop((j, i, j + crop_width, i + crop_height))

        # -------------------------------------------------------------------
        # Random Rotate (保持还原的逻辑)
        # -------------------------------------------------------------------
        rotate_angle = random.randint(0, 3) * 90
        if rotate_angle != 0:
            hazy = hazy.rotate(rotate_angle)
            clear = clear.rotate(rotate_angle)

        # ToTensor
        hazy = transform.ToTensor()(hazy)
        clear = transform.ToTensor()(clear)

        return hazy, clear


class TestDataset(Dataset):
    def __init__(self, hazy_path, clear_path):
        super().__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = [f for f in os.listdir(hazy_path) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        self.hazy_image_list.sort()
        self.set_attrs(total_len=len(self.hazy_image_list))

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        
        # [关键修复] 动态匹配
        clear_image_path = get_clear_image_path(hazy_image_name, self.clear_path)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        hazy = transform.ToTensor()(hazy)
        clear = transform.ToTensor()(clear)

        return hazy, clear, hazy_image_name


class ValDataset(Dataset):
    def __init__(self, hazy_path, clear_path):
        super().__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = [f for f in os.listdir(hazy_path) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        self.hazy_image_list.sort()
        self.set_attrs(total_len=len(self.hazy_image_list))

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        
        # [关键修复] 动态匹配
        clear_image_path = get_clear_image_path(hazy_image_name, self.clear_path)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        hazy = transform.ToTensor()(hazy)
        clear = transform.ToTensor()(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}