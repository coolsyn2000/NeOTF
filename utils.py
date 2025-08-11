import os
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import yaml
from datetime import datetime
import random

def set_seed(seed):
    """
    固定所有随机数种子以确保实验的可复现性。

    :param seed: 一个整数，用作所有库的种子。
    """
    # 1. 固定 Python 内置的随机数生成器
    random.seed(seed)
    
    # 2. 固定 NumPy 的随机数生成器
    np.random.seed(seed)
    
    # 3. 固定 PyTorch 的CPU随机数生成器
    torch.manual_seed(seed)
    
    # 4. 如果使用GPU，还需要固定GPU的随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # 如果使用多个GPU，还需要为所有GPU设置种子
        torch.cuda.manual_seed_all(seed)
    
    # 5. 确保cuDNN的确定性行为
    # cuDNN是NVIDIA用于深度学习的加速库，其某些算法具有不确定性。
    # 设置为True可以强制cuDNN使用确定性算法，但这可能会牺牲一些性能。
    torch.backends.cudnn.deterministic = True
    
    # 6. 禁用cuDNN的基准测试模式
    # benchmark模式会为当前输入尺寸自动寻找最高效的算法，这可能导致不确定性。
    torch.backends.cudnn.benchmark = False

    # (可选) 针对某些PyTorch和CUDA版本的额外设置
    # 有时需要设置这个环境变量来完全避免某些操作的不确定性
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Config:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def __repr__(self):

        return str(self.__dict__)

def load_config(path='config.yml'):

    print(f"Loading configuration from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    return config

def total_variation_loss(image, mode='isotropic', reduction='sum'):
    """
    Calculates the Total Variation (TV) loss for a batch of images.

    :param image: A PyTorch tensor of shape (B, C, H, W).
                  For grayscale images, C should be 1.
    :param mode: 'isotropic' or 'anisotropic'.
    :param reduction: 'sum' or 'mean'.
    :return: A scalar tensor representing the TV loss.
    """
    if image.dim() != 4:
        raise ValueError("Expected a 4D tensor (B, C, H, W).")

    # Calculate gradients
    grad_y = image[..., 1:, :] - image[..., :-1, :]
    grad_x = image[..., :, 1:] - image[..., :, :-1]

    if mode == 'isotropic':
        # Add a small epsilon for numerical stability
        eps = 1e-6
        tv_loss = torch.sqrt(grad_y[..., :, :-1]**2 + grad_x[..., :-1, :]**2 + eps)
    elif mode == 'anisotropic':
        tv_loss = torch.abs(grad_y).sum() + torch.abs(grad_x).sum()
    else:
        raise ValueError(f"Unknown mode: {mode}.")
    
    if reduction == 'sum':
        return tv_loss.sum()
    elif reduction == 'mean':
        return tv_loss.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}.")
    
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_circular_mgrid(sidelen, radius):
    """
    生成一个2D坐标网格，但仅保留位于中心圆形区域内的坐标点。

    Args:
        sidelen (int): 完整正方形网格的边长点数。
        radius (float): 中心圆形的半径。

    Returns:
        torch.Tensor: 一个形状为 [N, 2] 的张量，其中 N 是落在圆形区域内的点的数量。
    """
    # 步骤 1: 生成完整的正方形网格
    full_grid = get_mgrid(sidelen, dim=2)

    # 步骤 2: 计算每个点到中心(0,0)的距离的平方
    # full_grid**2 对每个坐标值求平方 -> [x^2, y^2]
    # torch.sum(..., dim=1) 将 x^2 和 y^2 相加
    dist_sq = torch.sum(full_grid**2, dim=1)

    # 步骤 3: 创建一个布尔掩码(mask)，筛选出在半径内的点
    mask = dist_sq <= radius**2

    # 步骤 4: 使用掩码从完整网格中选择符合条件的坐标
    circular_grid = full_grid[mask]

    return circular_grid, mask


def pad_to_size(input_tensor, target_height, target_width, mode='constant', value=0):
    """
    将一个2D (H, W) 或 4D (B, C, H, W) 的Tensor填充到指定的目标尺寸。

    :param input_tensor: 需要填充的Tensor，支持2D或4D。
    :param target_height: 目标高度。
    :param target_width: 目标宽度。
    :param mode: 填充模式。可选值为 'constant', 'reflect', 'replicate', 'circular'。
                 默认为 'constant'（常量填充）。
    :param value: 当 mode='constant' 时，用于填充的值。默认为 0。
    :return: 填充后的Tensor。
    """
    # 获取输入Tensor的高度和宽度
    input_height = input_tensor.size(-2)
    input_width = input_tensor.size(-1)

    # 检查目标尺寸是否小于当前尺寸
    if target_height < input_height or target_width < input_width:
        raise ValueError("目标尺寸不能小于当前Tensor的尺寸。该函数不支持裁剪。")

    # 计算需要填充的总量
    total_pad_h = target_height - input_height
    total_pad_w = target_width - input_width

    # 将填充量尽可能均匀地分配到两侧
    # (这可以处理总填充量为奇数的情况)
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    # F.pad 的 `pad` 参数是一个元组，顺序非常关键：
    # (左边填充, 右边填充, 上边填充, 下边填充)
    # 这个顺序是针对最后两个维度的。
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    # 执行填充操作
    padded_tensor = F.pad(input_tensor, padding, mode=mode, value=value)
    
    return padded_tensor

def read_speckles_from_folder(folder_path, data_config):
    # 初始化一个空列表来保存所有的NumPy矩阵
    image_array_list = []
    image_torch_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(data_config.format):  # 只处理 .bmp 文件
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 使用Pillow读取图片
            with Image.open(file_path) as img:
                # 将图片转换为NumPy数组
                if img.mode != 'L' and data_config.format == 'bmp':  # 如果不是灰度图则转换
                    img = img.convert('L')
                image_array = np.array(img)  # 转换为灰度图像
                image_array = image_array.astype(np.float32)

                crop_size = data_config.crop_size

                # 计算中心区域的起始和结束索引
                start_row = (image_array.shape[0] - crop_size) // 2
                end_row = start_row + crop_size
                start_col = (image_array.shape[1] - crop_size) // 2
                end_col = start_col + crop_size

                # 截取中心的区域
                image_array = image_array[start_row:end_row, start_col:end_col]

                min_value = image_array.min()
                max_value = image_array.max()

                # 将矩阵的值缩放到 [0, 1] 之间
                image_array = (image_array - min_value) / (max_value - min_value)

                image_array = cv2.GaussianBlur(image_array, (data_config.blur_kernel_size, data_config.blur_kernel_size), data_config.sigma)

                # plt.imshow(image_array)
                # plt.show()
    
                image_array = zoom(image_array, zoom=data_config.desired_size /data_config.crop_size)
                image_torch = torch.from_numpy(image_array)
                image_torch = image_torch.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

                # 将NumPy数组添加到列表中
                image_array_list.append(image_array)
                image_torch_list.append(image_torch)

    # plt.imshow(image_array_list[0], cmap='gray')
    # plt.show()
    return image_array_list, image_torch_list


def crop_center(image, crop_size):

    sidelen = image.shape[0]

    start = (sidelen-crop_size)//2

    return image[start:start+crop_size,start:start+crop_size]
