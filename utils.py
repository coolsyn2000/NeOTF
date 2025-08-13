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
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

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

    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_circular_mgrid(sidelen, radius):

    full_grid = get_mgrid(sidelen, dim=2)

    dist_sq = torch.sum(full_grid**2, dim=1)

    mask = dist_sq <= radius**2

    circular_grid = full_grid[mask]

    return circular_grid, mask


def pad_to_size(input_tensor, target_height, target_width, mode='constant', value=0):

    input_height = input_tensor.size(-2)
    input_width = input_tensor.size(-1)

    if target_height < input_height or target_width < input_width:
        raise ValueError("size mismatch: target size must be greater than or equal to input size.")

    total_pad_h = target_height - input_height
    total_pad_w = target_width - input_width


    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)

    padded_tensor = F.pad(input_tensor, padding, mode=mode, value=value)
    
    return padded_tensor

def read_speckles_from_folder(folder_path, data_config):
    image_array_list = []
    image_torch_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(data_config.format):
            file_path = os.path.join(folder_path, filename)

            with Image.open(file_path) as img:
                if img.mode != 'L' and data_config.format == 'bmp':
                    img = img.convert('L')
                image_array = np.array(img)
                image_array = image_array.astype(np.float32)

                crop_size = data_config.crop_size

                start_row = (image_array.shape[0] - crop_size) // 2
                end_row = start_row + crop_size
                start_col = (image_array.shape[1] - crop_size) // 2
                end_col = start_col + crop_size

                image_array = image_array[start_row:end_row, start_col:end_col]

                min_value = image_array.min()
                max_value = image_array.max()

                image_array = (image_array - min_value) / (max_value - min_value)

                image_array = cv2.GaussianBlur(image_array, (data_config.blur_kernel_size, data_config.blur_kernel_size), data_config.sigma)
    
                image_array = zoom(image_array, zoom=data_config.desired_size /data_config.crop_size)
                image_torch = torch.from_numpy(image_array)
                image_torch = image_torch.unsqueeze(0).unsqueeze(0)

                image_array_list.append(image_array)
                image_torch_list.append(image_torch)

    return image_array_list, image_torch_list

def load_image_array(image_path, zoom=1):

    img = Image.open(image_path)

    gray_img = img.convert('L')  # 转换为灰度图

    new_size = gray_img.size*zoom  # 例如 (800, 600)
    gray_img = gray_img.resize(new_size)

    # 将图像转换为 NumPy 数组
    gray_img = np.array(gray_img)
    gray_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img)) * 255.0


    gray_img_copy = Image.fromarray(gray_img.astype(np.uint8))
    gray_img_copy.save(image_path)

    return gray_img

def scattering_speckle(png_files, random_phase, config):

    image = load_image_array(png_files)

    image_sidelen = image.shape[0]
    pad_size = (config.speckle_size - image_sidelen) // 2
    image_pad = np.pad(image, (pad_size, pad_size), constant_values=0)
    size = config.speckle_size

    matrix = np.zeros((size, size))
    # 创建圆形区域
    center = (size // 2, size // 2)  # 中心点
    radius = int(config.speckle_size/2 * config.pupil_radius)  # 半径

    for x in range(size):
        for y in range(size):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                matrix[x, y] = 1

    # 计算复数矩阵
    complex_matrix = np.fft.ifftshift(matrix * np.exp(1j * random_phase))

    # 计算傅里叶变换
    fft_result = np.fft.ifft2(complex_matrix)

    # with random phase
    speckle_psf = np.abs(fft_result) ** 2
    speckle_psf = (speckle_psf-np.min(speckle_psf))/(np.max(speckle_psf)-np.min(speckle_psf))

    speckle = simu_memory_effect(image_pad,speckle_psf)

    # without random phase
    diffraction_psf = np.abs(np.fft.ifft2(np.fft.ifftshift(matrix))) ** 2
    diffraction_image = simu_memory_effect(image_pad, diffraction_psf)
    diffraction_image = crop_center(diffraction_image,image_sidelen)

    return speckle, speckle_psf, image, diffraction_image

def simu_memory_effect(image,psf):

    F_matrix1 = np.fft.fft2(image)
    F_matrix2 = np.fft.fft2(psf)
    F_convolution = F_matrix1 * F_matrix2
    convolution_result = np.fft.ifft2(F_convolution)
    speckle = np.abs(convolution_result)**2
    speckle = speckle - speckle.min()
    speckle = speckle / speckle.max()

    return speckle

def crop_center(image, crop_size):

    sidelen = image.shape[0]

    start = (sidelen-crop_size)//2

    return image[start:start+crop_size,start:start+crop_size]
