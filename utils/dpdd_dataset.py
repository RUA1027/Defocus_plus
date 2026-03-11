import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class DPDDDataset(Dataset):
    MODE_TO_FOLDER = {'train': 'train_c', 'val': 'val_c', 'test': 'test_c'}

    def __init__(self, root_dir, mode='train', crop_size=512, repeat_factor=1, transform=None, val_crop_size=1024, use_full_resolution=False, random_flip=False, random_rotate90=False):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.repeat_factor = repeat_factor if mode == 'train' else 1
        self.val_crop_size = val_crop_size
        self.use_full_resolution = use_full_resolution
        self.random_flip = random_flip
        self.random_rotate90 = random_rotate90
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        folder_name = self.MODE_TO_FOLDER.get(mode)
        if folder_name is None:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(self.MODE_TO_FOLDER.keys())}")
        self.split_dir = os.path.join(root_dir, folder_name)
        self.blur_dir = os.path.join(self.split_dir, 'source')
        self.sharp_dir = os.path.join(self.split_dir, 'target')
        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(f"Source or Target directory not found in {self.split_dir}. Expected 'source' and 'target' subdirectories.")
        self.blur_files = sorted([f for f in os.listdir(self.blur_dir) if self._is_image(f)])
        self.sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if self._is_image(f)])
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError(f'Mismatch number of images: {len(self.blur_files)} in source vs {len(self.sharp_files)} in target')
        self._real_length = len(self.blur_files)
        print(f'[DPDDDataset] Mode: {mode}, Real samples: {self._real_length}, Repeat factor: {self.repeat_factor}, Virtual length: {len(self)}')

    def _is_image(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))

    def __len__(self):
        return self._real_length * self.repeat_factor

    def get_real_length(self):
        return self._real_length

    def __getitem__(self, idx):
        real_idx = idx % self._real_length
        blur_filename = self.blur_files[real_idx]
        sharp_filename = self.sharp_files[real_idx]
        blur_path = os.path.join(self.blur_dir, blur_filename)
        sharp_path = os.path.join(self.sharp_dir, sharp_filename)
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        (W_orig, H_orig) = blur_img.size
        if self.use_full_resolution:
            (top, left) = (0, 0)
            (crop_h, crop_w) = (H_orig, W_orig)
        elif self.mode == 'train':
            crop_size = self.crop_size
            if H_orig >= crop_size and W_orig >= crop_size:
                max_top = H_orig - crop_size
                max_left = W_orig - crop_size
                top = random.randint(0, max_top)
                left = random.randint(0, max_left)
                box = (left, top, left + crop_size, top + crop_size)
                blur_img = blur_img.crop(box)
                sharp_img = sharp_img.crop(box)
                (crop_h, crop_w) = (crop_size, crop_size)
            else:
                (top, left) = (0, 0)
                (crop_h, crop_w) = (H_orig, W_orig)
        else:
            crop_size = self.val_crop_size
            if H_orig >= crop_size and W_orig >= crop_size:
                top = (H_orig - crop_size) // 2
                left = (W_orig - crop_size) // 2
                box = (left, top, left + crop_size, top + crop_size)
                blur_img = blur_img.crop(box)
                sharp_img = sharp_img.crop(box)
                (crop_h, crop_w) = (crop_size, crop_size)
            else:
                (top, left) = (0, 0)
                (crop_h, crop_w) = (H_orig, W_orig)
        crop_info = torch.tensor([top / H_orig, left / W_orig, crop_h / H_orig, crop_w / W_orig], dtype=torch.float32)
        if self.mode == 'train':
            if self.random_rotate90:
                rotate_k = random.randint(0, 3)
                if rotate_k == 1:
                    blur_img = blur_img.transpose(Image.Transpose.ROTATE_90)
                    sharp_img = sharp_img.transpose(Image.Transpose.ROTATE_90)
                elif rotate_k == 2:
                    blur_img = blur_img.transpose(Image.Transpose.ROTATE_180)
                    sharp_img = sharp_img.transpose(Image.Transpose.ROTATE_180)
                elif rotate_k == 3:
                    blur_img = blur_img.transpose(Image.Transpose.ROTATE_270)
                    sharp_img = sharp_img.transpose(Image.Transpose.ROTATE_270)
            if self.random_flip:
                if random.random() < 0.5:
                    blur_img = ImageOps.mirror(blur_img)
                    sharp_img = ImageOps.mirror(sharp_img)
                if random.random() < 0.5:
                    blur_img = ImageOps.flip(blur_img)
                    sharp_img = ImageOps.flip(sharp_img)
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)
        return {'blur': blur_tensor, 'sharp': sharp_tensor, 'crop_info': crop_info, 'filename': blur_filename, 'original_size': (H_orig, W_orig)}

class DPDDTestDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.split_dir = os.path.join(root_dir, 'test_c')
        self.blur_dir = os.path.join(self.split_dir, 'source')
        self.sharp_dir = os.path.join(self.split_dir, 'target')
        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(f"Test set directories not found in {self.split_dir}. Expected 'source' and 'target' subdirectories.")
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.blur_files = sorted([f for f in os.listdir(self.blur_dir) if f.lower().endswith(valid_exts)])
        self.sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if f.lower().endswith(valid_exts)])
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError('Mismatch in test set image counts')
        print(f'[DPDDTestDataset] Loaded {len(self.blur_files)} test image pairs')

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur_filename = self.blur_files[idx]
        sharp_filename = self.sharp_files[idx]
        blur_path = os.path.join(self.blur_dir, blur_filename)
        sharp_path = os.path.join(self.sharp_dir, sharp_filename)
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        (W_orig, H_orig) = blur_img.size
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)
        return {'blur': blur_tensor, 'sharp': sharp_tensor, 'crop_info': crop_info, 'filename': blur_filename, 'original_size': (H_orig, W_orig)}


class GenericPairedTestDataset(Dataset):
    """通用配对测试数据集，直接加载 {root}/source/ 和 {root}/target/。
    适用于 RealDOF、extreme aberration、dpdd_pixel 等非标准 DPDD 目录结构。"""

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.blur_dir = os.path.join(root_dir, 'source')
        self.sharp_dir = os.path.join(root_dir, 'target')
        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(
                f"Source or Target directory not found in {root_dir}. "
                f"Expected 'source' and 'target' subdirectories.")
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.blur_files = sorted([f for f in os.listdir(self.blur_dir) if f.lower().endswith(valid_exts)])
        self.sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if f.lower().endswith(valid_exts)])
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError(
                f'Mismatch: {len(self.blur_files)} source vs {len(self.sharp_files)} target in {root_dir}')
        print(f'[GenericPairedTestDataset] Loaded {len(self.blur_files)} paired images from {root_dir}')

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur_filename = self.blur_files[idx]
        sharp_filename = self.sharp_files[idx]
        blur_img = Image.open(os.path.join(self.blur_dir, blur_filename)).convert('RGB')
        sharp_img = Image.open(os.path.join(self.sharp_dir, sharp_filename)).convert('RGB')
        (W_orig, H_orig) = blur_img.size
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return {
            'blur': self.transform(blur_img),
            'sharp': self.transform(sharp_img),
            'crop_info': crop_info,
            'filename': blur_filename,
            'original_size': (H_orig, W_orig),
        }


class BlurOnlyTestDataset(Dataset):
    """无 GT 测试数据集（如 CUHK），仅加载散焦模糊图像。
    支持扁平目录结构（图片直接放在 root_dir 下）。"""

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.blur_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(valid_exts)])
        if len(self.blur_files) == 0:
            raise FileNotFoundError(f'No images found in {root_dir}')
        print(f'[BlurOnlyTestDataset] Loaded {len(self.blur_files)} images (no GT) from {root_dir}')

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur_filename = self.blur_files[idx]
        blur_img = Image.open(os.path.join(self.root_dir, blur_filename)).convert('RGB')
        (W_orig, H_orig) = blur_img.size
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return {
            'blur': self.transform(blur_img),
            'sharp': None,
            'crop_info': crop_info,
            'filename': blur_filename,
            'original_size': (H_orig, W_orig),
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle sharp=None."""
        return {
            'blur': torch.stack([item['blur'] for item in batch]),
            'sharp': None,  # No ground truth
            'crop_info': torch.stack([item['crop_info'] for item in batch]),
            'filename': [item['filename'] for item in batch],
            'original_size': [item['original_size'] for item in batch],
        }