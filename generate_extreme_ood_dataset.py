import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from models.zernike import DifferentiableZernikeGenerator
DPDD_WIDTH = 1680
DPDD_HEIGHT = 1120

class ExtremeAberrationNet(nn.Module):

    def __init__(self, n_modes: int=36):
        super().__init__()
        if n_modes < 11:
            raise ValueError('n_modes must be >= 11 to include required Noll modes.')
        self.n_modes = n_modes

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f'coords must be [N, 2], got {tuple(coords.shape)}')
        y = coords[:, 0]
        x = coords[:, 1]
        r2 = x.square() + y.square()
        r = torch.sqrt(torch.clamp(r2, min=0.0))
        coeffs = torch.zeros((coords.shape[0], self.n_modes), dtype=coords.dtype, device=coords.device)
        coeffs[:, 3] = 0.5
        ast_base = 0.9 * r2
        coeffs[:, 4] = ast_base * (0.6 + 0.4 * torch.sign(x + 1e-06))
        coeffs[:, 5] = ast_base * (0.6 + 0.4 * torch.sign(y + 1e-06))
        coma_gain = 0.7 * r + 0.55 * r2
        coeffs[:, 6] = coma_gain * torch.sign(x + 1e-06)
        coeffs[:, 7] = coma_gain * torch.sign(y + 1e-06)
        coeffs[:, 10] = 1.0
        return coeffs

def list_images(folder: Path):
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in valid_ext and p.is_file()])

def preprocess_image(pil_img: Image.Image, mode: str='center_crop') -> torch.Tensor:
    img = pil_img.convert('RGB')
    (w, h) = img.size
    (target_w, target_h) = (DPDD_WIDTH, DPDD_HEIGHT)
    resample_lanczos = getattr(Image, 'Resampling', Image).LANCZOS
    if mode == 'resize':
        img = img.resize((target_w, target_h), resample_lanczos)
    else:
        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), resample_lanczos)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    img_tensor = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0)

def tensor_to_pil_uint8(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    t = t.detach().clamp(0.0, 1.0).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    return Image.fromarray(arr, mode='RGB')

def build_physical_layer(device: torch.device) -> SpatiallyVaryingPhysicalLayer:
    aberration_net = ExtremeAberrationNet(n_modes=36).to(device)
    zernike_generator = DifferentiableZernikeGenerator(n_modes=36, pupil_size=129, kernel_size=65, device=str(device)).to(device)
    layer = SpatiallyVaryingPhysicalLayer(aberration_net=aberration_net, zernike_generator=zernike_generator, patch_size=128, stride=64).to(device)
    layer.eval()
    return layer

def parse_args():
    parser = argparse.ArgumentParser(description='Generate extreme OOD blurred dataset using analytic spatially-varying aberrations.')
    parser.add_argument('--source_dir', type=str, default='data/dd_dp_dataset_png/val_c/target', help='输入清晰图目录（默认使用 DPDD val_c/target）')
    parser.add_argument('--target_dir', type=str, default='data/OOD', help='输出 OOD 模糊图目录')
    parser.add_argument('--num_images', type=int, default=74, help='最多处理多少张图像')
    parser.add_argument('--preprocess_mode', type=str, choices=['center_crop', 'resize'], default='center_crop', help='输入图预处理方式')
    return parser.parse_args()

def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        raise FileNotFoundError(f'source_dir not found: {source_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] Device: {device}')
    print(f'[Info] Source: {source_dir}')
    print(f'[Info] Target: {target_dir}')
    layer = build_physical_layer(device)
    image_paths = list_images(source_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f'No images found in: {source_dir}')
    image_paths = image_paths[:args.num_images]
    print(f'[Info] Processing {len(image_paths)} images...')
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc='Generating Extreme OOD', ncols=100):
            img = Image.open(img_path)
            img_tensor = preprocess_image(img, mode=args.preprocess_mode).to(device)
            blurred = layer(img_tensor)
            out_img = tensor_to_pil_uint8(blurred)
            save_path = target_dir / img_path.name
            out_img.save(save_path)
    print('[Done] Extreme OOD dataset generation completed.')
if __name__ == '__main__':
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    main()