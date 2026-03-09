import torch
import torch.nn as nn
import os
from typing import Optional
from config import Config
from models.zernike import DifferentiableZernikeGenerator
from models.aberration_net import AberrationNet
from models.restoration_net import RestorationNet
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from models.local_grouped_newbp import LocalGroupedZernikeNewBP
from trainer import DualBranchTrainer
from utils.dpdd_dataset import DPDDDataset, DPDDTestDataset
from torch.utils.data import DataLoader

def build_models_from_config(config: Config, device: str):
    use_physical_layer = getattr(config.experiment, 'use_physical_layer', True)
    zernike_gen = None
    aberration_net = None
    physical_layer = None
    newbp_layer = None
    if use_physical_layer:
        zernike_gen = DifferentiableZernikeGenerator(n_modes=config.physics.n_modes, pupil_size=config.physics.pupil_size, kernel_size=config.physics.kernel_size, oversample_factor=config.physics.oversample_factor, wavelengths=config.physics.wavelengths, ref_wavelength=config.physics.ref_wavelength, device=device, learnable_wavelengths=getattr(config.physics, 'learnable_wavelengths', False), wavelength_bounds=getattr(config.physics, 'wavelength_bounds', None))
        use_newbp = getattr(config.aberration_net, 'use_local_grouped_newbp', True)
        aberration_net = AberrationNet(num_coeffs=config.aberration_net.n_coeffs, hidden_dim=config.aberration_net.mlp.hidden_dim, a_max=config.aberration_net.mlp.a_max_mlp, use_fourier=config.aberration_net.mlp.use_fourier, fourier_scale=getattr(config.aberration_net.mlp, 'fourier_scale', 5), output_raw=use_newbp).to(device)
        newbp_layer = None
        if use_newbp:
            groups_cfg = config.aberration_net.newbp.groups
            params_cfg = config.aberration_net.newbp.params
            learnable_cfg = config.aberration_net.newbp.learnable
            newbp_layer = LocalGroupedZernikeNewBP(
                groups_noll={
                    'special': list(groups_cfg.special),
                    'low': list(groups_cfg.low),
                    'mid': list(groups_cfg.mid),
                    'high': list(groups_cfg.high),
                },
                local_joint_enabled=getattr(config.aberration_net.local_joint_input, 'enabled', True),
                kernel_size=getattr(config.aberration_net.local_joint_input, 'kernel_size', 3),
                padding_mode=getattr(config.aberration_net.local_joint_input, 'padding_mode', 'replicate'),
                implementation=getattr(config.aberration_net.newbp, 'implementation', 'native_autograd'),
                separate_special_group=getattr(config.aberration_net.newbp, 'separate_special_group', True),
                special_use_neighborhood=getattr(config.aberration_net.newbp, 'special_use_neighborhood', False),
                params={
                    'special': vars(params_cfg.special),
                    'low': vars(params_cfg.low),
                    'mid': vars(params_cfg.mid),
                    'high': vars(params_cfg.high),
                },
                learnable=vars(learnable_cfg),
            ).to(device)
        print(f'  ├─ 像差网络: AberrationNet (hidden_dim={config.aberration_net.mlp.hidden_dim})')
    use_physics_injection = use_physical_layer and getattr(config.restoration_net, 'use_physics_injection', False)
    n_coeffs = config.aberration_net.n_coeffs if use_physics_injection else 0
    restoration_net = RestorationNet(n_channels=config.restoration_net.n_channels, n_classes=config.restoration_net.n_classes, bilinear=config.restoration_net.bilinear, base_filters=config.restoration_net.base_filters, use_coords=config.restoration_net.use_coords, n_coeffs=n_coeffs).to(device)
    if n_coeffs > 0:
        print(f'  ├─ 复原网络: RestorationNet (base_filters={config.restoration_net.base_filters}, use_coords={config.restoration_net.use_coords}, n_coeffs={n_coeffs})')
    else:
        print(f'  ├─ 复原网络: RestorationNet (base_filters={config.restoration_net.base_filters}, use_coords={config.restoration_net.use_coords})')
    if use_physical_layer:
        if aberration_net is None or zernike_gen is None:
            raise RuntimeError('Physical layer requested but required components are missing.')
        physical_layer = SpatiallyVaryingPhysicalLayer(aberration_net=aberration_net, zernike_generator=zernike_gen, patch_size=config.ola.patch_size, stride=config.ola.stride, newbp_layer=newbp_layer).to(device)
        print(f'  └─ 物理层: OLA (patch={config.ola.patch_size}, stride={config.ola.stride})')
    else:
        print('  └─ 物理层: disabled')
    return (zernike_gen, aberration_net, restoration_net, physical_layer)

def build_trainer_from_config(config: Config, restoration_net, physical_layer, device: str, tensorboard_dir: Optional[str]=None):
    if hasattr(config.training, 'accumulation_steps'):
        accumulation_steps = config.training.accumulation_steps
    else:
        accumulation_steps = 1
    if tensorboard_dir is None and hasattr(config, 'experiment') and hasattr(config.experiment, 'tensorboard'):
        tb_config = config.experiment.tensorboard
        if getattr(tb_config, 'enabled', False):
            base_dir = getattr(tb_config, 'log_dir', 'runs')
            if os.path.isabs(base_dir):
                tensorboard_dir = os.path.join(base_dir, config.experiment.name)
            else:
                tensorboard_dir = os.path.join(config.experiment.output_dir, base_dir, config.experiment.name)
    trainer = DualBranchTrainer(restoration_net=restoration_net, physical_layer=physical_layer, lr_restoration=config.training.optimizer.lr_restoration, lr_optics=config.training.optimizer.lr_optics, optimizer_type=getattr(config.training.optimizer, 'type', 'adamw'), weight_decay=getattr(config.training.optimizer, 'weight_decay', 0.0), lambda_sup=config.training.loss.lambda_sup, lambda_fft=getattr(config.training.loss, 'lambda_fft', 0.1), lambda_coeff=config.training.loss.lambda_coeff, lambda_smooth=config.training.loss.lambda_smooth, lambda_image_reg=config.training.loss.lambda_image_reg, grad_clip_restoration=getattr(config.training.gradient_clip, 'restoration', 5.0), grad_clip_optics=getattr(config.training.gradient_clip, 'optics', 1.0), stage_schedule=config.training.stage_schedule, stage_weights=config.training.stage_weights, smoothness_grid_size=config.training.smoothness_grid_size, use_amp=getattr(config.training, 'use_amp', True), amp_dtype=getattr(config.training, 'amp_dtype', 'float16'), accumulation_steps=accumulation_steps, keep_coeff_loss=getattr(config.training, 'keep_coeff_loss', True), keep_smooth_loss=getattr(config.training, 'keep_smooth_loss', True), device=device, tensorboard_dir=tensorboard_dir)
    return trainer

def build_dataloader_from_config(config: Config, mode: str='train'):
    crop_size = config.data.crop_size
    val_crop_size = getattr(config.data, 'val_crop_size', 1024)
    repeat_factor = getattr(config.data, 'repeat_factor', 1) if mode == 'train' else 1
    use_full_resolution = mode == 'test'
    dataset = DPDDDataset(root_dir=config.data.data_root, mode=mode, crop_size=crop_size, repeat_factor=repeat_factor, val_crop_size=val_crop_size, use_full_resolution=use_full_resolution, random_flip=getattr(getattr(config.data, 'augmentation', None), 'random_flip', False), random_rotate90=getattr(getattr(config.data, 'augmentation', None), 'random_rotate90', False), transform=None)
    shuffle = mode == 'train'
    if mode == 'test':
        batch_size = 1
    elif mode == 'val':
        batch_size = max(1, config.data.batch_size // 4)
    else:
        batch_size = config.data.batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.data.num_workers, pin_memory=True if config.experiment.device == 'cuda' else False, drop_last=mode == 'train')
    return loader

def build_test_dataloader_from_config(config: Config):
    dataset = DPDDTestDataset(root_dir=config.data.data_root, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers, pin_memory=True if config.experiment.device == 'cuda' else False)
    return loader