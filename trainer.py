import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import os
import time
from contextlib import nullcontext
from typing import Any, Mapping, Optional, Dict, Union
try:
    from torch.utils.tensorboard.writer import SummaryWriter as _SummaryWriter
    SummaryWriter = _SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
        SummaryWriter = _SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=0.001):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class FFTLoss(nn.Module):

    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x, norm='backward')
        y_fft = torch.fft.rfft2(y, norm='backward')
        return torch.mean(torch.abs(x_fft - y_fft))

class DualBranchTrainer:
    VALID_STAGES = ('physics_only', 'restoration_fixed_physics', 'joint', 'restoration_only')

    def __init__(self, restoration_net, physical_layer, lr_restoration, lr_optics, optimizer_type='adamw', weight_decay=0.0, lambda_sup=1.0, lambda_fft=0.1, lambda_coeff=0.05, lambda_smooth=0.1, lambda_image_reg=0.0, grad_clip_restoration=5.0, grad_clip_optics=1.0, stage_schedule=None, stage_weights=None, smoothness_grid_size=16, injection_grid_size=16, use_amp=True, amp_dtype='float16', device='cuda', accumulation_steps=4, keep_coeff_loss=True, keep_smooth_loss=True, tensorboard_dir=None):
        self.device = device
        self.restoration_net = restoration_net.to(device)
        self.physical_layer = physical_layer.to(device) if physical_layer is not None else None
        self.use_physical_layer = self.physical_layer is not None
        self.use_amp = bool(use_amp) and str(device).startswith('cuda') and torch.cuda.is_available()
        amp_dtype_str = str(amp_dtype).lower()
        if amp_dtype_str == 'bfloat16':
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
        self.aberration_net = self.physical_layer.aberration_net if self.physical_layer is not None else None
        self.base_lr_restoration = lr_restoration
        self.base_lr_optics = lr_optics
        opt_type = str(optimizer_type).lower()
        if opt_type == 'adamw':
            optimizer_cls = optim.AdamW
        elif opt_type == 'adam':
            optimizer_cls = optim.Adam
        else:
            raise ValueError(f'Unsupported optimizer type: {optimizer_type}')
        self.optimizer_W = optimizer_cls(self.restoration_net.parameters(), lr=lr_restoration, weight_decay=weight_decay)
        self.optimizer_Theta = None
        if self.aberration_net is not None:
            self.optimizer_Theta = optimizer_cls(self.aberration_net.parameters(), lr=lr_optics, weight_decay=weight_decay)
        self.lambda_sup = lambda_sup
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.lambda_image_reg = lambda_image_reg
        self.keep_coeff_loss = bool(keep_coeff_loss)
        self.keep_smooth_loss = bool(keep_smooth_loss)
        self.stage_weights = stage_weights if stage_weights is not None else {}
        default_schedule = {'stage1_epochs': 50, 'stage2_epochs': 200, 'stage3_epochs': 50}
        self.stage_schedule: Any = stage_schedule if stage_schedule is not None else default_schedule
        self.smoothness_grid_size = smoothness_grid_size
        self.injection_grid_size = injection_grid_size
        self.accumulation_steps = max(1, accumulation_steps)
        self.accumulation_counter = 0
        self.grad_clip_restoration = grad_clip_restoration
        self.grad_clip_optics = grad_clip_optics
        self.criterion_mse = nn.MSELoss()
        self.criterion_charbonnier = CharbonnierLoss(eps=0.001).to(device)
        self.criterion_fft = FFTLoss().to(device)
        self.lambda_fft = lambda_fft
        self._current_stage = 'joint' if self.use_physical_layer else 'restoration_only'
        self._previous_stage = None
        self._stage3_lr_halved = False
        self.writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE and (SummaryWriter is not None):
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f'[TensorBoard] Logging to: {tensorboard_dir}')
        elif tensorboard_dir and (not TENSORBOARD_AVAILABLE):
            print('[Warning] TensorBoard not available. Install with: pip install tensorboard')
        self.history = {'loss_total': [], 'loss_data': [], 'loss_sup': [], 'grad_norm_W': [], 'grad_norm_Theta': []}
        self.best_metrics = {'physics_only': {'reblur_mse': float('inf')}, 'restoration_fixed_physics': {'psnr': 0.0, 'ssim': 0.0}, 'joint': {'psnr': 0.0, 'combined': 0.0}, 'restoration_only': {'psnr': 0.0}}

    def reset_after_oom(self):
        self.accumulation_counter = 0
        self.optimizer_W.zero_grad(set_to_none=True)
        if self.optimizer_Theta is not None:
            self.optimizer_Theta.zero_grad(set_to_none=True)

    def _get_stage(self, epoch: int) -> str:
        if not self.use_physical_layer:
            return 'restoration_only'
        if isinstance(self.stage_schedule, Mapping):
            s1 = self.stage_schedule.get('stage1_epochs', 50)
            s2 = self.stage_schedule.get('stage2_epochs', 200)
        else:
            s1 = getattr(self.stage_schedule, 'stage1_epochs', 50)
            s2 = getattr(self.stage_schedule, 'stage2_epochs', 200)
        if epoch < s1:
            return 'physics_only'
        elif epoch < s1 + s2:
            return 'restoration_fixed_physics'
        return 'joint'

    def _adjust_learning_rate_for_stage3(self):
        if not self.use_physical_layer:
            return
        if self._stage3_lr_halved:
            return
        new_lr_W = self.base_lr_restoration / 2.0
        new_lr_Theta = self.base_lr_optics / 2.0
        for param_group in self.optimizer_W.param_groups:
            param_group['lr'] = new_lr_W
        if self.optimizer_Theta is not None:
            for param_group in self.optimizer_Theta.param_groups:
                param_group['lr'] = new_lr_Theta
        self._stage3_lr_halved = True
        print(f'[Stage 3] Learning rate halved: lr_restoration={new_lr_W:.2e}, lr_optics={new_lr_Theta:.2e}')

    def update_best_metrics(self, val_metrics: Dict[str, float], stage: str) -> Dict[str, bool]:
        is_best = {}
        if stage == 'physics_only':
            reblur_mse = val_metrics.get('Reblur_MSE', val_metrics.get('reblur_mse', float('inf')))
            if reblur_mse < self.best_metrics['physics_only']['reblur_mse']:
                self.best_metrics['physics_only']['reblur_mse'] = reblur_mse
                is_best['reblur_mse'] = True
            else:
                is_best['reblur_mse'] = False
        elif stage == 'restoration_fixed_physics':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            ssim = val_metrics.get('SSIM', val_metrics.get('ssim', 0.0))
            if psnr > self.best_metrics['restoration_fixed_physics']['psnr']:
                self.best_metrics['restoration_fixed_physics']['psnr'] = psnr
                is_best['psnr'] = True
            else:
                is_best['psnr'] = False
            if ssim > self.best_metrics['restoration_fixed_physics']['ssim']:
                self.best_metrics['restoration_fixed_physics']['ssim'] = ssim
                is_best['ssim'] = True
            else:
                is_best['ssim'] = False
        elif stage == 'joint':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            reblur_mse = val_metrics.get('Reblur_MSE', val_metrics.get('reblur_mse', float('inf')))
            combined = psnr - 10.0 * reblur_mse
            if psnr > self.best_metrics['joint']['psnr']:
                self.best_metrics['joint']['psnr'] = psnr
                is_best['psnr'] = True
            else:
                is_best['psnr'] = False
            if combined > self.best_metrics['joint']['combined']:
                self.best_metrics['joint']['combined'] = combined
                is_best['combined'] = True
            else:
                is_best['combined'] = False
        elif stage == 'restoration_only':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            if psnr > self.best_metrics.get('restoration_only', {}).get('psnr', 0.0):
                self.best_metrics.setdefault('restoration_only', {})['psnr'] = psnr
                is_best['psnr'] = True
            else:
                is_best['psnr'] = False
        return is_best

    def log_to_tensorboard(self, metrics: Dict[str, float], epoch: int, prefix: str='train'):
        if self.writer is None:
            return
        for (key, value) in metrics.items():
            if isinstance(value, (int, float)) and (not (isinstance(value, float) and value != value)):
                self.writer.add_scalar(f'{prefix}/{key}', value, epoch)

    def log_gradients_to_tensorboard(self, epoch: int):
        if self.writer is None:
            return
        for (name, param) in self.restoration_net.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/restoration/{name}', param.grad, epoch)
        if self.aberration_net is not None:
            for (name, param) in self.aberration_net.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/aberration/{name}', param.grad, epoch)

    def log_images_to_tensorboard(self, blur_img, sharp_img, restored_img, reblur_img, epoch: int):
        if self.writer is None:
            return
        self.writer.add_image('images/blur', blur_img[0].clamp(0, 1), epoch)
        self.writer.add_image('images/sharp_gt', sharp_img[0].clamp(0, 1), epoch)
        self.writer.add_image('images/restored', restored_img[0].clamp(0, 1), epoch)
        if reblur_img is not None:
            self.writer.add_image('images/reblur', reblur_img[0].clamp(0, 1), epoch)

    def close_tensorboard(self):
        if self.writer is not None:
            self.writer.close()

    def _get_stage_weights(self, stage: str):
        weights = {'w_data': 1.0, 'w_sup': 0.0, 'w_smooth': 0.0, 'w_coeff': 0.0, 'w_img_reg': 0.0}
        if stage in self.stage_weights:
            custom_weights = self.stage_weights[stage]
            weights.update(custom_weights)
            return weights
        if stage == 'physics_only':
            weights.update({'w_data': 1.0, 'w_sup': 0.0, 'w_smooth': self.lambda_smooth, 'w_coeff': self.lambda_coeff, 'w_img_reg': 0.0})
        elif stage == 'restoration_fixed_physics':
            weights.update({'w_data': 0.1, 'w_sup': 1.0, 'w_smooth': 0.0, 'w_coeff': 0.0, 'w_img_reg': self.lambda_image_reg})
        elif stage == 'joint':
            weights.update({'w_data': 0.5, 'w_sup': 1.0, 'w_smooth': self.lambda_smooth * 0.5, 'w_coeff': self.lambda_coeff * 0.2, 'w_img_reg': self.lambda_image_reg * 0.1})
        elif stage == 'restoration_only':
            weights.update({'w_data': 0.0, 'w_sup': 1.0, 'w_smooth': 0.0, 'w_coeff': 0.0, 'w_img_reg': self.lambda_image_reg})
        return weights

    def _set_trainable(self, stage: str):
        if stage == 'physics_only':
            for p in self.restoration_net.parameters():
                p.requires_grad = False
            if self.aberration_net is not None:
                for p in self.aberration_net.parameters():
                    p.requires_grad = True
            self.restoration_net.eval()
            if self.physical_layer is not None:
                self.physical_layer.train()
        elif stage == 'restoration_fixed_physics':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            if self.aberration_net is not None:
                for p in self.aberration_net.parameters():
                    p.requires_grad = False
            self.restoration_net.train()
            if self.physical_layer is not None:
                self.physical_layer.eval()
        elif stage == 'joint':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            if self.aberration_net is not None:
                for p in self.aberration_net.parameters():
                    p.requires_grad = True
            self.restoration_net.train()
            if self.physical_layer is not None:
                self.physical_layer.train()
        elif stage == 'restoration_only':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            self.restoration_net.train()
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

    def set_stage(self, stage: str):
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")
        self._current_stage = stage
        self._set_trainable(stage)

    def get_stage(self, epoch: int) -> str:
        return self._get_stage(epoch)

    def get_stage_weights(self, epoch: int):
        return self._get_stage_weights(self._get_stage(epoch))

    def _generate_coeffs_map(self, H, W, crop_info=None, batch_size=1, grid_size=None):
        if grid_size is None:
            grid_size = self.injection_grid_size
        if not self.use_physical_layer or self.physical_layer is None or getattr(self.restoration_net, 'n_coeffs', 0) == 0:
            return None
        return self.physical_layer.generate_coeffs_map(H, W, self.device, grid_size=grid_size, crop_info=crop_info, batch_size=batch_size)

    def train_step(self, Y_blur, X_gt, epoch, crop_info=None):
        current_stage = self._get_stage(epoch)
        if self._previous_stage is not None and self._previous_stage != current_stage:
            print(f'\n[Stage Transition] {self._previous_stage} -> {current_stage}')
            if current_stage == 'joint' and (not self._stage3_lr_halved):
                self._adjust_learning_rate_for_stage3()
        self._previous_stage = current_stage
        self._current_stage = current_stage
        self._set_trainable(current_stage)
        weights = self._get_stage_weights(current_stage)
        w_data = weights['w_data']
        w_sup = weights['w_sup']
        w_smooth = weights['w_smooth']
        w_coeff = weights['w_coeff']
        w_img_reg = weights['w_img_reg']
        if not self.keep_coeff_loss:
            w_coeff = 0.0
        if not self.keep_smooth_loss:
            w_smooth = 0.0
        Y_blur = Y_blur.to(self.device)
        X_gt = X_gt.to(self.device)
        if crop_info is not None:
            crop_info = crop_info.to(self.device)
        if self.accumulation_counter == 0:
            if not self.use_physical_layer:
                self.optimizer_W.zero_grad(set_to_none=True)
            elif current_stage == 'physics_only':
                if self.optimizer_Theta is not None:
                    self.optimizer_Theta.zero_grad(set_to_none=True)
            elif current_stage == 'restoration_fixed_physics':
                self.optimizer_W.zero_grad(set_to_none=True)
            else:
                self.optimizer_W.zero_grad(set_to_none=True)
                if self.optimizer_Theta is not None:
                    self.optimizer_Theta.zero_grad(set_to_none=True)
        loss_data = torch.tensor(0.0, device=self.device)
        loss_sup = torch.tensor(0.0, device=self.device)
        loss_coeff = torch.tensor(0.0, device=self.device)
        loss_smooth = torch.tensor(0.0, device=self.device)
        loss_image_reg = torch.tensor(0.0, device=self.device)
        amp_context = torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp) if self.use_amp else nullcontext()
        with amp_context:
            if not self.use_physical_layer:
                X_hat = self.restoration_net(Y_blur)
                loss_pixel = self.criterion_charbonnier(X_hat, X_gt)
                loss_freq = self.criterion_fft(X_hat, X_gt)
                loss_sup = loss_pixel + self.lambda_fft * loss_freq
                if w_img_reg > 0:
                    loss_image_reg = self.compute_image_tv_loss(X_hat)
            elif current_stage == 'physics_only':
                if self.physical_layer is None or self.aberration_net is None:
                    raise RuntimeError('physical_layer and aberration_net are required for physics_only stage')
                Y_reblur = self.physical_layer(X_gt, crop_info=crop_info)
                loss_data = self.criterion_mse(Y_reblur, Y_blur)
                if w_coeff > 0 or w_smooth > 0:
                    coords = self.physical_layer.get_patch_centers(Y_blur.shape[2], Y_blur.shape[3], self.device)
                    if coords.shape[0] > 64:
                        indices = torch.randperm(coords.shape[0])[:64]
                        coords = coords[indices]
                    coeffs = self.aberration_net(coords)
                    if w_coeff > 0:
                        loss_coeff = torch.mean(coeffs ** 2)
                    if w_smooth > 0:
                        loss_smooth = self.physical_layer.compute_coefficient_smoothness(self.smoothness_grid_size)
            else:
                if self.physical_layer is None or self.aberration_net is None:
                    raise RuntimeError('physical_layer and aberration_net are required for this stage')
                coeffs_map = self._generate_coeffs_map(Y_blur.shape[2], Y_blur.shape[3], crop_info=crop_info, batch_size=Y_blur.shape[0])
                X_hat = self.restoration_net(Y_blur, coeffs_map=coeffs_map)
                Y_reblur = self.physical_layer(X_hat, crop_info=crop_info)
                loss_data = self.criterion_mse(Y_reblur, Y_blur)
                loss_pixel = self.criterion_charbonnier(X_hat, X_gt)
                loss_freq = self.criterion_fft(X_hat, X_gt)
                loss_sup = loss_pixel + self.lambda_fft * loss_freq
                if w_img_reg > 0:
                    loss_image_reg = self.compute_image_tv_loss(X_hat)
                if current_stage == 'joint' and (w_coeff > 0 or w_smooth > 0):
                    coords = self.physical_layer.get_patch_centers(Y_blur.shape[2], Y_blur.shape[3], self.device)
                    if coords.shape[0] > 64:
                        indices = torch.randperm(coords.shape[0])[:64]
                        coords = coords[indices]
                    coeffs = self.aberration_net(coords)
                    if w_coeff > 0:
                        loss_coeff = torch.mean(coeffs ** 2)
                    if w_smooth > 0:
                        loss_smooth = self.physical_layer.compute_coefficient_smoothness(self.smoothness_grid_size)
        loss_data_w = w_data * loss_data
        loss_sup_w = w_sup * loss_sup
        loss_coeff_w = w_coeff * loss_coeff
        loss_smooth_w = w_smooth * loss_smooth
        loss_image_reg_w = w_img_reg * loss_image_reg
        total_loss = loss_data_w + loss_sup_w + loss_coeff_w + loss_smooth_w + loss_image_reg_w
        scaled_loss = total_loss / self.accumulation_steps
        if self.use_grad_scaler:
            self.grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self.accumulation_counter += 1
        should_step = self.accumulation_counter >= self.accumulation_steps
        gn_W = torch.tensor(0.0, device=self.device)
        gn_Theta = torch.tensor(0.0, device=self.device)
        if should_step:
            has_grad_W = any((p.grad is not None for p in self.restoration_net.parameters()))
            has_grad_Theta = self.aberration_net is not None and any((p.grad is not None for p in self.aberration_net.parameters()))
            if not self.use_physical_layer:
                if self.use_grad_scaler and has_grad_W:
                    self.grad_scaler.unscale_(self.optimizer_W)
                if has_grad_W:
                    gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), self.grad_clip_restoration)
                    if self.use_grad_scaler:
                        self.grad_scaler.step(self.optimizer_W)
                        self.grad_scaler.update()
                    else:
                        self.optimizer_W.step()
            elif current_stage == 'physics_only':
                if self.aberration_net is None:
                    raise RuntimeError('aberration_net is required for physics_only stage')
                if self.optimizer_Theta is None:
                    raise RuntimeError('optimizer_Theta is required for physics_only stage')
                if self.use_grad_scaler and has_grad_Theta:
                    self.grad_scaler.unscale_(self.optimizer_Theta)
                if has_grad_Theta:
                    gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), self.grad_clip_optics)
                    if self.use_grad_scaler:
                        self.grad_scaler.step(self.optimizer_Theta)
                        self.grad_scaler.update()
                    else:
                        self.optimizer_Theta.step()
            elif current_stage == 'restoration_fixed_physics':
                if self.use_grad_scaler and has_grad_W:
                    self.grad_scaler.unscale_(self.optimizer_W)
                if has_grad_W:
                    gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), self.grad_clip_restoration)
                    if self.use_grad_scaler:
                        self.grad_scaler.step(self.optimizer_W)
                        self.grad_scaler.update()
                    else:
                        self.optimizer_W.step()
            else:
                if self.aberration_net is None:
                    raise RuntimeError('aberration_net is required for joint stage')
                if self.optimizer_Theta is None:
                    raise RuntimeError('optimizer_Theta is required for joint stage')
                if self.use_grad_scaler and has_grad_W:
                    self.grad_scaler.unscale_(self.optimizer_W)
                if self.use_grad_scaler and has_grad_Theta:
                    self.grad_scaler.unscale_(self.optimizer_Theta)
                if has_grad_W:
                    gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), self.grad_clip_restoration)
                if has_grad_Theta:
                    gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), self.grad_clip_optics)
                if self.use_grad_scaler:
                    if has_grad_W:
                        self.grad_scaler.step(self.optimizer_W)
                    if has_grad_Theta:
                        self.grad_scaler.step(self.optimizer_Theta)
                    if has_grad_W or has_grad_Theta:
                        self.grad_scaler.update()
                else:
                    if has_grad_W:
                        self.optimizer_W.step()
                    if has_grad_Theta:
                        self.optimizer_Theta.step()
            self.accumulation_counter = 0
            self.history['loss_total'].append(total_loss.item())
            self.history['loss_data'].append(loss_data_w.item())
            self.history['grad_norm_W'].append(gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W)
            self.history['grad_norm_Theta'].append(gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta)
        metrics = {'loss': total_loss.item(), 'loss_data': loss_data_w.item(), 'loss_sup': loss_sup_w.item(), 'loss_coeff': loss_coeff_w.item(), 'loss_smooth': loss_smooth_w.item(), 'loss_image_reg': loss_image_reg_w.item(), 'loss_data_raw': loss_data.item(), 'loss_sup_raw': loss_sup.item(), 'loss_coeff_raw': loss_coeff.item(), 'loss_smooth_raw': loss_smooth.item(), 'loss_image_reg_raw': loss_image_reg.item(), 'grad_W': gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W, 'grad_Theta': gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta, 'stage': current_stage}
        if self.physical_layer is not None and hasattr(self.physical_layer, 'get_newbp_stats'):
            nb_stats = self.physical_layer.get_newbp_stats()
            for k, v in nb_stats.items():
                metrics[f'newbp/{k}'] = float(v)
        return metrics

    def compute_image_tv_loss(self, img):
        (B, C, H, W) = img.shape
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return dy + dx

    def save_checkpoint(self, path, epoch=None, stage=None, val_metrics=None):
        restoration_state = {k: v for (k, v) in self.restoration_net.state_dict().items() if 'total_ops' not in k and 'total_params' not in k}
        checkpoint = {'restoration_net': restoration_state, 'optimizer_W': self.optimizer_W.state_dict(), 'best_metrics': self.best_metrics}
        if self.aberration_net is not None:
            aberration_state = {k: v for (k, v) in self.aberration_net.state_dict().items() if 'total_ops' not in k and 'total_params' not in k}
            checkpoint['aberration_net'] = aberration_state
        if self.physical_layer is not None:
            physical_state = {k: v for (k, v) in self.physical_layer.state_dict().items() if 'total_ops' not in k and 'total_params' not in k}
            checkpoint['physical_layer'] = physical_state
        if self.optimizer_Theta is not None:
            checkpoint['optimizer_Theta'] = self.optimizer_Theta.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if stage is not None:
            checkpoint['stage'] = stage
        if val_metrics is not None:
            checkpoint['val_metrics'] = val_metrics
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        restoration_state = {k: v for (k, v) in checkpoint['restoration_net'].items() if 'total_ops' not in k and 'total_params' not in k}
        self.restoration_net.load_state_dict(restoration_state, strict=True)
        if self.aberration_net is not None and 'aberration_net' in checkpoint:
            aberration_state = {k: v for (k, v) in checkpoint['aberration_net'].items() if 'total_ops' not in k and 'total_params' not in k}
            self.aberration_net.load_state_dict(aberration_state, strict=False)
        if self.physical_layer is not None and 'physical_layer' in checkpoint:
            physical_state = {k: v for (k, v) in checkpoint['physical_layer'].items() if 'total_ops' not in k and 'total_params' not in k}
            self.physical_layer.load_state_dict(physical_state, strict=False)
        if load_optimizer:
            if 'optimizer_W' in checkpoint:
                self.optimizer_W.load_state_dict(checkpoint['optimizer_W'])
            if self.optimizer_Theta is not None and 'optimizer_Theta' in checkpoint:
                self.optimizer_Theta.load_state_dict(checkpoint['optimizer_Theta'])
        if 'best_metrics' in checkpoint:
            self.best_metrics = checkpoint['best_metrics']
        return {'epoch': checkpoint.get('epoch'), 'stage': checkpoint.get('stage'), 'val_metrics': checkpoint.get('val_metrics')}

    def get_current_lr(self) -> Dict[str, float]:
        lr_optics = float('nan')
        if self.optimizer_Theta is not None:
            lr_optics = self.optimizer_Theta.param_groups[0]['lr']
        return {'lr_restoration': self.optimizer_W.param_groups[0]['lr'], 'lr_optics': lr_optics}