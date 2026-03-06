import math
import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerformanceEvaluator:

    def __init__(self, device: str='cuda', ssim_window: int=11, ssim_sigma: float=1.5):
        self.device = device
        self.ssim_window = ssim_window
        self.ssim_sigma = ssim_sigma
        self._lpips = None
        self._lpips_available = False
        try:
            import lpips
            self._lpips = lpips.LPIPS(net='alex').to(device)
            self._lpips_available = True
        except Exception:
            self._lpips = None
            self._lpips_available = False

    @staticmethod
    def _psnr(x: torch.Tensor, y: torch.Tensor, max_val: float=1.0, eps: float=1e-08) -> torch.Tensor:
        mse = F.mse_loss(x, y, reduction='mean')
        psnr = 10.0 * torch.log10(max_val ** 2 / (mse + eps))
        return psnr

    @staticmethod
    def _mae(x: torch.Tensor, y: torch.Tensor, scale: float=255.0) -> torch.Tensor:
        return torch.mean(torch.abs(x - y)) * scale

    @staticmethod
    def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window_2d = window_2d / window_2d.sum()
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, max_val: float=1.0) -> torch.Tensor:
        (b, c, _, _) = x.shape
        window = self._gaussian_window(self.ssim_window, self.ssim_sigma, c, x.device, x.dtype)
        mu_x = F.conv2d(x, window, padding=self.ssim_window // 2, groups=c)
        mu_y = F.conv2d(y, window, padding=self.ssim_window // 2, groups=c)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(x * x, window, padding=self.ssim_window // 2, groups=c) - mu_x2
        sigma_y = F.conv2d(y * y, window, padding=self.ssim_window // 2, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.ssim_window // 2, groups=c) - mu_xy
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        ssim_map = (2 * mu_xy + c1) * (2 * sigma_xy + c2) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
        return ssim_map.mean()

    def _lpips_score(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._lpips_available or self._lpips is None:
            return None
        x_norm = (x * 2.0 - 1.0).clamp(-1, 1)
        y_norm = (y * 2.0 - 1.0).clamp(-1, 1)
        return self._lpips(x_norm, y_norm).mean()

    @staticmethod
    def _count_parameters(*models: nn.Module) -> float:
        total = 0
        for model in models:
            total += sum((p.numel() for p in model.parameters()))
        return total / 1000000.0

    @staticmethod
    def _remove_profile_buffers(model: nn.Module):
        target_keys = ('total_ops', 'total_params')
        for module in model.modules():
            for key in target_keys:
                if hasattr(module, '_buffers') and key in module._buffers:
                    module._buffers.pop(key, None)
                if hasattr(module, key):
                    try:
                        delattr(module, key)
                    except Exception:
                        pass

    @staticmethod
    def _try_flops(model: nn.Module, device: str, input_shape=(1, 3, 1024, 1024)) -> Optional[float]:
        try:
            from thop import profile
            PerformanceEvaluator._remove_profile_buffers(model)
            dummy = torch.randn(*input_shape, device=device)
            with torch.no_grad():
                (macs, _) = profile(model, inputs=(dummy,), verbose=False)
            return macs / 1000000000.0
        except Exception:
            return None
        finally:
            PerformanceEvaluator._remove_profile_buffers(model)

    @staticmethod
    def _build_injection_aware_benchmark_model(restoration_net: nn.Module, physical_layer: Optional[nn.Module], device: str, injection_grid_size: int=16) -> nn.Module:
        if physical_layer is None or not hasattr(physical_layer, 'generate_coeffs_map'):
            return restoration_net

        class _InjectionAwareWrapper(nn.Module):

            def __init__(self, restore_model, phys_model, grid_size, run_device):
                super().__init__()
                self.restore_model = restore_model
                self.phys_model = phys_model
                self.grid_size = grid_size
                self.run_device = run_device

            def forward(self, x):
                needs_coeffs = getattr(self.restore_model, 'n_coeffs', 0) > 0
                if not needs_coeffs:
                    return self.restore_model(x)
                (b, _, h, w) = x.shape
                coeffs_map = None
                if needs_coeffs:
                    coeffs_map = self.phys_model.generate_coeffs_map(h, w, self.run_device, grid_size=self.grid_size, crop_info=None, batch_size=b)
                return self.restore_model(x, coeffs_map=coeffs_map)
        wrapper = _InjectionAwareWrapper(restoration_net, physical_layer, injection_grid_size, device)
        wrapper.eval()
        return wrapper

    @staticmethod
    def _measure_inference_time(model: nn.Module, device: str, input_shape=(1, 3, 1024, 1024), warmup: int=5, repeat: int=20) -> float:
        model.eval()
        dummy = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
            if device.startswith('cuda') and torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record(stream=torch.cuda.current_stream())
                for _ in range(repeat):
                    _ = model(dummy)
                ender.record(stream=torch.cuda.current_stream())
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)
                return elapsed / repeat
            else:
                start = time.perf_counter()
                for _ in range(repeat):
                    _ = model(dummy)
                end = time.perf_counter()
                return (end - start) * 1000.0 / repeat

    def evaluate(self, restoration_net: nn.Module, physical_layer: Optional[nn.Module], val_loader, device: str, smoothness_grid_size: int=16, injection_grid_size: int=16) -> Dict[str, float]:
        restoration_net.eval()
        use_physical_layer = physical_layer is not None
        requires_physics_injection = getattr(restoration_net, 'n_coeffs', 0) > 0
        if use_physical_layer:
            physical_layer.eval()
        psnr_total = 0.0
        ssim_total = 0.0
        mae_total = 0.0
        lpips_total = 0.0
        reblur_total = 0.0
        n = 0
        lpips_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    blur = batch['blur'].to(device)
                    sharp = batch['sharp'].to(device)
                    crop_info = batch.get('crop_info', None)
                    if crop_info is not None:
                        crop_info = crop_info.to(device)
                else:
                    (blur, sharp) = batch
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    crop_info = None
                coeffs_map = None
                if use_physical_layer and getattr(restoration_net, 'n_coeffs', 0) > 0:
                    (B, _, H, W) = blur.shape
                    coeffs_map = physical_layer.generate_coeffs_map(H, W, device, grid_size=injection_grid_size, crop_info=crop_info, batch_size=B)
                x_hat = restoration_net(blur, coeffs_map=coeffs_map)
                if x_hat.shape[1] == 1 and sharp.shape[1] == 3:
                    x_hat = x_hat.repeat(1, 3, 1, 1)
                if sharp.shape[1] == 1 and x_hat.shape[1] == 3:
                    sharp = sharp.repeat(1, 3, 1, 1)
                psnr_total += self._psnr(x_hat, sharp).item()
                ssim_total += self._ssim(x_hat, sharp).item()
                mae_total += self._mae(x_hat, sharp).item()
                lp = self._lpips_score(x_hat, sharp)
                if lp is not None:
                    lpips_total += lp.item()
                    lpips_count += 1
                if use_physical_layer:
                    y_reblur = physical_layer(x_hat, crop_info=crop_info)
                    reblur_total += F.mse_loss(y_reblur, blur).item()
                n += 1
        smoothness = float('nan')
        if use_physical_layer and hasattr(physical_layer, 'compute_coefficient_smoothness'):
            with torch.no_grad():
                smoothness = physical_layer.compute_coefficient_smoothness(smoothness_grid_size).item()
        if use_physical_layer:
            params_m = self._count_parameters(restoration_net, physical_layer)
        else:
            params_m = self._count_parameters(restoration_net)
        flops_gmacs = None
        infer_ms = float('nan')
        benchmark_model = restoration_net
        can_benchmark = True
        if requires_physics_injection:
            if not use_physical_layer:
                can_benchmark = False
            else:
                benchmark_model = self._build_injection_aware_benchmark_model(restoration_net, physical_layer, device, injection_grid_size=injection_grid_size)
        if can_benchmark:
            try:
                flops_gmacs = self._try_flops(benchmark_model, device)
            except Exception:
                flops_gmacs = None
            try:
                infer_ms = self._measure_inference_time(benchmark_model, device)
            except Exception:
                infer_ms = float('nan')
        metrics = {'PSNR': psnr_total / max(n, 1), 'SSIM': ssim_total / max(n, 1), 'MAE': mae_total / max(n, 1), 'LPIPS': lpips_total / lpips_count if lpips_count > 0 else float('nan'), 'Reblur_MSE': reblur_total / max(n, 1) if use_physical_layer else float('nan'), 'PSF_Smoothness': smoothness, 'Params(M)': params_m, 'FLOPs(GMACs)': flops_gmacs if flops_gmacs is not None else float('nan'), 'Inference(ms)': infer_ms}
        return metrics

    @staticmethod
    def evaluate_model(restoration_net: nn.Module, physical_layer: Optional[nn.Module], val_loader, device: str, smoothness_grid_size: int=16) -> Dict[str, float]:
        evaluator = PerformanceEvaluator(device=device)
        return evaluator.evaluate(restoration_net, physical_layer, val_loader, device, smoothness_grid_size)

    @staticmethod
    def evaluate_stage1(physical_layer: nn.Module, val_loader, device: str, smoothness_grid_size: int=16) -> Dict[str, float]:
        physical_layer.eval()
        reblur_total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    blur = batch['blur'].to(device)
                    sharp = batch['sharp'].to(device)
                    crop_info = batch.get('crop_info', None)
                    if crop_info is not None:
                        crop_info = crop_info.to(device)
                else:
                    (blur, sharp) = batch
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    crop_info = None
                y_reblur = physical_layer(sharp, crop_info=crop_info)
                reblur_total += F.mse_loss(y_reblur, blur).item()
                n += 1
        smoothness = float('nan')
        if hasattr(physical_layer, 'compute_coefficient_smoothness'):
            with torch.no_grad():
                smoothness = physical_layer.compute_coefficient_smoothness(smoothness_grid_size).item()
        return {'Reblur_MSE': reblur_total / max(n, 1), 'PSF_Smoothness': smoothness}

    def evaluate_full_resolution(self, restoration_net: nn.Module, physical_layer: Optional[nn.Module], test_loader, device: str, injection_grid_size: int=16) -> Tuple[Dict[str, float], list]:
        restoration_net.eval()
        use_physical_layer = physical_layer is not None
        if use_physical_layer:
            physical_layer.eval()
        results = []
        psnr_total = 0.0
        ssim_total = 0.0
        mae_total = 0.0
        lpips_total = 0.0
        reblur_total = 0.0
        n = 0
        lpips_count = 0
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    blur = batch['blur'].to(device)
                    sharp = batch['sharp'].to(device)
                    crop_info = batch.get('crop_info', None)
                    filename = batch.get('filename', [f'image_{n}'])[0]
                    if crop_info is not None:
                        crop_info = crop_info.to(device)
                else:
                    (blur, sharp) = batch
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    crop_info = None
                    filename = f'image_{n}'
                coeffs_map = None
                if use_physical_layer and getattr(restoration_net, 'n_coeffs', 0) > 0:
                    (B, _, H, W) = blur.shape
                    coeffs_map = physical_layer.generate_coeffs_map(H, W, device, grid_size=injection_grid_size, crop_info=crop_info, batch_size=B)
                x_hat = restoration_net(blur, coeffs_map=coeffs_map)
                psnr = self._psnr(x_hat, sharp).item()
                ssim = self._ssim(x_hat, sharp).item()
                mae = self._mae(x_hat, sharp).item()
                lp = self._lpips_score(x_hat, sharp)
                lpips_val = lp.item() if lp is not None else float('nan')
                if use_physical_layer:
                    y_reblur = physical_layer(x_hat, crop_info=crop_info)
                    reblur_mse = F.mse_loss(y_reblur, blur).item()
                else:
                    reblur_mse = float('nan')
                results.append({'filename': filename, 'PSNR': psnr, 'SSIM': ssim, 'MAE': mae, 'LPIPS': lpips_val, 'Reblur_MSE': reblur_mse})
                psnr_total += psnr
                ssim_total += ssim
                mae_total += mae
                if not math.isnan(lpips_val):
                    lpips_total += lpips_val
                    lpips_count += 1
                if use_physical_layer:
                    reblur_total += reblur_mse
                n += 1
        avg_metrics = {'PSNR': psnr_total / max(n, 1), 'SSIM': ssim_total / max(n, 1), 'MAE': mae_total / max(n, 1), 'LPIPS': lpips_total / lpips_count if lpips_count > 0 else float('nan'), 'Reblur_MSE': reblur_total / max(n, 1) if use_physical_layer else float('nan'), 'Num_Images': n}
        return (avg_metrics, results)

def calculate_mae(prediction: torch.Tensor, target: torch.Tensor, scale: float=255.0) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f'Shape mismatch: prediction {prediction.shape} vs target {target.shape}')
    if prediction.ndim not in (3, 4):
        raise ValueError(f'Expected 3D (C,H,W) or 4D (B,C,H,W) tensor, got {prediction.ndim}D')
    return torch.mean(torch.abs(prediction - target)) * scale