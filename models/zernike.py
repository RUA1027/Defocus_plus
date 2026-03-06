import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
import math
from contextlib import nullcontext

def noll_to_nm(j):
    if j < 1:
        raise ValueError(f'Noll index must be >= 1, got {j}')
    n = int((-1.0 + (8.0 * j - 7) ** 0.5) / 2.0)
    if (n + 1) * (n + 2) // 2 < j:
        n += 1
    k = j - n * (n + 1) // 2
    if n % 2 == 0:
        if k == 1:
            m = 0
        else:
            abs_m = 2 * ((k - 2) // 2 + 1)
            if j % 2 == 0:
                m = abs_m
            else:
                m = -abs_m
    else:
        abs_m = 2 * ((k - 1) // 2) + 1
        if j % 2 == 0:
            m = abs_m
        else:
            m = -abs_m
    return (n, m)

def zernike_radial(n, m, rho):
    m = abs(m)
    if (n - m) % 2 != 0:
        return torch.zeros_like(rho)
    result = torch.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        coeff = (-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s))
        result = result + coeff * rho ** (n - 2 * s)
    return result

class ZernikeBasis(nn.Module):

    def __init__(self, n_modes=36, grid_size=64, device='cpu'):
        super().__init__()
        self.n_modes = n_modes
        self.grid_size = grid_size
        self.device = device
        u = torch.linspace(-1, 1, grid_size, device=device)
        v = torch.linspace(-1, 1, grid_size, device=device)
        (v_grid, u_grid) = torch.meshgrid(v, u, indexing='ij')
        rho = torch.sqrt(u_grid ** 2 + v_grid ** 2)
        theta = torch.atan2(v_grid, u_grid)
        self.mask = (rho <= 1.0).float()
        rho = rho * self.mask
        basis = []
        for j in range(1, n_modes + 1):
            (n, m) = noll_to_nm(j)
            if m == 0:
                norm = np.sqrt(n + 1)
                term = zernike_radial(n, m, rho)
            else:
                norm = np.sqrt(2 * (n + 1))
                R = zernike_radial(n, m, rho)
                if j % 2 == 0:
                    term = R * torch.cos(abs(m) * theta)
                else:
                    term = R * torch.sin(abs(m) * theta)
            Z = torch.tensor(norm, device=device) * term
            basis.append(Z)
        self.basis = torch.stack(basis, dim=0)
        self.basis = self.basis * self.mask.unsqueeze(0)
        self.register_buffer('zernike_basis', self.basis)
        self.register_buffer('aperture_mask', self.mask)

    def forward(self, coefficients):
        return torch.einsum('bn,nhw->bhw', coefficients, self.zernike_basis)

class DifferentiableZernikeGenerator(nn.Module):

    def __init__(self, n_modes, pupil_size, kernel_size, oversample_factor=2, wavelengths=None, ref_wavelength=5.5e-07, device='cpu', learnable_wavelengths=False, wavelength_bounds=None):
        super().__init__()
        self.n_modes = n_modes
        self.pupil_size = pupil_size
        self.kernel_size = kernel_size
        self.oversample_factor = oversample_factor
        self.learnable_wavelengths = learnable_wavelengths
        self.wavelength_bounds = wavelength_bounds if wavelength_bounds is not None else [4e-07, 7e-07]
        self.wavelengths = wavelengths if wavelengths is not None else [ref_wavelength]
        self.ref_wavelength = ref_wavelength
        if not isinstance(self.wavelength_bounds, (list, tuple)) or len(self.wavelength_bounds) != 2 or self.wavelength_bounds[0] >= self.wavelength_bounds[1]:
            raise ValueError(f'Invalid wavelength_bounds: {self.wavelength_bounds}')
        if self.learnable_wavelengths:
            (min_w, max_w) = self.wavelength_bounds
            wavelengths_tensor = torch.tensor(self.wavelengths, device=device, dtype=torch.float32)
            denom = max_w - min_w
            if denom <= 0:
                raise ValueError(f'Invalid wavelength bounds: {self.wavelength_bounds}')
            eps = 1e-06
            normalized = (wavelengths_tensor - min_w) / denom
            normalized = torch.clamp(normalized, eps, 1.0 - eps)
            raw = torch.log(normalized / (1.0 - normalized))
            self.raw_wavelengths = nn.Parameter(raw)
        else:
            self.register_buffer('wavelengths_tensor', torch.tensor(self.wavelengths, device=device, dtype=torch.float32))
        if kernel_size % 2 == 0:
            raise ValueError(f'Kernel size must be odd to ensure physical alignment, got {kernel_size}')
        self.basis = ZernikeBasis(n_modes, pupil_size, device)

    def forward(self, coefficients):
        autocast_ctx = torch.autocast(device_type=coefficients.device.type, enabled=False) if coefficients.device.type in ('cuda', 'cpu') else nullcontext()
        with autocast_ctx:
            coeffs_fp32 = coefficients.float()
            phi_ref = 2 * torch.pi * self.basis(coeffs_fp32).float()
        psf_channels = []
        wavelengths = self._get_wavelengths()
        for lam in wavelengths:
            with autocast_ctx:
                lam_fp32 = lam.float() if isinstance(lam, torch.Tensor) else torch.tensor(lam, device=phi_ref.device, dtype=torch.float32)
                scale = torch.tensor(self.ref_wavelength, device=phi_ref.device, dtype=torch.float32) / lam_fp32
                phi = phi_ref * scale
                A = self.basis.aperture_mask.float()
                pupil = A * torch.exp(1j * phi)
            '\n原始 FFT (64×64):\n- 分辨率低\n- PSF 边界有混叠 (aliasing)\n过采样 (128×128):\n- 2 倍分辨率\n- 减少混叠\n- 更准确的 PSF 边界\n实际应用:\n原始 → 过采样 → FFT → 下采样 → 裁剪\n64    128      128    64       33\n            '
            target_size = self.pupil_size * self.oversample_factor
            if target_size % 2 == 0:
                target_size += 1
            pad_total = target_size - self.pupil_size
            half_pad = pad_total // 2
            (p_l, p_r, p_t, p_b) = (half_pad, half_pad, half_pad, half_pad)
            pupil_padded = F.pad(pupil, (p_l, p_r, p_t, p_b), mode='constant', value=0)
            complex_field = torch.fft.ifftshift(pupil_padded, dim=(-2, -1))
            psf_complex = torch.fft.fft2(complex_field)
            psf_complex = torch.fft.fftshift(psf_complex, dim=(-2, -1))
            psf_high_res = psf_complex.abs() ** 2
            if self.oversample_factor > 1:
                psf = F.avg_pool2d(psf_high_res.unsqueeze(1), kernel_size=self.oversample_factor, stride=self.oversample_factor).squeeze(1)
            else:
                psf = psf_high_res
            psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-08)
            G = self.pupil_size
            K = self.kernel_size
            if K > G:
                raise ValueError(f'Kernel size {K} > Pupil size {G}')
            start = G // 2 - K // 2
            end = start + K
            psf_cropped = psf[:, start:end, start:end]
            psf_cropped = psf_cropped / (psf_cropped.sum(dim=(-2, -1), keepdim=True) + 1e-08)
            psf_channels.append(psf_cropped)
        return torch.stack(psf_channels, dim=1)

    def _get_wavelengths(self):
        if self.learnable_wavelengths:
            (min_w, max_w) = self.wavelength_bounds
            scale = torch.sigmoid(self.raw_wavelengths)
            return min_w + (max_w - min_w) * scale
        return self.wavelengths_tensor