# 将 Zernike 像差系数 转换为 PSF (点扩散函数) 卷积核
'''
Zernike 系数 [a₁, a₂, ..., a₃₆]
        ↓ (物理光学计算)
波前相位 φ(x, y)
        ↓ (FFT)
点扩散函数 PSF (卷积核)
'''
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
import math
from contextlib import nullcontext
'''
Zernike 系数 (从 AberrationNet)
    [B*N, 36]
        │
        ▼
        │
   ┌────────────────────────────────────────────┐
   │  DifferentiableZernikeGenerator.forward()  │
   └────────────────────────────────────────────┘
        │
        ├─ 步骤 1: ZernikeBasis(系数)
        │          ↓ 计算波前相位 φ [B, 64, 64]
        │
        ├─ 步骤 2: 多波长处理
        │  ┌─ 红光 (650nm)
        │  │   └─ φ_scale × (λ_ref / λ_R)
        │  ├─ 绿光 (550nm) [参考]
        │  │   └─ φ_scale × (λ_ref / λ_G)
        │  └─ 蓝光 (450nm)
        │      └─ φ_scale × (λ_ref / λ_B)
        │
        ├─ 步骤 3: 瞳孔函数
        │   P = A × exp(i×φ)
        │
        ├─ 步骤 4: 过采样 (2×)
        │   64 → 128
        │
        ├─ 步骤 5: FFT (Fourier Transform)
        │   |FFT(P)| → PSF 空间
        │
        ├─ 步骤 6: 下采样 (回到原大小)
        │   128 → 64 → 33×33
        │
        └─ 步骤 7: 归一化
            ∫∫ PSF(x,y) dxdy = 1
        
        ▼
    PSF 卷积核 [B*N, C, 33, 33]
    (如 [128, 3, 33, 33] for RGB)
        │
        ▼
    送入 physical_layer 进行卷积
'''
def noll_to_nm(j):
    """
    索引转换: Noll index j -> (n, m)
    通用实现，支持任意高阶 Noll 索引。
    
    参考: Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence."
    
    j=1:  (0,0)   Piston
    j=2:  (1,1)   Tilt-X          j=3:  (1,-1)  Tilt-Y
    j=4:  (2,0)   Defocus
    j=5:  (2,-2)  Astigmatism     j=6:  (2,2)   Astigmatism
    j=7:  (3,-1)  Coma            j=8:  (3,1)   Coma
    j=9:  (3,-3)  Trefoil         j=10: (3,3)   Trefoil
    j=11: (4,0)   Spherical
    ...以此类推到任意高阶...
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    
    # 计算径向阶数 n: 满足 n(n+1)/2 < j <= (n+1)(n+2)/2
    n = int((-1.0 + (8.0 * j - 7) ** 0.5) / 2.0)
    if (n + 1) * (n + 2) // 2 < j:
        n += 1
    
    # 计算 |m|
    # 在第 n 阶内的余数
    k = j - n * (n + 1) // 2  # 1-indexed
    
    # n 阶内 |m| 的排列: 按照 Noll 约定
    # 对于偶数 n: m 可取 0, ±2, ±4, ..., ±n
    # 对于奇数 n: m 可取 ±1, ±3, ±5, ..., ±n
    # k=1 对应最小的 |m|
    
    # |m| 从 n%2 开始，步长 2:  n%2, n%2+2, n%2+4, …
    # k=1 -> |m|=n%2; k=2,3 -> |m|=n%2+2; k=4,5 -> |m|=n%2+4; ...
    # 但 |m|=0 只占 1 个位置，|m|>0 占 2 个位置
    
    if n % 2 == 0:
        # m=0 占 k=1; |m|=2 占 k=2,3; |m|=4 占 k=4,5; ...
        if k == 1:
            m = 0
        else:
            abs_m = 2 * ((k - 2) // 2 + 1)
            if j % 2 == 0:
                m = abs_m
            else:
                m = -abs_m
    else:
        # m=1 占 k=1,2; m=3 占 k=3,4; m=5 占 k=5,6; ...
        abs_m = 2 * ((k - 1) // 2) + 1
        if j % 2 == 0:
            m = abs_m
        else:
            m = -abs_m
    
    return (n, m)

def zernike_radial(n, m, rho):
    """
    Compute Radial Zernike Polynomial R_n^|m|(rho).
    通用实现，基于解析公式支持任意阶数。
    
    R_n^m(ρ) = Σ_{s=0}^{(n-m)/2} (-1)^s * (n-s)! / (s! * ((n+m)/2 - s)! * ((n-m)/2 - s)!) * ρ^(n-2s)
    """
    m = abs(m)
    
    if (n - m) % 2 != 0:
        return torch.zeros_like(rho)
    
    result = torch.zeros_like(rho)
    
    for s in range((n - m) // 2 + 1):
        coeff = (
            ((-1) ** s) * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + m) // 2 - s)
                * math.factorial((n - m) // 2 - s)
            )
        )
        result = result + coeff * rho ** (n - 2 * s)
    
    return result

class ZernikeBasis(nn.Module):
    """
    Precomputes and stores Zernike basis functions on a grid.
    预计算所有 n_modes 个 Zernike 基函数在空间网格上的值，避免重复计算。
    
    归一化约定 (Noll RMS=1):
      m=0:  norm = √(n+1)
      m≠0:  norm = √(2(n+1))
    角向函数:
      偶数 j → cos(|m|θ)  (正 m 分量)
      奇数 j → sin(|m|θ)  (负 m 分量)
    """
    def __init__(self, n_modes=36, grid_size=64, device='cpu'):
        super().__init__()
        self.n_modes = n_modes
        self.grid_size = grid_size
        self.device = device
        
        # Create coordinate grid
        u = torch.linspace(-1, 1, grid_size, device=device)
        v = torch.linspace(-1, 1, grid_size, device=device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        rho = torch.sqrt(u_grid**2 + v_grid**2)
        theta = torch.atan2(v_grid, u_grid)
        
        # Aperture mask
        self.mask = (rho <= 1.0).float()
        rho = rho * self.mask # Zero out rho outside
        
        basis = []
        # Precompute Z1 to Zn
        for j in range(1, n_modes + 1):
            n, m = noll_to_nm(j)
            
            # Normalization factor (Noll convention usually has sqrt(n+1) or similar?)
            # Noll: RMS = 1. 
            # Z_polar = sqrt(n+1) * R_nm(rho) * ...
            #   if m=0: * 1
            #   if m!=0, even j: * sqrt(2) cos(m theta)
            #   if m!=0, odd j:  * sqrt(2) sin(|m| theta)
            
            # Let's verify normalization from reference or standard Noll.
            # Ref j=4 (defocus): sqrt(3)*(2rho^2-1). n=2, sqrt(n+1)=sqrt(3). Correct.
            # Ref j=5 (astig): sqrt(6)*rho^2*sin(2theta). n=2, m=2. sqrt(n+1)=sqrt(3)? 
            #   Actually Noll normalization is sqrt(2)*sqrt(n+1) for m!=0. 
            #   sqrt(2)*sqrt(3) = sqrt(6). Correct.
            
            if m == 0:
                norm = np.sqrt(n + 1)
                term = zernike_radial(n, m, rho)
            else:
                norm = np.sqrt(2 * (n + 1))
                R = zernike_radial(n, m, rho)
                if j % 2 == 0: # Even j -> cos (usually, check mapping)
                    # mapping: 2->(1,1) cos; 3->(1,-1) sin.
                    # 6->(2,2) cos; 5->(2,-2) sin.
                    # 12->(4,2) cos; 13->(4,-2) sin.
                    # So Even j -> cos, Odd j (excluding m=0 cases) -> sin?
                    # Wait, j=3 (odd) is sin. j=5 (odd) is sin. 
                    # j=2 (even) is cos. j=6 (even) is cos.
                    # So yes: for m!=0: j even -> cos, j odd -> sin.
                    term = R * torch.cos(abs(m) * theta)
                else:
                    term = R * torch.sin(abs(m) * theta)
            
            Z = torch.tensor(norm, device=device) * term
            basis.append(Z)
            
        self.basis = torch.stack(basis, dim=0) # [N_modes, G, G]
        self.basis = self.basis * self.mask.unsqueeze(0)
        
        # Register buffer so it saves with state_dict but isn't a parameter
        self.register_buffer('zernike_basis', self.basis)
        self.register_buffer('aperture_mask', self.mask)

    def forward(self, coefficients):
        """
        coefficients: [B, N_modes]
        Returns: wavefront phase [B, G, G]
        """
        # [B, N, 1, 1] * [1, N, G, G] -> [B, N, G, G] -> sum -> [B, G, G]
        # or einsum
        return torch.einsum('bn,nhw->bhw', coefficients, self.zernike_basis)

class DifferentiableZernikeGenerator(nn.Module):
    def __init__(self, n_modes, pupil_size, kernel_size, 
                 oversample_factor=2, 
                 wavelengths=None, ref_wavelength=550e-9,
                 device='cpu', learnable_wavelengths=False, wavelength_bounds=None):
        """
        Args:
            n_modes: Zernike 模式数量
            pupil_size: 光瞳网格大小 (simulation grid size)
            kernel_size: 输出 PSF 卷积核的大小
            oversample_factor: 过采样因子 (default: 2)
            wavelengths: List of wavelengths [R, G, B] in meters. If None, mono (ref_wavelength).
            ref_wavelength: Reference wavelength for the coefficients.
            device: 计算设备
            learnable_wavelengths: 是否将波长设为可学习参数。
            wavelength_bounds: 波长范围 [min, max]，用于约束可学习波长。
        """
        super().__init__()
        self.n_modes = n_modes
        self.pupil_size = pupil_size
        self.kernel_size = kernel_size
        self.oversample_factor = oversample_factor
        self.learnable_wavelengths = learnable_wavelengths
        self.wavelength_bounds = wavelength_bounds if wavelength_bounds is not None else [400e-9, 700e-9]
        self.wavelengths = wavelengths if wavelengths is not None else [ref_wavelength]
        self.ref_wavelength = ref_wavelength

        if (not isinstance(self.wavelength_bounds, (list, tuple))
                or len(self.wavelength_bounds) != 2
                or self.wavelength_bounds[0] >= self.wavelength_bounds[1]):
            raise ValueError(f"Invalid wavelength_bounds: {self.wavelength_bounds}")

        if self.learnable_wavelengths:
            min_w, max_w = self.wavelength_bounds
            wavelengths_tensor = torch.tensor(self.wavelengths, device=device, dtype=torch.float32)
            denom = max_w - min_w
            if denom <= 0:
                raise ValueError(f"Invalid wavelength bounds: {self.wavelength_bounds}")
            eps = 1e-6
            normalized = (wavelengths_tensor - min_w) / denom
            normalized = torch.clamp(normalized, eps, 1.0 - eps)
            raw = torch.log(normalized / (1.0 - normalized))
            self.raw_wavelengths = nn.Parameter(raw)
        else:
            self.register_buffer(
                "wavelengths_tensor",
                torch.tensor(self.wavelengths, device=device, dtype=torch.float32)
            )
        
        # [Fix] Enforce odd kernel size for alignment
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd to ensure physical alignment, got {kernel_size}")
        
        # 基础 Zernike Basis 依然在原始分辨率 pupil_size 上计算，节省显存
        self.basis = ZernikeBasis(n_modes, pupil_size, device)
        
    def forward(self, coefficients):
        """
        coefficients: [B, N_modes] (defined at ref_wavelength)
        Output: PSF kernels [B, C, K, K] where C = len(wavelengths)
        """
        # 1. 计算波前相位 (Reference Phase)
        # phi_ref = 2pi * OPD / lambda_ref
        # coefficients are in "waves" at lambda_ref => OPD = C * lambda_ref
        # phi_ref = 2pi * C
        autocast_ctx = (
            torch.autocast(device_type=coefficients.device.type, enabled=False)
            if coefficients.device.type in ("cuda", "cpu") else nullcontext()
        )

        # 在复数物理计算路径中固定使用 float32，避免 AMP 将其降为 ComplexHalf。
        with autocast_ctx:
            coeffs_fp32 = coefficients.float()
            phi_ref = 2 * torch.pi * self.basis(coeffs_fp32).float()  # [B, G, G]
        
        # 2. Multi-wavelength Loop
        psf_channels = []

        wavelengths = self._get_wavelengths()
        for lam in wavelengths:
            # 仅此复数计算段保持 fp32，保证实验精度与数值稳定性
            with autocast_ctx:
                # Scale phase: phi_lambda = phi_ref * (lambda_ref / lambda)
                lam_fp32 = lam.float() if isinstance(lam, torch.Tensor) else torch.tensor(lam, device=phi_ref.device, dtype=torch.float32)
                scale = torch.tensor(self.ref_wavelength, device=phi_ref.device, dtype=torch.float32) / lam_fp32
                phi = phi_ref * scale

                # Pupil Function P = A * exp(i * phi)
                A = self.basis.aperture_mask.float()
                pupil = A * torch.exp(1j * phi)  # [B, G, G], complex64
            
            # Oversampling
            '''
原始 FFT (64×64):
- 分辨率低
- PSF 边界有混叠 (aliasing)

过采样 (128×128):
- 2 倍分辨率
- 减少混叠
- 更准确的 PSF 边界

实际应用:
原始 → 过采样 → FFT → 下采样 → 裁剪
64    128      128    64       33
            '''
            # ====================================================
            # [Fix] Grid Alignment Strategy
            # 强制过采样后的网格为奇数，确保 Padding 完美对称，
            # 彻底消除偶数网格带来的亚像素偏移 (Sub-pixel shift)。
            # ====================================================
            target_size = self.pupil_size * self.oversample_factor
            if target_size % 2 == 0:
                target_size += 1  # 强制转为奇数 (e.g., 65*2=130 -> 131)

            pad_total = target_size - self.pupil_size
            # 此时 pad_total 必为偶数 (e.g., 131 - 65 = 66)

            half_pad = pad_total // 2
            # 完美对称填充
            p_l, p_r, p_t, p_b = half_pad, half_pad, half_pad, half_pad

            pupil_padded = F.pad(pupil, (p_l, p_r, p_t, p_b), mode='constant', value=0)
            # ====================================================
            
            # FFT
            complex_field = torch.fft.ifftshift(pupil_padded, dim=(-2, -1))
            psf_complex = torch.fft.fft2(complex_field)
            psf_complex = torch.fft.fftshift(psf_complex, dim=(-2, -1))
            
            # Intensity
            psf_high_res = (psf_complex.abs()) ** 2
            '''
瞳孔函数 P(x,y)(复数)
    ↓ FFT
频域复数场
    ↓ |·|
振幅谱
    ↓ (·)²
强度 (PSF)
    
结果: 高斯-like 的亮点
┌──────────────┐
│              │
│    ╱╲╱╲      │
│  ╱      ╲    │
│ │  亮点  │    │  ← PSF 中心集中
│  ╲      ╱    │
│    ╲╱╲╱      │
│              │
└──────────────┘
            '''
            # Downsample：回到原大小
            # 下采样方法：平均池化 (Average Pooling)
            if self.oversample_factor > 1:
                # [Fix] Explicit dimension for pooling to avoid ambiguity
                # psf_high_res: [B, G, G] -> [B, 1, G, G]
                psf = F.avg_pool2d(psf_high_res.unsqueeze(1), 
                                 kernel_size=self.oversample_factor, 
                                 stride=self.oversample_factor).squeeze(1)
            else:
                psf = psf_high_res
                
            # Global Normalize
            psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            
            # Crop
            G = self.pupil_size
            K = self.kernel_size
            if K > G: raise ValueError(f"Kernel size {K} > Pupil size {G}")
            start = G // 2 - K // 2
            end = start + K
            psf_cropped = psf[:, start:end, start:end]
            
            # Re-normalize
            # PSF 代表单位光源的响应，应该守恒能量
            psf_cropped = psf_cropped / (psf_cropped.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            
            psf_channels.append(psf_cropped)
            
        # Stack channels: [B, C, K, K]
        return torch.stack(psf_channels, dim=1)

    def _get_wavelengths(self):
        if self.learnable_wavelengths:
            min_w, max_w = self.wavelength_bounds
            scale = torch.sigmoid(self.raw_wavelengths)
            return min_w + (max_w - min_w) * scale
        return self.wavelengths_tensor
    
'''
AberrationNet 输出
    │
    ▼
Zernike 系数 [B*N, 36]
(如 [128, 36])
    │
    ├─ a₁ (Piston)
    ├─ a₂ (Tilt-X)
    ├─ a₃ (Tilt-Y)
    ├─ a₄ (Defocus) ← 最重要
    ├─ a₅ (Astigmatism-45°)
    ├─ ...
    └─ a₃₆ (Higher-order mode)
    │
    ▼
DifferentiableZernikeGenerator.forward()
    │
    ├─ 调用 ZernikeBasis(coeffs)
    │  └─ einsum: aⱼ × Zⱼ → φ_ref [B, 64, 64]
    │
    ├─ 多波长循环 (R, G, B)
    │  │
    │  ├─ 红光 (650 nm)
    │  │  ├─ φ = φ_ref × (550/650)
    │  │  ├─ P = A × exp(i×φ)
    │  │  ├─ 过采样: 64 → 128
    │  │  ├─ FFT 计算 PSF
    │  │  ├─ 下采样: 128 → 64
    │  │  ├─ 裁剪: [64, 64] → [33, 33]
    │  │  └─ PSF_R [B, 33, 33]
    │  │
    │  ├─ 绿光 (550 nm)
    │  │  └─ PSF_G [B, 33, 33]
    │  │
    │  └─ 蓝光 (450 nm)
    │     └─ PSF_B [B, 33, 33]
    │
    ▼
堆叠 RGB 通道
    │
    ▼
PSF 卷积核 [B*N, 3, 33, 33]
(如 [128, 3, 33, 33])
    │
    ▼
返回到 physical_layer
进行 FFT 卷积
'''

