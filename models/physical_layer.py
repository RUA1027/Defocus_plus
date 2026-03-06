import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from .zernike import DifferentiableZernikeGenerator
from .aberration_net import AberrationNet

class SpatiallyVaryingPhysicalLayer(nn.Module):

    def __init__(self, aberration_net: nn.Module, zernike_generator: DifferentiableZernikeGenerator, patch_size, stride):
        super().__init__()
        self.aberration_net = aberration_net
        self.zernike_generator = zernike_generator
        self.patch_size = patch_size
        self.stride = stride
        self.kernel_size = zernike_generator.kernel_size
        self.gamma = nn.Parameter(torch.tensor(2.2))
        self.epsilon = 1e-06
        '\n        1D Hann 窗口:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n1.0 |        ╱╲\n    |       ╱  ╲\n0.5 |      ╱    ╲\n    |     ╱      ╲\n0.0 |____╱________╲____ \n    0   32   64   96  128\n特点:\n• 起点 (n=0): w=0\n• 中点 (n=64): w=1 (最大值)\n• 终点 (n=127): w≈0\n• 平滑过渡，无尖角\n2D Hann 窗口 (补丁):\n━━━━━━━━━━━━━━━━━\n    ┌─────────────────┐\n    │     亮 (1.0)    │ ← 中心\n    │   ╱         ╲   │\n    │  ╱           ╲  │\n    │ ╱             ╲ │\n    │╱_______________╲│\n    │ 暗 (0)         │ ← 边缘\n    └─────────────────┘\n中心最亮，边缘逐渐变暗\n        '
        hann = torch.hann_window(patch_size)
        window_2d = torch.outer(hann, hann)
        self.register_buffer('window', window_2d)

    def get_patch_centers(self, H, W, device, H_orig=None, W_orig=None, crop_info=None):
        n_h = (H - self.patch_size) // self.stride + 1
        n_w = (W - self.patch_size) // self.stride + 1
        y_centers_local = torch.arange(n_h, device=device) * self.stride + self.patch_size / 2
        x_centers_local = torch.arange(n_w, device=device) * self.stride + self.patch_size / 2
        if crop_info is not None and H_orig is not None and (W_orig is not None):
            crop_info = crop_info.to(device)
            (top_norm, left_norm, crop_h_norm, crop_w_norm) = crop_info
            top_pix = top_norm * H_orig
            left_pix = left_norm * W_orig
            crop_h_pix = crop_h_norm * H_orig
            crop_w_pix = crop_w_norm * W_orig
            y_centers_global = y_centers_local + top_pix
            x_centers_global = x_centers_local + left_pix
            y_norm = y_centers_global / H_orig * 2 - 1
            x_norm = x_centers_global / W_orig * 2 - 1
        else:
            y_norm = y_centers_local / H * 2 - 1
            x_norm = x_centers_local / W * 2 - 1
        (grid_y, grid_x) = torch.meshgrid(y_norm, x_norm, indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1)
        return coords.reshape(-1, 2)

    def compute_coefficient_smoothness(self, grid_size=16):
        device = next(self.aberration_net.parameters()).device
        y = torch.linspace(-1, 1, grid_size, device=device)
        x = torch.linspace(-1, 1, grid_size, device=device)
        (grid_y, grid_x) = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        coeffs = self.aberration_net(coords)
        coeffs_map = coeffs.view(grid_size, grid_size, -1).permute(2, 0, 1)
        dy = torch.abs(coeffs_map[:, 1:, :] - coeffs_map[:, :-1, :]).mean()
        dx = torch.abs(coeffs_map[:, :, 1:] - coeffs_map[:, :, :-1]).mean()
        return dy + dx

    def generate_coeffs_map(self, H, W, device, grid_size=16, crop_info=None, batch_size=1):
        with torch.no_grad():
            if crop_info is not None:
                if crop_info.dim() == 1:
                    crop_info = crop_info.unsqueeze(0)
                all_coords = []
                for b in range(batch_size):
                    ci = crop_info[b] if b < crop_info.shape[0] else crop_info[0]
                    (top_n, left_n) = (ci[0].item(), ci[1].item())
                    (h_n, w_n) = (ci[2].item(), ci[3].item())
                    if h_n > 0 and w_n > 0:
                        y_s = top_n * 2 - 1
                        y_e = (top_n + h_n) * 2 - 1
                        x_s = left_n * 2 - 1
                        x_e = (left_n + w_n) * 2 - 1
                    else:
                        (y_s, y_e, x_s, x_e) = (-1.0, 1.0, -1.0, 1.0)
                    y = torch.linspace(y_s, y_e, grid_size, device=device)
                    x = torch.linspace(x_s, x_e, grid_size, device=device)
                    (gy, gx) = torch.meshgrid(y, x, indexing='ij')
                    coords = torch.stack([gy.flatten(), gx.flatten()], dim=1)
                    all_coords.append(coords)
                all_coords_cat = torch.cat(all_coords, dim=0)
                all_coeffs = self.aberration_net(all_coords_cat)
                n_coeffs = all_coeffs.shape[1]
                all_coeffs = all_coeffs.view(batch_size, grid_size, grid_size, n_coeffs)
                all_coeffs = all_coeffs.permute(0, 3, 1, 2)
                coeffs_map = F.interpolate(all_coeffs, size=(H, W), mode='bilinear', align_corners=True)
            else:
                y = torch.linspace(-1, 1, grid_size, device=device)
                x = torch.linspace(-1, 1, grid_size, device=device)
                (gy, gx) = torch.meshgrid(y, x, indexing='ij')
                coords = torch.stack([gy.flatten(), gx.flatten()], dim=1)
                coeffs = self.aberration_net(coords)
                n_coeffs = coeffs.shape[1]
                cmap = coeffs.view(1, grid_size, grid_size, n_coeffs).permute(0, 3, 1, 2)
                cmap = F.interpolate(cmap, size=(H, W), mode='bilinear', align_corners=True)
                coeffs_map = cmap.expand(batch_size, -1, -1, -1)
            return coeffs_map

    def forward(self, x_hat, crop_info=None):
        (B, C, H, W) = x_hat.shape
        P = self.patch_size
        S = self.stride
        K = self.kernel_size
        x_linear = (F.relu(x_hat) + self.epsilon).pow(self.gamma)
        pad_h = (S - (H - P) % S) % S
        pad_w = (S - (W - P) % S) % S
        if H < P:
            pad_h += P - H
        if W < P:
            pad_w += P - W
        if H < pad_h or W < pad_w:
            mode_pad = 'replicate'
        else:
            mode_pad = 'reflect'
        x_padded = F.pad(x_linear, (0, pad_w, 0, pad_h), mode=mode_pad)
        (H_pad, W_pad) = x_padded.shape[2:]
        '\n[B, C, H_pad, W_pad] \n    ↓\n[B, C*P*P, N_patches]  ← 每列是一个 P×P 补丁的展平版本\n    ↓\nreshape → [B*N_patches, C, P, P]\n例如 B=2, C=3, N_patches=64:\n[2, 3*128*128, 64]\n    ↓\n[2, 49152, 64]\n    ↓\n[128, 3, 128, 128]\n        '
        patches_unfolded = F.unfold(x_padded, kernel_size=P, stride=S)
        N_patches = patches_unfolded.shape[2]
        patches_unfolded = patches_unfolded.transpose(1, 2)
        patches = patches_unfolded.reshape(B * N_patches, C, P, P)
        "\n        不同补丁使用不同的卷积核:\n        补丁 1 (中心: 图像左上)\n  ├─ 坐标 (-0.778, -0.778)\n    ├─ AberrationNet 预测系数 [a₁, a₂, ..., a₃₆]\n  └─ ZernikeGenerator → PSF 核 K₁ [3, 31, 31]\n补丁 2 (中心: 图像中心)\n  ├─ 坐标 (0, 0)\n    ├─ AberrationNet 预测系数 [a'₁, a'₂, ..., a'₃₆]\n  └─ ZernikeGenerator → PSF 核 K₂ [3, 31, 31]\n补丁 64 (中心: 图像右下)\n  ├─ 坐标 (0.778, 0.778)\n    ├─ AberrationNet 预测系数 [a''₁, a''₂, ..., a''₃₆]\n  └─ ZernikeGenerator → PSF 核 K₆₄ [3, 31, 31]\n        "
        if crop_info is not None:
            if crop_info.dim() == 1:
                crop_info = crop_info.unsqueeze(0)
            coords_list = []
            for b in range(B):
                crop_info_single = crop_info[b]
                crop_h_norm = crop_info_single[2].item()
                crop_w_norm = crop_info_single[3].item()
                if crop_h_norm > 0 and crop_w_norm > 0:
                    H_orig = int(H / crop_h_norm)
                    W_orig = int(W / crop_w_norm)
                else:
                    H_orig = H
                    W_orig = W
                    crop_info_single = None
                coords_1img = self.get_patch_centers(H_pad, W_pad, x_hat.device, H_orig=H_orig, W_orig=W_orig, crop_info=crop_info_single)
                coords_list.append(coords_1img)
            coords = torch.cat(coords_list, dim=0)
        else:
            coords_1img = self.get_patch_centers(H_pad, W_pad, x_hat.device)
            coords = coords_1img.repeat(B, 1)
        coeffs = self.aberration_net(coords)
        kernels = self.zernike_generator(coeffs)
        C_k = kernels.shape[1]
        if C == C_k:
            C_out = C
        elif C == 1 and C_k > 1:
            C_out = C_k
        elif C > 1 and C_k == 1:
            C_out = C
        else:
            raise ValueError(f'Channel mismatch: Input ({C}) and Kernel ({C_k}) are not compatible for broadcasting.')
        kernels_flipped = torch.flip(kernels, dims=[-2, -1])
        BN = patches.shape[0]
        pad = K // 2
        patches_padded = F.pad(patches, (pad, pad, pad, pad), mode='constant', value=0)
        if C == C_k:
            patches_grouped = patches_padded.view(1, BN * C, patches_padded.shape[2], patches_padded.shape[3])
            kernels_grouped = kernels_flipped.view(BN * C, 1, K, K)
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
            y_patches = y_grouped.view(BN, C, P, P)
            C_out = C
        elif C == 1 and C_k > 1:
            patches_expanded = patches_padded.expand(-1, C_k, -1, -1)
            patches_grouped = patches_expanded.reshape(1, BN * C_k, patches_padded.shape[2], patches_padded.shape[3])
            kernels_grouped = kernels_flipped.view(BN * C_k, 1, K, K)
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C_k)
            y_patches = y_grouped.view(BN, C_k, P, P)
            C_out = C_k
        else:
            kernels_expanded = kernels_flipped.expand(-1, C, -1, -1)
            patches_grouped = patches_padded.view(1, BN * C, patches_padded.shape[2], patches_padded.shape[3])
            kernels_grouped = kernels_expanded.reshape(BN * C, 1, K, K)
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
            y_patches = y_grouped.view(BN, C, P, P)
            C_out = C
        window_4d = self.window.view(1, 1, P, P)
        y_patches = y_patches * window_4d
        y_patches_reshaped = y_patches.reshape(B, N_patches, C_out * P * P).transpose(1, 2)
        output_h = H_pad
        output_w = W_pad
        y_accum = F.fold(y_patches_reshaped, output_size=(output_h, output_w), kernel_size=P, stride=S)
        w_patches = window_4d.expand(B * N_patches, C_out, P, P)
        w_patches_reshaped = w_patches.reshape(B, N_patches, C_out * P * P).transpose(1, 2)
        w_accum = F.fold(w_patches_reshaped, output_size=(output_h, output_w), kernel_size=P, stride=S)
        y_hat_padded = y_accum / (w_accum + 1e-08)
        y_hat = y_hat_padded[..., :H, :W]
        y_hat = y_hat.clamp(min=self.epsilon).pow(1.0 / self.gamma)
        return y_hat