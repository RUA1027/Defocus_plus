# =============================================================================
# U-Net 架构的物理驱动图像复原网络 (Physics-Guided Restoration Network)
# =============================================================================
#
# 核心设计理念：多尺度空间特征变换 (Multi-scale SFT Modulation)
#
# 与早期融合 (Early Fusion / Concat) 不同，本方案通过 SFT (Spatial Feature
# Transform) 在解码器的每个尺度上动态注入 Zernike 像差先验。
#
# 为什么在解码器（而非编码器）注入物理先验？
#   - 编码器负责从退化图像中提取多尺度语义与结构特征——这一过程应当
#     独立于像差信息，以确保学到通用的图像表征能力。
#   - 解码器负责细节重建与像素级重分布。在此阶段注入空间变化的像差
#     先验，可以直接指导网络根据局部像差严重程度动态调整复原力度，
#     实现"哪里模糊严重就在哪里加强复原"的物理感知策略。
#   - 这种设计使得物理信息通过轻量的 1×1 卷积以仿射变换形式调制
#     特征激活，计算开销极低 (MACs 增量 < 2%)，同时避免了 Concat
#     方案中高通道数带来的参数冗余与特征稀释问题。
#
# 数据流：
#
# 模糊图像 [B, 3, H, W]   Zernike系数图 [B, 36, H, W]
#     │                         │
#     ├─ (可选) CoordConv       │  (低频空间平滑的控制信号)
#     │  └─ [B, 5, H, W]       │
#     ▼                         │
# ┌─ Encoder ──────────────┐    │
# │ Inc:  → base           │    │
# │ Down1: → base×2        │    │
# │ Down2: → base×4        │    │
# │ Down3: → base×8        │    │
# │ Down4: → base×16/f     │    │
# └────────────────────────┘    │
#     │                         │
# ┌─ Decoder + SFT ────────┐   │
# │ Up1 → SFT1 (H/8)  ◄───┼───┤  物理调制: 引导粗尺度结构复原
# │ Up2 → SFT2 (H/4)  ◄───┼───┤  物理调制: 引导中尺度纹理复原
# │ Up3 → SFT3 (H/2)  ◄───┼───┤  物理调制: 引导细尺度边缘复原
# │ Up4 → SFT4 (H)    ◄───┼───┘  物理调制: 引导全分辨率像素重分布
# └────────────────────────┘
#     │
#     ▼
#   OutConv → [B, 3, H, W] (残差修正量)
#     │
#     + x_input (残差连接)
#     │
#     ▼
#   [B, 3, H, W] 复原图像
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # 功能：提取特征的基本单元
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels), # Batch Norm can sometimes be problematic in low batch size physics sims, but usually fine.
            # Using GroupNorm or InstanceNorm is often safer for restoration, lets stick to simple Conv+ReLU for now or include BN.
            # Deconvolution often prefers removing BN or using Instance Norm. Let's use standard BN for U-Net baseline.
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    '''
输入 [B, in_c, H, W]
    │
    ├─ MaxPool2d(2)  ← 2×2 最大池化
    │  [B, in_c, H/2, W/2]
    │
    ├─ DoubleConv
    │  [B, out_c, H/2, W/2]
    │
    ▼
输出 [B, out_c, H/2, W/2]

分辨率降低 (↓2)
通道增加 (通常翻倍)
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    '''
解码器的上一层 x1       编码器的同级特征 x2
[B, 512, H/16, W/16]   [B, 256, H/8, W/8]
        │                       │
        │ Upsample(×2)         │
        ├─→ [B, 512, H/8, W/8] │
        │                       │
        ├─ 处理尺寸差异 (padding)
        │                       │
        └──────┬────────────────┘
               │
           Concatenate (通道维)
         [B, 768, H/8, W/8]
               │
             Conv
         [B, 256, H/8, W/8]
               │
               ▼
           输出
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    '''
输入 [B, 64, H, W]
    │
    ├─ Conv1×1 (无 padding，无激活)
    │  降维到输出通道数
    │
    ▼
输出 [B, 3, H, W]
    (RGB 三通道)
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SFTLayer(nn.Module):
    """
    Spatial Feature Transform (SFT)

    使用低频、空间平滑的 Zernike 系数图作为控制信号，
    对当前尺度特征进行逐像素仿射调制：
        F_out = F_in * (1 + gamma) + beta

    其中 gamma / beta 通过 1x1 卷积从 cond 中预测。
    为保证训练初期稳定性，gamma/beta 最后一层卷积采用全零初始化，
    使 SFT 初始行为为恒等映射（Identity）。
    """
    def __init__(self, in_channels, cond_channels=36):
        super().__init__()

        hidden_channels = max(32, in_channels // 2)

        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.to_gamma = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        self.to_beta = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)

        # Critical: zero-init guardrail，确保初始阶段 SFT = Identity
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_beta.weight)
        if self.to_gamma.bias is not None:
            nn.init.zeros_(self.to_gamma.bias)
        if self.to_beta.bias is not None:
            nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, cond):
        # 1) 空间对齐：将全局 coeffs_map 对齐到当前特征尺度
        cond = F.interpolate(
            cond,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 2) 参数提取：从物理先验中映射 gamma / beta
        feat = self.shared(cond)
        gamma = self.to_gamma(feat)
        beta = self.to_beta(feat)

        # 3) 仿射调制
        return x * (1.0 + gamma) + beta

class RestorationNet(nn.Module):
    """
    物理驱动 U-Net 复原网络 (Multi-scale SFT Modulation)
    
    通过 SFTLayer 在解码器各尺度注入 Zernike 像差先验，实现空间自适应复原。
    输入层仅接收 RGB (+ 可选坐标)，物理信息不再拼接到输入通道。
    
    Args:
        n_channels: 输入图像通道数 (RGB=3)
        n_classes:  输出图像通道数 (RGB=3)
        base_filters: 基础滤波器数量
        bilinear: 上采样方式 (True=双线性插值)
        use_coords: 是否启用 CoordConv 坐标注入
        n_coeffs: Zernike 系数通道数 (用于 SFT 调制, 0=不使用物理先验)
    """
    def __init__(self, n_channels, n_classes, base_filters=32, bilinear=True, use_coords=False, n_coeffs=0):
        super(RestorationNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_coords = use_coords
        self.n_coeffs = n_coeffs  # Zernike 系数通道数 (>0 时启用 SFT 调制)
        
        factor = 2 if bilinear else 1
        
        # =====================================================================
        # 输入通道数: 仅包含图像通道 + 可选坐标通道
        # 严禁在此加入 n_coeffs —— 物理先验通过 SFT 在解码器注入
        # =====================================================================
        input_channels = n_channels + (2 if use_coords else 0)
        
        # =====================================================================
        # Encoder (编码器): 纯图像特征提取, 不涉及物理先验
        # =====================================================================
        self.inc = DoubleConv(input_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # =====================================================================
        # Decoder (解码器): 上采样 + 跳跃连接
        # =====================================================================
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # =====================================================================
        # SFT 调制层: 在解码器每个尺度注入 Zernike 像差先验
        # 
        # 物理意义:
        #   sft1 (H/8 尺度): 根据像差分布引导粗尺度结构复原 (全局散焦补偿)
        #   sft2 (H/4 尺度): 引导中尺度纹理重建 (空间变化的去模糊力度)
        #   sft3 (H/2 尺度): 引导细尺度边缘锐化 (局部像差自适应)
        #   sft4 (H   尺度): 引导全分辨率像素级重分布 (最终精细校正)
        #
        # 由于使用 1×1 卷积处理 Zernike 控制信号, 不参与复杂的空间卷积,
        # MACs 增量极低 (<2%), 是高效的物理信息注入方案。
        # =====================================================================
        if n_coeffs > 0:
            self.sft1 = SFTLayer(base_filters * 8 // factor, cond_channels=n_coeffs)
            self.sft2 = SFTLayer(base_filters * 4 // factor, cond_channels=n_coeffs)
            self.sft3 = SFTLayer(base_filters * 2 // factor, cond_channels=n_coeffs)
            self.sft4 = SFTLayer(base_filters, cond_channels=n_coeffs)
        
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x_input, coeffs_map=None):
        """
        前向传播: 编码器提取特征 → 解码器逐尺度重建 + SFT 物理调制 → 残差输出
        
        Args:
            x_input: [B, C, H, W] 模糊图像
            coeffs_map: [B, n_coeffs, H, W] Zernike 系数空间分布图 (可选)
                        每一通道代表一个 Zernike 模式的空间强度分布，
                        属于低频、空间平滑的控制信号,
                        用于引导解码器各尺度的特征变换。
        
        Returns:
            [B, C, H, W] 复原图像 = x_input + 残差修正量
        """
        x = x_input
        
        # ----- 可选: CoordConv 坐标注入 -----
        if self.use_coords:
            B, C, H, W = x.shape
            y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            x = torch.cat([x, grid_y, grid_x], dim=1)
        
        # =====================================================================
        # Encoder: 多尺度特征提取 (纯图像, 不含物理先验)
        # =====================================================================
        x1 = self.inc(x)       # [B, base,     H,    W   ]
        x2 = self.down1(x1)    # [B, base*2,   H/2,  W/2 ]
        x3 = self.down2(x2)    # [B, base*4,   H/4,  W/4 ]
        x4 = self.down3(x3)    # [B, base*8,   H/8,  W/8 ]
        x5 = self.down4(x4)    # [B, base*16/f, H/16, W/16] (瓶颈)
        
        # =====================================================================
        # Decoder + SFT 物理调制:
        # 每个 Up 模块完成后, 立即通过 SFTLayer 注入像差先验,
        # 根据 Zernike 图指示的局部像差严重程度动态调整特征激活增益。
        # =====================================================================
        use_sft = (coeffs_map is not None and self.n_coeffs > 0)
        
        d1 = self.up1(x5, x4)  # [B, base*8/f, H/8, W/8]
        if use_sft:
            # SFT1: H/8 尺度 — 粗尺度结构物理调制 (全局散焦补偿)
            d1 = self.sft1(d1, coeffs_map)
        
        d2 = self.up2(d1, x3)  # [B, base*4/f, H/4, W/4]
        if use_sft:
            # SFT2: H/4 尺度 — 中尺度纹理物理调制 (空间变化去模糊)
            d2 = self.sft2(d2, coeffs_map)
        
        d3 = self.up3(d2, x2)  # [B, base*2/f, H/2, W/2]
        if use_sft:
            # SFT3: H/2 尺度 — 细尺度边缘物理调制 (局部像差自适应)
            d3 = self.sft3(d3, coeffs_map)
        
        d4 = self.up4(d3, x1)  # [B, base, H, W]
        if use_sft:
            # SFT4: 全分辨率 — 像素级精细物理调制 (最终校正)
            d4 = self.sft4(d4, coeffs_map)
        
        logits = self.outc(d4)  # [B, 3, H, W] 残差修正量
        
        # 残差连接: 网络学习的是"模糊修正量", 而非直接预测清晰图像
        # Restored = Input + Correction
        return x_input + logits