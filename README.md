# 物理驱动盲去卷积网络 (Physics-Driven Blind Deconvolution Network)

本项目是一个基于 PyTorch 实现的物理驱动盲去卷积网络，专门用于复原由空间变化的光学像差导致的模糊图像。该模型利用 Zernike 多项式参数化波前像差，通过可微傅里叶光学将其转换为点扩散函数 (PSF)，并结合深度学习网络实现高质量的图像复原。

## 核心特性

- **双分支架构**: 图像复原网络 (RestorationNet, U-Net) + 光学辨识网络 (AberrationNet, MLP)。
- **可微物理层**: Zernike 系数 → 波前相位 → PSF → 空间变化卷积 (基于 Overlap-Add 和 FFT 高效实现)。
- **三阶段解耦训练**: 物理层预训练 → 固定物理层训练复原网络 → 联合微调，确保物理模型的准确性和训练的稳定性。
- **熔断机制 (Circuit Breaker)**: 严格的质量阈值控制阶段切换，防止过早进入下一阶段导致误差放大。
- **自监督物理约束**: 通过重模糊一致性损失 (Reblurring Consistency Loss) 实现无监督的 PSF 估计。
- **极端 OOD 数据集生成**: 提供脚本生成极端像差数据集，用于测试模型的泛化能力。

## 系统架构

```text
输入模糊图像 (Y) 
    │
    ├─▶ RestorationNet (U-Net) ──▶ 复原图像 (X̂)
    │                                  │
    └─▶ AberrationNet (MLP)            │
             │                         │
        Zernike 系数                   │
             │                         │
        PSF 生成 (FFT)                 │
             │                         │
        空间变化模糊 ◀─────────────────┘
             │
        重模糊图像 (Ŷ)
             │
        Loss = MSE(Ŷ, Y) + L1(X̂, X_gt)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备 (DPDD 数据集)

将 DPDD 数据集放置在 `data/dd_dp_dataset_png/` 目录下。目录结构如下：
```text
data/dd_dp_dataset_png/
    train_c/
        source/ (模糊图像)
        target/ (清晰图像)
    val_c/
        source/
        target/
    test_c/
        source/
        target/
```
*注：本项目直接从 source/target 文件夹读取数据，无需额外的缩放预处理。*

### 2. 模型训练

训练过程采用**三阶段解耦策略**，由 `trainer.py` 中的 `DualBranchTrainer` 控制。

```bash
# 使用默认配置进行标准训练
python train.py --config config/default.yaml

# 从检查点恢复训练
python train.py --resume results/latest.pt

# 指定训练特定阶段 (1, 2, 3 或 all)
python train.py --stage all
```

**训练阶段说明:**
1. **Stage 1 (Physics Only)**: 仅训练 `AberrationNet`，利用重模糊一致性损失拟合光学像差。验证指标：Re-blur MSE。
2. **Stage 2 (Restoration)**: 冻结物理层，训练 `RestorationNet`。验证指标：PSNR & SSIM。
3. **Stage 3 (Joint Finetuning)**: 解冻所有模块，学习率减半进行联合微调。验证指标：综合指标。

### 3. 模型测试

在全分辨率 (1680 x 1120) 测试集上评估模型性能。

```bash
# 使用联合微调后的最佳模型进行测试，并保存复原图像
python test.py --checkpoint results/best_stage3_joint.pt --config config/default.yaml --save-restored

# 仅评估指标，不保存图像
python test.py --checkpoint results/best_stage2_psnr.pt
```
测试结果将保存在 `results/` 目录下，包含详细的 `test_results.csv` (PSNR, SSIM, LPIPS, Reblur_MSE) 和可视化对比图。

### 4. 极端 OOD 数据集生成

本项目提供了一个脚本，用于生成具有极端空间变化像差（如严重边缘崩坏、强彗差和像散）的测试集，以评估模型的鲁棒性。

```bash
python generate_extreme_ood_dataset.py
```

## 目录结构

```text
Defocus_plus/
├── config/                   # YAML 配置文件 (包含消融实验配置)
│   ├── default.yaml          # 默认完整配置
│   ├── ablation_1_no_physics.yaml # 消融实验：无物理层
│   └── ...
├── data/                     # 数据集及预处理脚本
├── models/                   # 核心网络模型
│   ├── aberration_net.py     # 像差预测网络 (MLP + 傅里叶特征编码)
│   ├── physical_layer.py     # 空间变化物理层 (基于 Overlap-Add)
│   ├── restoration_net.py    # 图像复原网络 (U-Net)
│   └── zernike.py            # Zernike 多项式与 PSF 生成器
├── utils/                    # 工具函数 (数据加载、指标计算、可视化等)
├── generate_extreme_ood_dataset.py # 极端像差数据集生成脚本
├── train.py                  # 训练主脚本
├── test.py                   # 测试主脚本
├── trainer.py                # 三阶段解耦训练器
└── README.md                 # 项目说明文档
```

## 核心组件解析

### 1. Zernike PSF 生成器 (`models/zernike.py`)
将 Zernike 系数转换为 PSF 卷积核。支持高阶像差（默认 36 阶 Noll 模式），通过可微的傅里叶光学过程（波前相位 -> 瞳孔函数 -> PSF）实现。

### 2. 像差网络 (`models/aberration_net.py`)
基于 MLP 的网络，输入归一化空间坐标，输出该位置的 Zernike 系数。引入了**傅里叶特征编码 (Fourier Feature Encoding)**，有效提升了网络对高频空间变化的拟合能力。

### 3. 物理层 (`models/physical_layer.py`)
实现了高效的**空间变化卷积**。采用 Overlap-Add (OLA) 策略，将图像分割为重叠的补丁 (Patch)，为每个补丁生成局部 PSF，并在频域 (FFT) 进行快速卷积，最后通过 Hann 窗口加权拼接，确保平滑过渡。

### 4. 复原网络 (`models/restoration_net.py`)
基于 U-Net 架构，支持坐标注入 (CoordConv) 和物理先验注入 (将预测的 Zernike 系数图作为额外特征输入)，引导网络更好地处理空间变化的模糊。

## 消融实验 (Ablation Studies)

`config/` 目录下提供了多个消融实验配置文件：
- `ablation_1_no_physics.yaml`: 移除物理层，退化为纯数据驱动的端到端复原网络。
- `ablation_2_no_fourier.yaml`: 移除 AberrationNet 中的傅里叶特征编码。
- `ablation_3_end_to_end.yaml`: 取消三阶段解耦训练，直接进行端到端联合训练。

可以通过指定 `--config` 参数运行相应的消融实验：
```bash
python train.py --config config/ablation_1_no_physics.yaml
```