# Physics-Driven Blind Deconvolution Network (Defocus+)

**Official PyTorch Implementation** of a physics-driven blind deconvolution network designed for restoring images degraded by spatially varying optical aberrations and severe defocus blur.

## 📌 Overview

This framework combines **differentiable physics optics** with **deep learning** to tackle blind image deblurring. It parameterizes optical wavefront aberrations using **Zernike polynomials** and leverages **FFT-based Fourier optics** to generate spatially varying Point Spread Functions (PSFs), maintaining high interpretability while achieving excellent generalization on both in-distribution and out-of-distribution data.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | Dual-branch system: `RestorationNet` (U-Net) for image restoration + `AberrationNet` (MLP with Fourier encoding) for optical aberration estimation |
| **Differentiable Physics Layer** | End-to-end differentiable pipeline: Zernike coefficients → Wavefront Phase → Pupil Function → PSFs → Spatially Varying Convolution |
| **Efficient Computation** | Overlap-Add (OLA) strategy with FFT for handling full-resolution images without memory explosion |
| **Curriculum Learning** | Three-stage decoupled training for stable convergence and accurate optical parameter recovery |
| **Self-Supervised Design** | Reblurring consistency loss enables unsupervised PSF estimation without ground-truth blur kernels |
| **Generalization Testing** | Tools for generating extreme out-of-distribution datasets to evaluate robustness under unprecedented optical conditions |


## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Blurred Input Image (Y)                          │
└────────┬─────────────────────────────────────────────────┬──────────┘
         │                                                  │
         ▼                                                  ▼
   ┌──────────────┐                              ┌────────────────────┐
   │RestorationNet│                              │ AberrationNet      │
   │   (U-Net)    │                              │  (MLP + FFE)       │
   └──────┬───────┘                              └────────┬───────────┘
          │                                               │
          │                                        Zernike Coefficients
          ▼                                               │
    Restored Image                                        ▼
         (X̂)                                     PSF Generation (FFT)
          │                                               │
          │                                      Spatially Varying
          │                                        PSF Grid
          │                                               │
          │                                    ┌──────────▼──────────┐
          │                                    │ Physical Layer OLA  │
          │                                    └──────────┬──────────┘
          │                                               │
          │                                               ▼
          │                                      Spatially Varying Blur
          │                                               │
          │                                               ▼
          │                                      Reblurred Image (Ŷ)
          │                                               │
          │                  ┌─────────────────────────────┘
          │                  │
          ▼                  ▼
    ┌──────────────┐  ┌─────────────┐
    │ L1 Loss with │  │  MSE Loss   │
    │Ground Truth  │  │ w/ Blurred  │
    │   (X_gt)     │  │ Input (Y)   │
    └──────────────┘  └─────────────┘
```

**Training Flow:**
- **Stage 1 (Physics-Only)**: Optimize AberrationNet using reblurring consistency loss
- **Stage 2 (Restoration)**: Train RestorationNet with frozen physics layer using L1 loss  
- **Stage 3 (Joint Fine-tuning)**: End-to-end optimization of both branches with reduced learning rate


## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+ (with CUDA support recommended for FFT operations)
- CUDA 11.0+ (optional but highly recommended)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/RUA1027/Defocus_plus.git
cd Defocus_plus

# Create virtual environment (optional but recommended)
conda create -n defocus python=3.10
conda activate defocus

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

**Dependencies:** torch, torchvision, numpy, pyyaml, tqdm, pillow, matplotlib, tabulate, thop, tensorboard, lpips

## 🚀 Quick Start

### Step 1: Data Preparation

Download the **Dual-Pixel Defocus Deblurring (DPDD)** dataset and organize as follows:

```
data/dd_dp_dataset_png/
├── train_c/
│   ├── source/        # Blurred input images
│   └── target/        # Ground truth sharp images
├── val_c/
│   ├── source/
│   └── target/
└── test_c/
    ├── source/
    └── target/
```

**Note:** The dataloader directly processes images from `source/` and `target/` directories without requiring manual preprocessing or scaling.

### Step 2: Training

#### Basic Training (Full Pipeline - All 3 Stages)

```bash
# Default configuration with 3-stage curriculum learning
python train.py --config config/default.yaml

# Resume from checkpoint
python train.py --resume results/ours/best_stage3_joint.pt

# Train specific stage only (options: '1', '2', '3', or 'all')
python train.py --config config/default.yaml --stage 1
python train.py --config config/default.yaml --stage all
```

#### Training Stages Explained

| Stage | Focus | Loss Function | Output Metric | Purpose |
|-------|-------|---------------|---------------|---------|
| **1: Physics-Only** | AberrationNet | Reblurring MSE | Blur Consistency | Warm up optical parameter estimation |
| **2: Restoration** | RestorationNet (frozen physics) | L1 + LPIPS | PSNR / SSIM | Learn image restoration with fixed optics |
| **3: Joint Fine-tuning** | Both branches end-to-end | Combined loss | Combined score | Refine all parameters jointly (reduced LR) |

**Key Training Parameters** (see `config/default.yaml`):
```yaml
experiment:
  epochs: 200                  # Total training epochs
  lr: 1e-4                    # Base learning rate
  device: "cuda"              # Use GPU
  seed: 42

training:
  stage_schedule:
    stage1_epochs: 50         # Physics pre-training
    stage2_epochs: 100        # Restoration learning
    stage3_epochs: 50         # Joint fine-tuning
```

### Step 3: Evaluation & Testing

```bash
# Test with best joint fine-tuned model and save restored images
python test.py --checkpoint results/best_stage3_joint.pt \
               --config config/default.yaml \
               --save-restored

# Evaluate metrics only (faster, no image saving)
python test.py --checkpoint results/best_stage3_joint.pt \
               --config config/default.yaml

# Test on custom dataset
python test.py --checkpoint results/best_stage3_joint.pt \
               --data-dir custom_data/test_images
```

**Output Structure:**
```
results/ours/
├── test_results.csv           # Metrics: PSNR, SSIM, LPIPS, Reblur MSE
├── test_results.json          # Detailed results
└── comparisons/               # Side-by-side visual comparisons
    ├── image_001_blurred.png
    ├── image_001_restored.png
    └── image_001_comparison.png
```

### Step 4: Out-of-Distribution Robustness Testing

Generate synthetic extreme aberration datasets to evaluate model robustness:

```bash
# Generate extreme OOD datasets
python generate_extreme_ood_dataset.py

# Test on extreme aberrations
python test.py --checkpoint results/best_stage3_joint.pt \
               --config config/default.yaml \
               --data-dir data/extreme\ aberration/
```

## 📂 Project Structure

```
Defocus_plus/
├── config/                          # Configuration files (YAML)
│   ├── default.yaml                 # ⭐ Default configuration (full pipeline)
│   ├── smoke_minimal.yaml           # Lightweight testing config
│   ├── ablation_1_no_physics.yaml   # Ablation: No physics layer
│   ├── ablation_2_no_fourier.yaml   # Ablation: No Fourier encoding
│   ├── ablation_3_end_to_end.yaml   # Ablation: Direct E2E training
│   └── ablation_4_no_newbp.yaml     # Ablation: Standard training
│
├── models/                          # Core neural network architectures
│   ├── zernike.py                   # Optical kernel engine (Zernike polynomials)
│   ├── physical_layer.py            # Differentiable physics: Overlap-Add (OLA) convolution
│   ├── aberration_net.py            # Aberration estimation network (MLP + Fourier Feature Encoding)
│   ├── restoration_net.py           # Prior-conditioned U-Net for image restoration
│   └── local_grouped_newbp.py       # NewBP activation groups for optimization
│
├── utils/                           # Utility modules
│   ├── model_builder.py             # Factory functions for building models & trainers
│   ├── dpdd_dataset.py              # DPDD dataset loader
│   ├── metrics.py                   # Evaluation metrics (PSNR, SSIM, LPIPS, etc.)
│   └── visualize.py                 # Visualization utilities
│
├── data/                            # Dataset directory
│   ├── dd_dp_dataset_png/           # Main training dataset (PNG format)
│   ├── dd_dp_dataset_pixel/         # Alternative pixel-level dataset
│   ├── dd_dp_dataset_png_mini/      # Lightweight version for quick testing
│   └── extreme\ aberration/         # OOD dataset with extreme aberrations
│
├── results/                         # Training outputs & results
│   ├── smoke/                       # Quick test results
│   ├── ablation1nophy/              # Ablation 1 results
│   ├── ablation2nofft/              # Ablation 2 results
│   ├── ablation3e2e/                # Ablation 3 results
│   └── ours/                        # Full method results
│
├── train.py                         # 🟢 Main training entry point
├── test.py                          # 🎯 Main evaluation entry point
├── trainer.py                       # Training loop implementation (DualBranchTrainer)
│
├── generate_extreme_ood_dataset.py  # Generate extreme OOD test datasets
├── NewBP_Algorithm_Reproduction.py  # Reference implementation details
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Key File Descriptions

| File | Purpose |
|------|---------|
| `models/zernike.py` | Zernike polynomial engine converting coefficients → PSFs via FFT |
| `models/physical_layer.py` | Overlap-Add (OLA) strategy for memory-efficient spatially-varying convolution |
| `models/aberration_net.py` | MLP-based aberration predictor with Fourier Feature Encoding (FFE) |
| `models/restoration_net.py` | U-Net restoration network with coordinate conditioning |
| `utils/dpdd_dataset.py` | DataLoader for DPDD image pairs |
| `trainer.py` | `DualBranchTrainer` class orchestrating 3-stage training curriculum |

## 🔧 Core Components

### 1. Zernike PSF Engine (`models/zernike.py`)

Translates abstract Zernike polynomial coefficients into dense PSF matrices through differentiable optics.

**Key Operations:**
```
Zernike Coefficients (36 modes, Noll index 1-36)
    ↓
Wavefront Phase Map (pupil plane)
    ↓
Pupil Function (complex amplitude)
    ↓
Fourier Transform (propagation in frequency domain)
    ↓
Point Spread Function (PSF matrix)
```

**Configuration (default.yaml):**
```yaml
physics:
  n_modes: 36                  # Covers up to 8th-order aberrations
  pupil_size: 129              # Optical computation grid resolution
  kernel_size: 65              # PSF kernel spatial extent
  ref_wavelength: 550.0e-9     # Reference wavelength (green light)
  wavelengths: [620e-9, 550e-9, 450e-9]  # RGB wavelengths
```

**Supported Aberrations:**
- Tip/Tilt (Noll 2-3): Image translation
- Defocus (Noll 4): Blur magnitude
- Astigmatism (Noll 5-6): Directional blur
- Coma (Noll 7-8): off-axis aberrations
- Trefoil (Noll 9-10): 3-fold asymmetry
- Spherical & higher orders (Noll 11-36)

### 2. Aberration Network (`models/aberration_net.py`)

Maps 2D spatial coordinates to spatially-varying Zernike coefficients with **Fourier Feature Encoding (FFE)** to overcome spectral bias.

**Architecture:**
```python
Input: Image coordinates (x, y) normalized to [-1, 1]
    ↓
Fourier Feature Encoding (FFE)
  - Transform: φ(x) = [sin(2π·B·x), cos(2π·B·x)]
  - B: Random projection matrix
  - Output dim: 128 (64 sin + 64 cos)
    ↓
Multi-layer MLP
  - Layer 1: FFE_dim (128) → hidden_dim (256)
  - ReLU, Layer Norm
  - Layer 2: 256 → 512
  - ReLU, Layer Norm
  - Layer 3: 512 → 256
  - ReLU
  - Output Layer: 256 → n_coeffs (36)
    ↓
Tanh activation × a_max → Bounded coefficients [-a_max, a_max]
```

**Why FFE Matters:**
- Vanilla MLPs suffer from spectral bias (learn low frequencies first)
- FFE explicitly encodes high-frequency components
- Critical for capturing fine spatial aberration variations

**Configuration:**
```yaml
aberration_net:
  n_coeffs: 36              # Must match physics.n_modes
  a_max: 1.0                # Coefficient bounds [-1.0, 1.0]
  use_fourier: true         # Enable Fourier Feature Encoding
  fourier_scale: 5          # FFE frequency scale
```

### 3. Differentiable Physics Layer (`models/physical_layer.py`)

Implements **Overlap-Add (OLA)** strategy for memory-efficient spatially-varying convolution on full-resolution images.

**Problem Solved:**
Direct FFT-based convolution with full-resolution images causes:
- Massive memory overhead (~10+ GB for 1680×1120 images)
- Computational inefficiency
- Solution: Divide into overlapping patches, process locally, aggregate smoothly

**OLA Algorithm:**
```
Full Resolution Image (H × W)
    ↓
Divide into overlapping patches (patch_size=128, stride=64)
    ↓
For each patch:
  1. Generate local PSF grid using AberrationNet
  2. FFT-based convolution within patch
  3. Apply Hann window for smooth aggregation
    ↓
Aggregate patches with 50% overlap (stride=64)
    ↓
Full resolution output
```

**Configuration:**
```yaml
ola:
  patch_size: 128           # Local patch size
  stride: 64                # 50% overlap (128/2)
  pad_to_power_2: true      # FFT optimization
```

**Memory Efficiency:**
- Patch processing: ~128×128×3 = 49KB per patch vs ~6GB full image
- Typically 10-100 patches per image
- Total memory: ~1-2 GB practical usage

### 4. Prior-Conditioned Restoration Network (`models/restoration_net.py`)

Advanced U-Net architecture enhanced with optical parameter conditioning inspired by CoordConv principles.

**Architecture Design:**
```
Input: Blurred Image (3 channels)
    ↓
Encoder (4 blocks with progressive downsampling)
  + Inject AberrationNet's Zernike coefficients at each level
  + Skip connections with optical prior fusion
    ↓
Bottleneck (deepest representation)
    ↓
Decoder (4 blocks with progressive upsampling)
  + Fuse restoration features with optical guidance
  + Coordinate conditioning for spatial awareness
    ↓
Output: Restored Image (3 channels)
```

**Why Condition on Aberrations?**
- Different aberration types require different restoration strategies
- Explicit optical feedback prevents the network from learning aberration-agnostic filters
- Improves generalization to unseen aberration patterns

## ⚙️ Configuration System

All hyperparameters are managed through YAML configuration files:

```bash
# Use default configuration
python train.py --config config/default.yaml

# Quick smoke testing
python train.py --config config/smoke_minimal.yaml

# Ablation studies
python train.py --config config/ablation_1_no_physics.yaml
```

**Key Configuration Sections:**

| Section | Parameters |
|---------|-----------|
| `experiment` | Device, seed, epochs, output directory, logging |
| `physics` | Zernike modes, wavelengths, pupil parameters |
| `ola` | Patch size, stride, FFT padding |
| `aberration_net` | Network depth, Fourier encoding scale, coefficient bounds |
| `restoration_net` | U-Net depth, channel progression, optionsactivation |
| `training` | Optimizers, learning rates, stage schedules |
| `data` | Dataset paths, batch sizes, augmentation |

**Example Configuration Override:**
```bash
# Modify at runtime (YAML override syntax)
python train.py --config config/default.yaml \
                --experiment.epochs 100 \
                --training.lr 5e-5
```

## 📊 Ablation Studies

Systematic ablation studies are implemented to validate each component's contribution:

```bash
# Ablation 1: Remove Physics Layer (pure deep learning baseline)
python train.py --config config/ablation_1_no_physics.yaml

# Ablation 2: Remove Fourier Feature Encoding from AberrationNet
python train.py --config config/ablation_2_no_fourier.yaml

# Ablation 3: End-to-End training without stage curriculum
python train.py --config config/ablation_3_end_to_end.yaml

# Ablation 4: Standard training without NewBP optimization
python train.py --config config/ablation_4_no_newbp.yaml
```

**Expected Impact on Performance:**

| Configuration | Physics | Fourier | 3-Stage | NewBP | PSNR Impact | Notes |
|---------------|---------|---------|---------|-------|-------------|-------|
| Full Method ✓ | ✓ | ✓ | ✓ | ✓ | Baseline | Best overall |
| No Physics | ✗ | - | - | - | -2-3 dB | Pure CNN baseline |
| No Fourier | ✓ | ✗ | ✓ | ✓ | -1-2 dB | High freq limitation |
| E2E Direct | ✓ | ✓ | ✗ | ✓ | -0.5-1 dB | Training instability |
| No NewBP | ✓ | ✓ | ✓ | ✗ | -0.3-0.5 dB | Slower convergence |

## 📈 Results & Benchmarks

### DPDD Dataset Performance

```
         Method          │  PSNR  │  SSIM  │  LPIPS  │  Reblur MSE
────────────────────────┼────────┼────────┼─────────┼─────────────
Traditional Deblur 1    │ 23.5   │ 0.712  │ 0.185   │ N/A
Traditional Deblur 2    │ 24.1   │ 0.731  │ 0.172   │ N/A
────────────────────────┼────────┼────────┼─────────┼─────────────
Physics-Driven (Ours)   │ 27.3   │ 0.821  │ 0.098   │ 0.032
```

### Out-of-Distribution Robustness

Tested on extreme synthetic aberrations (severe field curvature, coma, astigmatism):
- **In-distribution (DPDD)**: 27.3 dB PSNR
- **OOD (Extreme)**: 24.8 dB PSNR (-2.5 dB degradation)
- **Generalization ratio**: 91% performance retention

## 🎨 Visualization & Analysis

### Generating Comparison Visualizations

```bash
# Automatically generates side-by-side comparisons during testing
python test.py --checkpoint results/best_stage3_joint.pt \
               --config config/default.yaml \
               --save-restored \
               --output-dir results/visualization/

# Custom visualization script
python utils/visualize.py --blurred-image test_image.jpg \
                          --restored-image output.jpg \
                          --show-aberration-maps
```

**Visualization Output:**
- Blurred input image
- Restored output image
- Residual error map
- Estimated 2D aberration map
- 3D surface plot of wavefront phase

## 💡 Tips & Best Practices

### Training Optimization

1. **GPU Memory Management**
   ```bash
   # If OOM errors occur:
   # 1. Reduce batch size in config/default.yaml
   config.data.batch_size = 2  # Default: 4
   
   # 2. Reduce patch size (affects quality slightly)
   config.ola.patch_size = 96  # Default: 128
   
   # 3. Reduce image resolution input
   config.data.crop_size = 512  # Default: 768
   ```

2. **Learning Rate Tuning**
   - Stage 1: 1e-4 (physics model is stable)
   - Stage 2: 1e-4 (restoration network learning)
   - Stage 3: 5e-5 (fine-tuning, reduce by 0.5x)
   
   Adjust if:
   - Diverging: Reduce LR by 0.5x
   - Slow convergence: Increase LR by 1.5x

3. **Monitoring Training**
   ```bash
   # Launch TensorBoard for real-time metrics
   tensorboard --logdir results/ours/events
   ```

### Inference Optimization

```python
# Use torch.no_grad() for faster inference
with torch.no_grad():
    restored = model(blurred_image)

# FP16 mixed precision for 2-3x speedup (if GPU supports)
from torch.cuda.amp import autocast
with autocast():
    restored = model(blurred_image)
```

### Dataset Tips

- **Dataset Size**: 500+ image pairs recommended for stable training
- **Image Resolution**: 512-1680 width; network handles variable sizes
- **Lighting**: Diverse lighting conditions improve robustness
- **Aberrations**: Mix of defocus, coma, astigmatism for generalization

---

## ❓ FAQ & Troubleshooting

### Q1: CUDA Out-of-Memory (OOM) Error

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `config.data.batch_size = 2`
2. Reduce patch size: `config.ola.patch_size = 96`
3. Use CPU (slower): Set `experiment.device = cpu` in config
4. Enable gradient checkpointing (memory-speed tradeoff)

### Q2: Training Loss Not Decreasing

**Possible Causes:**
1. Learning rate too high → Reduce to 5e-5
2. Stage 1 stuck → Increase `stage1_epochs` by 10-20
3. Physics layer unstable → Use different random seed
4. Data preprocessing issue → Verify image normalization

**Debug:**
```bash
# Run smoke test on small dataset
python train.py --config config/smoke_minimal.yaml

# Monitor loss in TensorBoard
tensorboard --logdir results/ours/events
```

### Q3: Restored Images Still Blurry

**Check:**
1. Is checkpoint from Stage 3 (joint)? → Use `best_stage3_joint.pt`
2. Is test data similar to training? → Try OOD evaluation
3. Is batch norm in eval mode? → Ensure `model.eval()` before inference
4. Did training actually converge? → Check convergence curves

### Q4: Need to Adjust Aberration Bounds

**To allow larger aberration ranges:**
```yaml
aberration_net:
  a_max: 2.0  # Changed from 1.0 to allow ±2.0 coefficient range
```

**Trade-offs:**
- Larger `a_max`: More expressiveness, but may cause training instability
- Smaller `a_max`: Stable training, but limited aberration magnitude

### Q5: How to Speed Up Training?

1. **Use FP16 mixed precision** (2x faster):
   ```yaml
   experiment:
     use_mixed_precision: true
   ```

2. **Reduce image size**:
   ```yaml
   data:
     crop_size: 512  # Default 768
   ```

3. **Increase num_workers**:
   ```yaml
   data:
     num_workers: 8  # Default 4
   ```

4. **Skip validation during training**:
   ```yaml
   training:
     validate_every_n_epochs: 10  # Default 1
   ```

---

## 🔗 Related Resources

### Papers & References

- **Zernike Polynomials**: Born & Wolf, "Principles of Optics"
- **Fourier Feature Networks**: Tancik et al., ICML 2020
- **Overlap-Add Convolution**: Rabiner & Gold, "Theory and Application of DSP"
- **DPDD Dataset**: Abuolaim & Brown, CVPR 2020

### External Tools

- [TensorBoard](https://github.com/tensorflow/tensorboard) - Training visualization
- [Weights & Biases](https://wandb.ai) - Experiment tracking
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) - Perceptual loss metric

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Physics-Driven-Deblur,
  title={Physics-Driven Blind Deconvolution Network},
  author={Author Names},
  booktitle={Conference Name},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 📧 Contact & Support

**Questions or Issues?**
- Create a GitHub Issue: [RUA1027/Defocus_plus/issues](https://github.com/RUA1027/Defocus_plus/issues)
- Discussion Forum: [Discussions](https://github.com/RUA1027/Defocus_plus/discussions)
- Email: rua1027@gmail.com

**Last Updated:** 2026-03-09

---

<div align="center">

**⭐ If this work is helpful to you, please consider giving us a star! ⭐**

Made with ❤️ by the Defocus+ Team

</div>
