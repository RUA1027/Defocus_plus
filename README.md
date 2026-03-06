# Physics-Driven Blind Deconvolution Network

This repository provides the official PyTorch implementation of the **Physics-Driven Blind Deconvolution Network**. The framework is specifically designed to restore images degraded by spatially varying optical aberrations and severe defocus blur. By parameterizing wavefront aberrations using Zernike polynomials and leveraging differentiable Fourier optics, the model accurately translates these mathematical representations into spatially varying Point Spread Functions (PSFs). This approach seamlessly couples deep learning with physical optics fundamentals, achieving highly generalized and interpretable image restoration.

## Key Features

- **Dual-Branch Architecture**: Consists of a Restoration Network (`RestorationNet`, based on an augmented U-Net) and an Optical Identification Network (`AberrationNet`, based on an MLP).
- **Differentiable Physics Layer**: Systematically maps Zernike coefficients → Wavefront Phase → Pupil Function → PSFs → Spatially Varying Convolution. This is implemented with high computational efficiency via an Overlap-Add (OLA) strategy and Fast Fourier Transform (FFT).
- **Three-Stage Decoupled Training**: 
  1. **Physics Prior Pre-training**: Warms up the aberration estimator.
  2. **Restoration Learning**: Trains the restoration U-Net with fixed optical parameters.
  3. **Joint Fine-tuning**: End-to-end optimization. 
  This curriculum ensures both optical metric accuracy and stable convergence.
- **Self-Supervised Physical Constraint**: Employs a Reblurring Consistency Loss (`Loss = MSE(Ŷ, Y)`) to facilitate unsupervised PSF estimation without requiring ground-truth blur kernels.
- **Extreme OOD Evaluation**: Includes customized scripts to generate Out-Of-Distribution (OOD) datasets featuring extreme, synthetically induced aberrations (e.g., severe field curvature or coma) for robust generalization testing.


## System Architecture

```text
Blurred Input (Y) 
    │
    ├─▶ RestorationNet (U-Net) ──▶ Restored Image (X̂) ──▶ L1 Loss with Ground Truth (X_gt)
    │                                  │
    └─▶ AberrationNet (MLP)            │
             │                         │
        Zernike Coefficients           │
             │                         │
        PSF Generation (FFT)           │
             │                         │
        Spatially Varying Blur ◀───────┘
             │
        Reblurred Image (Ŷ) ──▶ MSE Loss with Blurred Input (Y)
```


## Installation

**Requirements:**
- Python 3.8+
- PyTorch 1.10+
- CUDA-enabled GPU (Highly recommended for FFT operations)

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/RUA1027/Defocus_plus.git
cd Defocus_plus
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation (DPDD Dataset)

Download the Dual-Pixel Defocus Deblurring (DPDD) dataset and place it in the `data/dd_dp_dataset_png/` directory. The structure should be organized as follows:
```text
data/dd_dp_dataset_png/
    train_c/
        source/    # Blurred input images
        target/    # Ground truth sharp images
    val_c/
        source/
        target/
    test_c/
        source/
        target/
```
*Note: The dataloader fetches images directly from the `source/` and `target/` directories, handling formatting internally without requiring manual prior scaling.*

### 2. Training Pipeline

The training paradigm is driven by a **Three-Stage Decoupled Strategy**, orchestrated by the `DualBranchTrainer` inside `trainer.py`.

```bash
# Standard training using the default configuration
python train.py --config config/default.yaml

# Resume training from a specific checkpoint
python train.py --resume results/latest.pt

# Execute a specific training stage (options: '1', '2', '3', or 'all')
python train.py --stage all
```

**Detailed Stage Breakdown:**
1. **Stage 1 (Physics Only)**: Trains the `AberrationNet` exclusively. It fits the optical distortion distributions by heavily relying on the reblurring consistency constraint. *Validation Metric: Re-blur MSE.*
2. **Stage 2 (Restoration)**: Freezes the physics layer to stabilize the gradients, subsequently training the `RestorationNet` against the sharp ground-truth images. *Validation Metrics: PSNR & SSIM.*
3. **Stage 3 (Joint Fine-tuning)**: Unfreezes all components and conducts an end-to-end joint training phase with a decayed learning rate (typically halved). *Validation Metric: Comprehensive combined score.*

### 3. Evaluation and Testing

To evaluate the model on the full-resolution (e.g., 1680 x 1120) DPDD test set:

```bash
# Test using the jointly fine-tuned model and export the restored images
python test.py --checkpoint results/best_stage3_joint.pt --config config/default.yaml --save-restored

# Evaluate numerical metrics exclusively (faster, skips saving images)
python test.py --checkpoint results/best_stage2_psnr.pt
```
The test outputs are stored in the `results/` directory. This includes a comprehensively formatted `test_results.csv` tracking PSNR, SSIM, LPIPS, and Reblur MSE, alongside side-by-side visual comparisons in the `comparisons/` folder.

### 4. Extreme OOD Dataset Generation

To strictly evaluate the theoretical robustness of the network under unprecedented optical conditions, we provide an automated script to synthesize Out-Of-Distribution datasets presenting severe spatially varying defects (like extreme field curvature).

```bash
# Generates standard/extreme datasets into the 'data/extreme aberration' directory
python generate_extreme_ood_dataset.py
```

## Modular Codebase Structure

```text
Defocus_plus/
├── config/                   # Centralized YAML configuration files
│   ├── default.yaml          # Standard hyperparameter definitions
│   ├── ablation_1_no_physics.yaml # Ablation: Purges the physics layer
│   └── ...
├── data/                     # Data hosting and preprocessing directives
├── models/                   # Core architectural algorithms
│   ├── aberration_net.py     # Aberration MLP incorporating Fourier Feature Encoding
│   ├── physical_layer.py     # Differentiable overlapping convolution (Overlap-Add)
│   ├── restoration_net.py    # Generative U-Net incorporating optical priors
│   └── zernike.py            # Optical kernel engine generating Zernike polynomials 
├── utils/                    # Shared utilities (dataset loaders, metrics, logging)
├── generate_extreme_ood_dataset.py # Script for generating robustness stress-tests
├── train.py                  # Primary training entry point
├── test.py                   # Primary evaluation entry point
├── trainer.py                # Implementations of the multi-stage training mechanisms
└── README.md                 # Technical documentation entry
```

## Core Components Deep-Dive

### 1. Zernike PSF Engine (`models/zernike.py`)
Translates abstract polynomial coefficients into dense PSF matrices. It mathematically supports high-order optical aberrations (using 36 Noll modes) via a differentiable physical pipeline: mapping discrete phases to continuous pupil functions, and deriving complex PSFs via Fourier domain integration.

### 2. Aberration Network (`models/aberration_net.py`)
Constructed as an MLP mapping normalized 2D image coordinates to granular sets of 36 continuous Zernike coefficients. Crucially, it integrates **Fourier Feature Encoding** (FFE) to overcome spectral bias, exponentially enhancing its capability to memorize high-frequency spatially varying aberrations mapping the full lens field.

### 3. Differentiable Physical Layer (`models/physical_layer.py`)
Houses a meticulously optimized **spatially varying convolution module**. Operating over large full-resolution image arrays natively is computationally prohibitive. Therefore, this component relies on the Overlap-Add (OLA) mathematical approach: dividing feature maps into localized patches, applying FFT-based targeted localized convolutions per patch based on the dynamically formulated PSF grids, and subsequently aggregating via continuous Hann windows to guarantee boundary smoothness without artifacts.

### 4. Prior-Conditioned Restoration Network (`models/restoration_net.py`)
An advanced encoding-decoding U-Net model enhanced with explicit physical coordinate conditioning (CoordConv principles). It fuses the blurry input visual states dynamically correlated alongside the continuous Zernike maps formulated by the `AberrationNet`. This robust topological fusion instructs the restorer to implicitly decode exact spatially dependent pixel transformations required for high-fidelity deblurring.

## Ablation Studies & Custom Configurations

The `config/` directory manages independent YAML declarations meant directly for conducting replicable ablation analyses:
- `ablation_1_no_physics.yaml`: Disables the Zernike-Fourier branch entirely, forcing the architecture to emulate standard data-driven baselines.
- `ablation_2_no_fourier.yaml`: Removes Fourier Feature parameters from the `AberrationNet` to quantify its necessity for high-frequency coordinate mapping.
- `ablation_3_end_to_end.yaml`: Completely overrides the standard three-stage strategy to perform brute-force direct end-to-end training.

To seamlessly execute any specific topological modification defined in configurations:
```bash
python train.py --config config/ablation_1_no_physics.yaml
```
