# 模型测试结果汇总

本文档汇总了主体实验模型（Ours）以及三个消融实验模型在不同数据集上的测试结果。

**模型说明：**
- **Ours (Main Model)**: 完整的主体实验模型。
- **Ablation 1 (No Physics)**: 消融实验模型 1，移除了物理层（Physical Layer）。
- **Ablation 2 (No Fourier)**: 消融实验模型 2，移除了傅里叶变换模块（Fourier Transform）。
- **Ablation 3 (End-to-End)**: 消融实验模型 3，采用端到端（End-to-End）训练策略，不使用分步或特定模块的独立监督。

---

## 1. DPDD Canon 数据集测试结果

| 监测数据 (Metrics) | Ours (Main Model) | Ablation 1 (No Physics) | Ablation 2 (No Fourier) | Ablation 3 (End-to-End) |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR** | 25.44 | 25.47 | 25.31 | 25.28 |
| **SSIM** | 0.7925 | 0.7945 | 0.7903 | 0.7881 |
| **MAE** | 10.35 | 10.06 | 10.38 | 10.62 |
| **LPIPS** | 0.2416 | 0.2202 | 0.2317 | 0.2363 |
| **Reblur_MSE** | 0.002019 | N/A | 0.002257 | 0.002147 |
| **Params_M** | 18.70 | 17.26 | 18.43 | 18.70 |
| **FLOPs_G** | 1.17e-06 | N/A | 1.17e-06 | 1.17e-06 |
| **Num_Images** | 76 | 76 | 76 | 76 |

---

## 2. DPDD Pixel 数据集测试结果

| 监测数据 (Metrics) | Ours (Main Model) | Ablation 1 (No Physics) | Ablation 2 (No Fourier) | Ablation 3 (End-to-End) |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR** | 35.74 | 34.15 | 35.46 | 35.09 |
| **SSIM** | 0.9743 | 0.9654 | 0.9726 | 0.9698 |
| **MAE** | 3.23 | 4.23 | 3.57 | 3.71 |
| **LPIPS** | 0.1615 | 0.1730 | 0.1473 | 0.1665 |
| **Reblur_MSE** | 0.000306 | 0.000419 | 0.000325 | 0.000365 |
| **Params_M** | 18.70 | 18.70 | 18.43 | 18.70 |
| **FLOPs_G** | 1.17e-06 | 1.17e-06 | 1.17e-06 | 1.17e-06 |
| **Num_Images** | 13 | 13 | 13 | 13 |

---

## 3. Extreme Aberration 数据集测试结果

| 监测数据 (Metrics) | Ours (Main Model) | Ablation 1 (No Physics) | Ablation 2 (No Fourier) | Ablation 3 (End-to-End) |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR** | 21.04 | 20.86 | 20.84 | 20.83 |
| **SSIM** | 0.5729 | 0.5589 | 0.5639 | 0.5659 |
| **MAE** | 15.69 | 16.20 | 16.36 | 16.35 |
| **LPIPS** | 0.6367 | 0.5835 | 0.6086 | 0.6198 |
| **Reblur_MSE** | 0.000733 | 0.001137 | 0.000974 | 0.001265 |
| **Params_M** | 18.70 | 18.70 | 18.43 | 18.70 |
| **FLOPs_G** | 1.17e-06 | 1.17e-06 | 1.17e-06 | 1.17e-06 |
| **Num_Images** | 74 | 74 | 74 | 74 |
