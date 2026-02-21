"""
DPDD 物理驱动复原网络 - 测试脚本
================================
使用全分辨率测试集图像进行评估，生成最终论文结果。

特性:
- 全分辨率图像评估 (1680 x 1120)
- 完整指标计算: PSNR, SSIM, LPIPS, Re-blur Error
- 结果可视化和保存
- 详细的每张图像结果报告

Usage:
    python test.py --checkpoint results/best_stage3_joint.pt --config config/default.yaml
"""

import argparse
import os
import sys
import math
import json
from datetime import datetime


def _normalize_thread_env(default_threads: int = 8):
    """规范化线程环境变量，避免非法值导致 OpenMP/BLAS 警告或异常。"""
    cpu_count = os.cpu_count() or 1
    fallback = str(max(1, min(default_threads, cpu_count)))
    thread_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )

    overrides = {}
    for key in thread_vars:
        raw = os.environ.get(key, "").strip()
        if raw == "":
            os.environ[key] = fallback
            overrides[key] = fallback
            continue

        try:
            value = int(raw)
            if value <= 0:
                raise ValueError
            os.environ[key] = str(value)
        except (TypeError, ValueError):
            os.environ[key] = fallback
            overrides[key] = fallback

    return fallback, overrides


# 在导入 torch 前先规范化线程变量，避免底层库读取到非法值
_THREAD_FALLBACK, _THREAD_ENV_OVERRIDES = _normalize_thread_env()

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from utils.model_builder import build_models_from_config, build_test_dataloader_from_config
from utils.metrics import PerformanceEvaluator
from trainer import DualBranchTrainer


def _sanitize_for_json(obj):
    """递归地将 NaN/Inf 替换为 None，使输出符合 JSON 标准（RFC 8259）。"""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def count_params(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def compute_model_stats(restoration_net, physical_layer, device, image_height, image_width):
    """统计 Params 和 FLOPs (如可用)"""
    stats = {
        'restoration_params': count_params(restoration_net),
        'physical_params': count_params(physical_layer) if physical_layer is not None else 0,
        'restoration_flops': None,
        'physical_flops': None
    }
    stats['total_params'] = stats['restoration_params'] + stats['physical_params']

    evaluator = PerformanceEvaluator(device=device)
    input_shape = (1, 3, image_height, image_width)

    benchmark_restoration = restoration_net
    if physical_layer is not None and getattr(restoration_net, 'n_coeffs', 0) > 0:
        benchmark_restoration = evaluator._build_injection_aware_benchmark_model(
            restoration_net,
            physical_layer,
            device,
            injection_grid_size=16
        )

    restoration_flops = evaluator._try_flops(benchmark_restoration, device, input_shape=input_shape)
    physical_flops = evaluator._try_flops(physical_layer, device, input_shape=input_shape) if physical_layer is not None else None

    if restoration_flops is not None:
        stats['restoration_flops'] = float(restoration_flops)
    if physical_flops is not None:
        stats['physical_flops'] = float(physical_flops)

    if stats['restoration_flops'] is not None and stats['physical_flops'] is not None:
        stats['total_flops'] = stats['restoration_flops'] + stats['physical_flops']
    else:
        stats['total_flops'] = None

    return stats


def save_comparison_image(blur, sharp_gt, restored, reblur, save_path):
    """
    保存对比图像：模糊输入 | 复原结果 | 真实清晰 | 重模糊
    
    Args:
        blur: 模糊图像 tensor [C, H, W]
        sharp_gt: 真实清晰图像 tensor [C, H, W]
        restored: 复原图像 tensor [C, H, W]
        reblur: 重模糊图像 tensor [C, H, W]
        save_path: 保存路径
    """
    def tensor_to_pil(t):
        t = t.clamp(0, 1).cpu().numpy()
        t = (t * 255).astype(np.uint8)
        if t.shape[0] == 3:
            t = t.transpose(1, 2, 0)
        return Image.fromarray(t)
    
    # 转换为 PIL 图像
    blur_pil = tensor_to_pil(blur)
    sharp_pil = tensor_to_pil(sharp_gt)
    restored_pil = tensor_to_pil(restored)
    reblur_pil = tensor_to_pil(reblur)
    
    # 获取尺寸
    w, h = blur_pil.size
    
    # 创建拼接图像 (2x2 布局)
    combined = Image.new('RGB', (w * 2, h * 2))
    combined.paste(blur_pil, (0, 0))
    combined.paste(restored_pil, (w, 0))
    combined.paste(sharp_pil, (0, h))
    combined.paste(reblur_pil, (w, h))
    
    combined.save(save_path)


def save_single_result(restored, save_path):
    """保存单张复原结果"""
    restored = restored.clamp(0, 1).cpu().numpy()
    restored = (restored * 255).astype(np.uint8)
    if restored.shape[0] == 3:
        restored = restored.transpose(1, 2, 0)
    Image.fromarray(restored).save(save_path)


def main():
    parser = argparse.ArgumentParser(description='DPDD Testing Script')
    parser.add_argument('--checkpoint', '-ckpt', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root directory for OOD/custom datasets (default: use config value)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results (default: results/test_<timestamp>)')
    parser.add_argument('--save-images', action='store_true',
                        help='Save comparison images')
    parser.add_argument('--save-restored', action='store_true',
                        help='Save restored images only')
    args = parser.parse_args()

    # 同步 PyTorch CPU 线程设置
    torch_threads = int(os.environ.get("OMP_NUM_THREADS", _THREAD_FALLBACK))
    torch.set_num_threads(torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, torch_threads)))
        except RuntimeError:
            # 该接口在某些场景下只能设置一次，失败时保持默认即可
            pass

    if _THREAD_ENV_OVERRIDES:
        overrides = ", ".join([f"{k}={v}" for k, v in _THREAD_ENV_OVERRIDES.items()])
        print(f"Normalized thread env vars: {overrides}")
    
    # 加载配置
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # 如果指定了 --data-root，覆盖配置中的数据集路径
    if args.data_root is not None:
        config.data.data_root = args.data_root
        print(f"Override data root to: {args.data_root}")
    
    # 设置设备
    device = config.experiment.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # 创建输出目录
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.experiment.output_dir, f"test_{timestamp}")
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    if args.save_restored:
        os.makedirs(os.path.join(output_dir, 'restored'), exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # 构建模型
    print("\nBuilding models...")
    zernike_gen, aberration_net, restoration_net, physical_layer = \
        build_models_from_config(config, device)
    
    # 加载检查点
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    restoration_state = checkpoint['restoration_net']
    # Remove thop-injected profiling buffers if present (e.g., total_ops/total_params)
    restoration_state = {
        k: v for k, v in restoration_state.items()
        if 'total_ops' not in k and 'total_params' not in k
    }
    restoration_net.load_state_dict(restoration_state, strict=True)
    if aberration_net is not None and 'aberration_net' in checkpoint:
        aberration_net.load_state_dict(checkpoint['aberration_net'])
    
    restoration_net.eval()
    if physical_layer is not None:
        physical_layer.eval()

    # 统计模型复杂度
    print("\nComputing model Params/FLOPs...")
    model_stats = compute_model_stats(
        restoration_net,
        physical_layer,
        device,
        config.data.image_height,
        config.data.image_width
    )
    print(
        f"  Restoration Params: {model_stats['restoration_params']:,} | "
        f"Physical Params: {model_stats['physical_params']:,} | "
        f"Total Params: {model_stats['total_params']:,}"
    )
    if model_stats['total_flops'] is not None:
        print(
            f"  Restoration FLOPs: {model_stats['restoration_flops']:.3e} | "
            f"Physical FLOPs: {model_stats['physical_flops']:.3e} | "
            f"Total FLOPs: {model_stats['total_flops']:.3e}"
        )
    else:
        print("  FLOPs: unavailable (thop failed or unsupported ops)")
    
    # 打印检查点信息
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'stage' in checkpoint:
        print(f"  Checkpoint stage: {checkpoint['stage']}")
    if 'val_metrics' in checkpoint:
        print(f"  Validation metrics at checkpoint:")
        for k, v in checkpoint['val_metrics'].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
    
    # 构建测试集 DataLoader
    print("\nLoading test dataset (full resolution)...")
    test_loader = build_test_dataloader_from_config(config)
    # Pylance 可能无法推断 dataset 具有 __len__，但 DPDDDataset 确实实现了它
    print(f"  Test set size: {len(test_loader.dataset)}")  # type: ignore
    
    # 创建评估器
    evaluator = PerformanceEvaluator(device=device)
    
    # 评估
    print("\n" + "="*60)
    print("Running Full-Resolution Evaluation on Test Set")
    print("="*60)
    
    results = []
    psnr_total = 0.0
    ssim_total = 0.0
    mae_total = 0.0
    lpips_total = 0.0
    use_physical_layer = getattr(config.experiment, 'use_physical_layer', True)
    injection_grid_size = getattr(config.restoration_net, 'injection_grid_size', 16)
    reblur_total = 0.0
    n = 0
    lpips_count = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            blur = batch['blur'].to(device)
            sharp = batch['sharp'].to(device)
            crop_info = batch.get('crop_info', None)
            filename = batch.get('filename', [f"image_{n}"])[0]
            
            if crop_info is not None:
                crop_info = crop_info.to(device)
            
            # 生成物理系数图（如有）
            coeffs_map = None
            if use_physical_layer and physical_layer is not None and getattr(restoration_net, 'n_coeffs', 0) > 0:
                B, _, H, W = blur.shape
                coeffs_map = physical_layer.generate_coeffs_map(
                    H, W, device, grid_size=injection_grid_size,
                    crop_info=crop_info, batch_size=B
                )
            
            # 复原
            restored = restoration_net(blur, coeffs_map=coeffs_map)
            
            # 重模糊
            if use_physical_layer and physical_layer is not None:
                reblur = physical_layer(restored, crop_info=crop_info)
            else:
                reblur = None
            
            # 计算指标
            psnr = evaluator._psnr(restored, sharp).item()
            ssim = evaluator._ssim(restored, sharp).item()
            mae = evaluator._mae(restored, sharp).item()
            lpips = evaluator._lpips_score(restored, sharp)
            lpips_val = lpips.item() if lpips is not None else float('nan')
            reblur_mse = torch.nn.functional.mse_loss(reblur, blur).item() if reblur is not None else float('nan')
            
            # 记录结果
            result = {
                'filename': filename,
                'PSNR': psnr,
                'SSIM': ssim,
                'MAE': mae,
                'LPIPS': lpips_val,
                'Reblur_MSE': reblur_mse
            }
            results.append(result)
            
            psnr_total += psnr
            ssim_total += ssim
            mae_total += mae
            if not math.isnan(lpips_val):
                lpips_total += lpips_val
                lpips_count += 1
            if reblur is not None:
                reblur_total += reblur_mse
            n += 1
            
            # 更新进度条
            pbar.set_postfix({
                'PSNR': f"{psnr:.2f}",
                'SSIM': f"{ssim:.4f}"
            })
            
            # 保存图像
            if args.save_images and reblur is not None:
                save_path = os.path.join(output_dir, 'comparisons', 
                                         f"{os.path.splitext(filename)[0]}_comparison.png")
                save_comparison_image(
                    blur[0], sharp[0], restored[0], reblur[0], save_path
                )
            
            if args.save_restored:
                save_path = os.path.join(output_dir, 'restored', 
                                         f"{os.path.splitext(filename)[0]}_restored.png")
                save_single_result(restored[0], save_path)
    
    # 计算平均指标
    avg_metrics = {
        'PSNR': psnr_total / max(n, 1),
        'SSIM': ssim_total / max(n, 1),
        'MAE': mae_total / max(n, 1),
        'LPIPS': lpips_total / lpips_count if lpips_count > 0 else float('nan'),
        'Reblur_MSE': (reblur_total / max(n, 1)) if use_physical_layer else float('nan'),
        'Params_M': model_stats['total_params'] / 1e6,
        'FLOPs_G': model_stats['total_flops'] / 1e9 if model_stats['total_flops'] is not None else float('nan'),
        'Num_Images': n
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    try:
        from tabulate import tabulate
        rows = []
        for k, v in avg_metrics.items():
            if isinstance(v, float) and math.isnan(v):
                rows.append([k, "NaN"])
            elif isinstance(v, float):
                rows.append([k, f"{v:.6f}"])
            else:
                rows.append([k, str(v)])
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))
    except ImportError:
        for k, v in avg_metrics.items():
            if isinstance(v, float) and math.isnan(v):
                print(f"  {k}: NaN")
            elif isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
    
    # 保存详细结果
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(_sanitize_for_json({
            'average_metrics': avg_metrics,
            'model_stats': model_stats,
            'per_image_results': results,
            'checkpoint': args.checkpoint,
            'config': args.config
        }), f, indent=2, allow_nan=False)
    print(f"\n✓ Detailed results saved to: {results_path}")
    
    # 保存摘要为 CSV
    csv_path = os.path.join(output_dir, 'test_results.csv')
    with open(csv_path, 'w') as f:
        f.write("filename,PSNR,SSIM,MAE,LPIPS,Reblur_MSE\n")
        for r in results:
            def _fmt(v):
                return "" if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else f"{v:.6f}"
            f.write(f"{r['filename']},{_fmt(r['PSNR'])},{_fmt(r['SSIM'])},"
                    f"{_fmt(r['MAE'])},{_fmt(r['LPIPS'])},{_fmt(r['Reblur_MSE'])}\n")
    print(f"✓ CSV results saved to: {csv_path}")
    
    # 打印最佳/最差样本
    sorted_by_psnr = sorted(results, key=lambda x: x['PSNR'], reverse=True)
    print("\nTop 5 images (by PSNR):")
    for r in sorted_by_psnr[:5]:
        print(f"  {r['filename']}: PSNR={r['PSNR']:.2f}, SSIM={r['SSIM']:.4f}")
    
    print("\nBottom 5 images (by PSNR):")
    for r in sorted_by_psnr[-5:]:
        print(f"  {r['filename']}: PSNR={r['PSNR']:.2f}, SSIM={r['SSIM']:.4f}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
