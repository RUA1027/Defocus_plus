import argparse
import os
import sys
import math
import json
from datetime import datetime

def _normalize_thread_env(default_threads: int=8):
    cpu_count = os.cpu_count() or 1
    fallback = str(max(1, min(default_threads, cpu_count)))
    thread_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
    overrides = {}
    for key in thread_vars:
        raw = os.environ.get(key, '').strip()
        if raw == '':
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
    return (fallback, overrides)
(_THREAD_FALLBACK, _THREAD_ENV_OVERRIDES) = _normalize_thread_env()
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config
from utils.model_builder import build_models_from_config, build_test_dataloader_from_config, build_test_dataloader_by_type, get_supported_dataset_types
from utils.metrics import PerformanceEvaluator
from trainer import DualBranchTrainer

def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj

def count_params(model):
    return sum((p.numel() for p in model.parameters()))

def compute_model_stats(restoration_net, physical_layer, device, image_height, image_width):
    stats = {'restoration_params': count_params(restoration_net), 'physical_params': count_params(physical_layer) if physical_layer is not None else 0, 'restoration_flops': None, 'physical_flops': None}
    stats['total_params'] = stats['restoration_params'] + stats['physical_params']
    evaluator = PerformanceEvaluator(device=device)
    input_shape = (1, 3, image_height, image_width)
    benchmark_restoration = restoration_net
    if physical_layer is not None and getattr(restoration_net, 'n_coeffs', 0) > 0:
        benchmark_restoration = evaluator._build_injection_aware_benchmark_model(restoration_net, physical_layer, device, injection_grid_size=16)
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

    def tensor_to_pil(t):
        t = t.clamp(0, 1).cpu().numpy()
        t = (t * 255).astype(np.uint8)
        if t.shape[0] == 3:
            t = t.transpose(1, 2, 0)
        return Image.fromarray(t)
    blur_pil = tensor_to_pil(blur)
    sharp_pil = tensor_to_pil(sharp_gt)
    restored_pil = tensor_to_pil(restored)
    reblur_pil = tensor_to_pil(reblur)
    (w, h) = blur_pil.size
    combined = Image.new('RGB', (w * 2, h * 2))
    combined.paste(blur_pil, (0, 0))
    combined.paste(restored_pil, (w, 0))
    combined.paste(sharp_pil, (0, h))
    combined.paste(reblur_pil, (w, h))
    combined.save(save_path)

def save_single_result(restored, save_path):
    restored = restored.clamp(0, 1).cpu().numpy()
    restored = (restored * 255).astype(np.uint8)
    if restored.shape[0] == 3:
        restored = restored.transpose(1, 2, 0)
    Image.fromarray(restored).save(save_path)

def main():
    supported_types = get_supported_dataset_types()
    parser = argparse.ArgumentParser(description='DPDD Testing Script (Multi-Dataset)')
    parser.add_argument('--checkpoint', '-ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--dataset-type', '-dt', type=str, default='dpdd',
                        choices=supported_types,
                        help=f'Dataset type to test on ({"/".join(supported_types)})')
    parser.add_argument('--data-root', type=str, default=None, help='Override data root directory for OOD/custom datasets (default: use config value)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory for results (default: results/test_<dataset_type>_<timestamp>)')
    parser.add_argument('--save-images', action='store_true', help='Save comparison images (requires GT)')
    parser.add_argument('--save-restored', action='store_true', help='Save restored images only')
    args = parser.parse_args()
    torch_threads = int(os.environ.get('OMP_NUM_THREADS', _THREAD_FALLBACK))
    torch.set_num_threads(torch_threads)
    if hasattr(torch, 'set_num_interop_threads'):
        try:
            torch.set_num_interop_threads(max(1, min(4, torch_threads)))
        except RuntimeError:
            pass
    if _THREAD_ENV_OVERRIDES:
        overrides = ', '.join([f'{k}={v}' for (k, v) in _THREAD_ENV_OVERRIDES.items()])
        print(f'Normalized thread env vars: {overrides}')
    print(f'Loading config from {args.config}...')
    config = load_config(args.config)
    if args.data_root is not None:
        print(f'Override data root to: {args.data_root}')
    device = config.experiment.device
    if device == 'cuda' and (not torch.cuda.is_available()):
        print('Warning: CUDA not available, using CPU')
        device = 'cpu'
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config.experiment.output_dir, f'test_{args.dataset_type}_{timestamp}')
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    if args.save_restored:
        os.makedirs(os.path.join(output_dir, 'restored'), exist_ok=True)
    print(f'Device: {device}')
    print(f'Output directory: {output_dir}')
    print('\nBuilding models...')
    (zernike_gen, aberration_net, restoration_net, physical_layer) = build_models_from_config(config, device)
    print(f'\nLoading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    restoration_state = checkpoint['restoration_net']
    restoration_state = {k: v for (k, v) in restoration_state.items() if 'total_ops' not in k and 'total_params' not in k}
    restoration_net.load_state_dict(restoration_state, strict=True)
    if aberration_net is not None and 'aberration_net' in checkpoint:
        aberration_net.load_state_dict(checkpoint['aberration_net'])
    if physical_layer is not None and 'physical_layer' in checkpoint:
        physical_layer.load_state_dict(checkpoint['physical_layer'], strict=False)
    restoration_net.eval()
    if physical_layer is not None:
        physical_layer.eval()
    print('\nComputing model Params/FLOPs...')
    model_stats = compute_model_stats(restoration_net, physical_layer, device, config.data.image_height, config.data.image_width)
    print(f"  Restoration Params: {model_stats['restoration_params']:,} | Physical Params: {model_stats['physical_params']:,} | Total Params: {model_stats['total_params']:,}")
    if model_stats['total_flops'] is not None:
        print(f"  Restoration FLOPs: {model_stats['restoration_flops']:.3e} | Physical FLOPs: {model_stats['physical_flops']:.3e} | Total FLOPs: {model_stats['total_flops']:.3e}")
    else:
        print('  FLOPs: unavailable (thop failed or unsupported ops)')
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'stage' in checkpoint:
        print(f"  Checkpoint stage: {checkpoint['stage']}")
    if 'val_metrics' in checkpoint:
        print(f'  Validation metrics at checkpoint:')
        for (k, v) in checkpoint['val_metrics'].items():
            if isinstance(v, float):
                print(f'    {k}: {v:.6f}')
    print(f'\nLoading test dataset: {args.dataset_type} (full resolution)...')
    test_loader, has_gt = build_test_dataloader_by_type(
        dataset_type=args.dataset_type, config=config,
        data_root_override=args.data_root)
    print(f'  Test set size: {len(test_loader.dataset)}')
    print(f'  Has ground truth: {has_gt}')
    if not has_gt:
        args.save_restored = True
        print('  [Auto] --save-restored enabled (no GT dataset)')
    if args.save_restored:
        os.makedirs(os.path.join(output_dir, 'restored'), exist_ok=True)
    evaluator = PerformanceEvaluator(device=device)
    print('\n' + '=' * 60)
    print(f'Running Full-Resolution Evaluation on [{args.dataset_type.upper()}]')
    print('=' * 60)
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
        pbar = tqdm(test_loader, desc=f'Testing [{args.dataset_type}]')
        for batch in pbar:
            blur = batch['blur'].to(device)
            sharp_raw = batch['sharp']
            sharp = sharp_raw.to(device) if sharp_raw is not None and torch.is_tensor(sharp_raw) else None
            crop_info = batch.get('crop_info', None)
            filename = batch.get('filename', [f'image_{n}'])[0]
            if crop_info is not None:
                crop_info = crop_info.to(device)
            coeffs_map = None
            if use_physical_layer and physical_layer is not None and (getattr(restoration_net, 'n_coeffs', 0) > 0):
                (B, _, H, W) = blur.shape
                coeffs_map = physical_layer.generate_coeffs_map(H, W, device, grid_size=injection_grid_size, crop_info=crop_info, batch_size=B)
            restored = restoration_net(blur, coeffs_map=coeffs_map)
            if use_physical_layer and physical_layer is not None:
                reblur = physical_layer(restored, crop_info=crop_info)
            else:
                reblur = None
            reblur_mse = torch.nn.functional.mse_loss(reblur, blur).item() if reblur is not None else float('nan')
            if has_gt and sharp is not None:
                psnr = evaluator._psnr(restored, sharp).item()
                ssim = evaluator._ssim(restored, sharp).item()
                mae = evaluator._mae(restored, sharp).item()
                lpips_score = evaluator._lpips_score(restored, sharp)
                lpips_val = lpips_score.item() if lpips_score is not None else float('nan')
            else:
                psnr = float('nan')
                ssim = float('nan')
                mae = float('nan')
                lpips_val = float('nan')
            result = {'filename': filename, 'PSNR': psnr, 'SSIM': ssim, 'MAE': mae, 'LPIPS': lpips_val, 'Reblur_MSE': reblur_mse}
            results.append(result)
            if has_gt:
                psnr_total += psnr
                ssim_total += ssim
                mae_total += mae
                if not math.isnan(lpips_val):
                    lpips_total += lpips_val
                    lpips_count += 1
            if reblur is not None:
                reblur_total += reblur_mse
            n += 1
            if has_gt:
                pbar.set_postfix({'PSNR': f'{psnr:.2f}', 'SSIM': f'{ssim:.4f}'})
            else:
                pbar.set_postfix({'Reblur_MSE': f'{reblur_mse:.6f}' if not math.isnan(reblur_mse) else 'N/A'})
            if args.save_images and has_gt and reblur is not None and sharp is not None:
                save_path = os.path.join(output_dir, 'comparisons', f'{os.path.splitext(filename)[0]}_comparison.png')
                save_comparison_image(blur[0], sharp[0], restored[0], reblur[0], save_path)
            if args.save_restored:
                save_path = os.path.join(output_dir, 'restored', f'{os.path.splitext(filename)[0]}_restored.png')
                save_single_result(restored[0], save_path)
    if has_gt:
        avg_metrics = {
            'PSNR': psnr_total / max(n, 1),
            'SSIM': ssim_total / max(n, 1),
            'MAE': mae_total / max(n, 1),
            'LPIPS': lpips_total / lpips_count if lpips_count > 0 else float('nan'),
            'Reblur_MSE': reblur_total / max(n, 1) if use_physical_layer else float('nan'),
            'Params_M': model_stats['total_params'] / 1000000.0,
            'FLOPs_G': model_stats['total_flops'] if model_stats['total_flops'] is not None else float('nan'),
            'Num_Images': n,
        }
    else:
        avg_metrics = {
            'Reblur_MSE': reblur_total / max(n, 1) if use_physical_layer else float('nan'),
            'Params_M': model_stats['total_params'] / 1000000.0,
            'FLOPs_G': model_stats['total_flops'] if model_stats['total_flops'] is not None else float('nan'),
            'Num_Images': n,
        }
    print('\n' + '=' * 60)
    print(f'Test Results Summary [{args.dataset_type.upper()}]')
    print('=' * 60)
    try:
        from tabulate import tabulate
        rows = []
        for (k, v) in avg_metrics.items():
            if isinstance(v, float) and math.isnan(v):
                rows.append([k, 'NaN'])
            elif isinstance(v, float):
                rows.append([k, f'{v:.6f}'])
            else:
                rows.append([k, str(v)])
        print(tabulate(rows, headers=['Metric', 'Value'], tablefmt='github'))
    except ImportError:
        for (k, v) in avg_metrics.items():
            if isinstance(v, float) and math.isnan(v):
                print(f'  {k}: NaN')
            elif isinstance(v, float):
                print(f'  {k}: {v:.6f}')
            else:
                print(f'  {k}: {v}')
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(_sanitize_for_json({
            'dataset_type': args.dataset_type,
            'has_gt': has_gt,
            'average_metrics': avg_metrics,
            'model_stats': model_stats,
            'per_image_results': results,
            'checkpoint': args.checkpoint,
            'config': args.config,
        }), f, indent=2, allow_nan=False)
    print(f'\n[OK] Detailed results saved to: {results_path}')
    csv_path = os.path.join(output_dir, 'test_results.csv')
    with open(csv_path, 'w') as f:
        if has_gt:
            f.write('filename,PSNR,SSIM,MAE,LPIPS,Reblur_MSE\n')
            for r in results:
                def _fmt(v):
                    return '' if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f'{v:.6f}'
                f.write(f"{r['filename']},{_fmt(r['PSNR'])},{_fmt(r['SSIM'])},{_fmt(r['MAE'])},{_fmt(r['LPIPS'])},{_fmt(r['Reblur_MSE'])}\n")
        else:
            f.write('filename,Reblur_MSE\n')
            for r in results:
                rmse = r['Reblur_MSE']
                rmse_str = f'{rmse:.6f}' if not (isinstance(rmse, float) and math.isnan(rmse)) else ''
                f.write(f"{r['filename']},{rmse_str}\n")
    print(f'[OK] CSV results saved to: {csv_path}')
    if has_gt:
        sorted_by_psnr = sorted(results, key=lambda x: x['PSNR'], reverse=True)
        print('\nTop 5 images (by PSNR):')
        for r in sorted_by_psnr[:5]:
            print(f"  {r['filename']}: PSNR={r['PSNR']:.2f}, SSIM={r['SSIM']:.4f}")
        print('\nBottom 5 images (by PSNR):')
        for r in sorted_by_psnr[-5:]:
            print(f"  {r['filename']}: PSNR={r['PSNR']:.2f}, SSIM={r['SSIM']:.4f}")
    else:
        valid_reblur = [r for r in results if not math.isnan(r['Reblur_MSE'])]
        if valid_reblur:
            sorted_by_reblur = sorted(valid_reblur, key=lambda x: x['Reblur_MSE'])
            print('\nTop 5 images (lowest Reblur_MSE):')
            for r in sorted_by_reblur[:5]:
                print(f"  {r['filename']}: Reblur_MSE={r['Reblur_MSE']:.6f}")
            print('\nBottom 5 images (highest Reblur_MSE):')
            for r in sorted_by_reblur[-5:]:
                print(f"  {r['filename']}: Reblur_MSE={r['Reblur_MSE']:.6f}")
    print('\n' + '=' * 60)
    print('Testing Complete!')
    print('=' * 60)
if __name__ == '__main__':
    main()