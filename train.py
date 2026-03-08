import argparse
import os
import sys
import math
from typing import Sized, cast, Any, Dict
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

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
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '').strip()
    if 'expandable_segments' not in alloc_conf:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'{alloc_conf},expandable_segments:True' if alloc_conf else 'expandable_segments:True'
    return (fallback, overrides)
(_THREAD_FALLBACK, _THREAD_ENV_OVERRIDES) = _normalize_thread_env()
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config
from utils.model_builder import build_models_from_config, build_trainer_from_config, build_dataloader_from_config
from utils.metrics import PerformanceEvaluator

def print_stage_info(stage: str, epoch: int, total_epochs: int):
    stage_names = {'physics_only': 'Stage 1: Physics Only (物理层训练)', 'restoration_fixed_physics': 'Stage 2: Restoration with Fixed Physics (复原网络训练)', 'joint': 'Stage 3: Joint Fine-tuning (联合微调, 学习率减半)', 'restoration_only': 'Restoration Only (无物理层)'}
    stage_name = stage_names.get(stage, stage)
    print(f"\n{'=' * 60}")
    print(f'Epoch {epoch}/{total_epochs} - {stage_name}')
    print(f"{'=' * 60}")

def main():
    parser = argparse.ArgumentParser(description='DPDD Training Script')
    parser.add_argument('--config', '-c', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--stage', type=str, default='all', choices=['1', '2', '3', 'all'], help='Specific stage to run (1, 2, 3) or "all" for full training')
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
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f'Loading config from {args.config}...')
    config = load_config(args.config)
    device = config.experiment.device
    if device == 'cuda' and (not torch.cuda.is_available()):
        print('Warning: CUDA not available, using CPU')
        device = 'cpu'
    config.experiment.device = device
    torch.manual_seed(config.experiment.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(config.experiment.seed)
    print(f'Device: {device}')
    print(f'Seed: {config.experiment.seed}')
    use_physical_layer = getattr(config.experiment, 'use_physical_layer', True)
    s1 = 0
    s2 = 0
    s3 = 0
    if use_physical_layer:
        s1 = config.training.stage_schedule.stage1_epochs
        s2 = config.training.stage_schedule.stage2_epochs
        s3 = config.training.stage_schedule.stage3_epochs
        schedule_total = s1 + s2 + s3
        if config.experiment.epochs != schedule_total:
            raise ValueError(f'config.experiment.epochs must match the sum of training.stage_schedule (epochs={config.experiment.epochs}, schedule_total={schedule_total}).')
        stage_boundaries = {'1': (0, s1), '2': (s1, s1 + s2), '3': (s1 + s2, schedule_total), 'all': (0, schedule_total)}
        (run_start_epoch, run_end_epoch) = stage_boundaries[args.stage]
        print(f'Target Run Stage: {args.stage} (Epochs {run_start_epoch} -> {run_end_epoch})')
    else:
        run_start_epoch = 0
        run_end_epoch = config.experiment.epochs
    base_output_dir = Path(config.experiment.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.experiment.run_name or config.experiment.name
    if config.experiment.use_timestamp:
        run_name = f'{run_name}_{datetime.now().strftime(config.experiment.timestamp_format)}'
    output_dir = str(base_output_dir / run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_subdir = config.experiment.checkpoints_subdir or 'checkpoints'
    checkpoints_dir = os.path.join(output_dir, checkpoints_subdir)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')
    print(f'Checkpoints will be saved to: {checkpoints_dir}')
    config.experiment.output_dir = output_dir
    tb_log_dir = None
    if getattr(config.experiment.tensorboard, 'enabled', False):
        base_tb_dir = config.experiment.tensorboard.log_dir
        if not os.path.isabs(base_tb_dir):
            base_tb_dir = os.path.join(output_dir, base_tb_dir)
        if config.experiment.tensorboard.append_run_name:
            tb_log_dir = os.path.join(base_tb_dir, run_name)
        else:
            tb_log_dir = base_tb_dir
        Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
        print(f'TensorBoard logs: {tb_log_dir}')
    print('\n' + '=' * 60)
    print('Initializing DataLoaders...')
    print('=' * 60)
    try:
        train_loader = build_dataloader_from_config(config, mode='train')
        val_loader = build_dataloader_from_config(config, mode='val')
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        train_size = len(cast(Sized, train_dataset)) if hasattr(train_dataset, '__len__') else 'Unknown'
        val_size = len(cast(Sized, val_dataset)) if hasattr(val_dataset, '__len__') else 'Unknown'
        real_train_size = cast(Any, train_dataset).get_real_length() if hasattr(train_dataset, 'get_real_length') else train_size
        print(f'✓ Train set: {real_train_size} real images, virtual length: {train_size}')
        print(f'✓ Val set size: {val_size}')
        print(f'✓ Batch size: {config.data.batch_size}')
        print(f'✓ Crop size (train): {config.data.crop_size}')
        print(f"✓ Center crop size (val): {getattr(config.data, 'val_crop_size', 1024)}")
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print('Please ensure the dataset exists at the specified path.')
        return
    print('\n' + '=' * 60)
    print('Building Models...')
    print('=' * 60)
    (zernike_gen, aberration_net, restoration_net, physical_layer) = build_models_from_config(config, device)
    print('\nInitializing Trainer...')
    trainer = build_trainer_from_config(config, restoration_net, physical_layer, device, tensorboard_dir=tb_log_dir)
    print(f'\n训练计划:')
    if use_physical_layer:
        print(f'  Stage 1 (Physics Only):        Epochs 1-{s1}')
        print(f'  Stage 2 (Restoration):         Epochs {s1 + 1}-{s1 + s2}')
        print(f'  Stage 3 (Joint, LR halved):    Epochs {s1 + s2 + 1}-{s1 + s2 + s3}')
        print(f'  Total: {s1 + s2 + s3} epochs')
    else:
        print(f'  Restoration Only:              Epochs 1-{config.experiment.epochs}')
    start_epoch = run_start_epoch
    if args.resume:
        print(f'\nResuming from checkpoint: {args.resume}')
        resume_info = trainer.load_checkpoint(args.resume)
        if resume_info.get('epoch') is not None:
            start_epoch = resume_info['epoch']
            print(f'  Resumed at epoch {start_epoch}')
    if start_epoch >= run_end_epoch:
        print(f'Warning: Start epoch ({start_epoch}) is >= End epoch ({run_end_epoch}) for stage {args.stage}.')
        print('Training skipped.')
        return
    if args.stage != 'all' and start_epoch < run_start_epoch:
        print(f'Warning: Start epoch ({start_epoch}) is before the requested stage start ({run_start_epoch}).')
        print(f'Training will start from {start_epoch}, covering previous stages until {run_end_epoch}.')
    total_epochs = run_end_epoch
    print('\n' + '=' * 60)
    print(f'Start Training (Target: Epochs {start_epoch} -> {run_end_epoch})')
    print('=' * 60)
    save_interval = config.experiment.save_interval
    prev_stage = None
    stage = 'physics_only' if use_physical_layer else 'restoration_only'
    val_metrics: Dict[str, Any] = {}
    evaluator = PerformanceEvaluator(device=device)
    for epoch in range(start_epoch, run_end_epoch):
        current_epoch = epoch + 1
        stage = trainer.get_stage(epoch)
        if stage != prev_stage:
            print_stage_info(stage, current_epoch, total_epochs)
            prev_stage = stage
        else:
            print(f'\nEpoch {current_epoch}/{total_epochs} [{stage}]')
        epoch_loss = 0.0
        epoch_loss_data = 0.0
        epoch_loss_sup = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f'Train E{current_epoch}')
        acc_steps = getattr(trainer, 'accumulation_steps', 1)
        for (batch_idx, batch) in enumerate(pbar):
            if isinstance(batch, dict):
                blur_imgs = batch['blur']
                sharp_imgs = batch['sharp']
                crop_info = batch.get('crop_info', None)
            else:
                (blur_imgs, sharp_imgs) = batch
                crop_info = None
            try:
                metrics = trainer.train_step(Y_blur=blur_imgs, X_gt=sharp_imgs, epoch=epoch, crop_info=crop_info)
            except torch.OutOfMemoryError:
                print(f'[OOM] Epoch {current_epoch}, batch {batch_idx + 1}: skipped this batch to continue training.')
                trainer.reset_after_oom()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            epoch_loss += metrics['loss']
            epoch_loss_data += metrics['loss_data']
            epoch_loss_sup += metrics['loss_sup']
            steps += 1
            if (batch_idx + 1) % acc_steps == 0:
                pbar.set_postfix({'Loss': f"{metrics['loss']:.4f}", 'Data': f"{metrics['loss_data']:.4f}", 'Sup': f"{metrics['loss_sup']:.4f}", 'GradW': f"{metrics.get('grad_W', 0):.2f}"})
        avg_loss = epoch_loss / max(steps, 1)
        avg_loss_data = epoch_loss_data / max(steps, 1)
        avg_loss_sup = epoch_loss_sup / max(steps, 1)
        print(f'  Train Loss: {avg_loss:.6f} (Data: {avg_loss_data:.6f}, Sup: {avg_loss_sup:.6f})')
        print('  Evaluating on validation set...')
        if stage == 'physics_only':
            if trainer.physical_layer is None:
                raise RuntimeError('physical_layer is required for physics_only evaluation')
            val_metrics = PerformanceEvaluator.evaluate_stage1(trainer.physical_layer, val_loader, device, config.training.smoothness_grid_size)
        else:
            val_metrics = evaluator.evaluate(trainer.restoration_net, trainer.physical_layer, val_loader, device, config.training.smoothness_grid_size)
        try:
            from tabulate import tabulate
            rows = []
            for (k, v) in val_metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    rows.append([k, 'NaN'])
                else:
                    rows.append([k, f'{v:.6f}'] if isinstance(v, float) else [k, str(v)])
            print(tabulate(rows, headers=['Metric', 'Value'], tablefmt='github'))
        except Exception:
            for (k, v) in val_metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    v_str = 'NaN'
                else:
                    v_str = f'{v:.6f}' if isinstance(v, float) else str(v)
                print(f'    {k}: {v_str}')
        train_metrics = {'loss': avg_loss, 'loss_data': avg_loss_data, 'loss_sup': avg_loss_sup}
        if 'metrics' in locals() and isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(k, str) and k.startswith('newbp/') and isinstance(v, (int, float)):
                    train_metrics[k] = float(v)
        trainer.log_to_tensorboard(train_metrics, current_epoch, prefix='train')
        trainer.log_to_tensorboard(val_metrics, current_epoch, prefix='val')
        lr_info = trainer.get_current_lr()
        trainer.log_to_tensorboard(lr_info, current_epoch, prefix='lr')
        is_best = trainer.update_best_metrics(val_metrics, stage)
        if stage == 'physics_only' and is_best.get('reblur_mse', False):
            best_path = os.path.join(checkpoints_dir, 'best_stage1_physics.pt')
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best Stage 1 model saved: Reblur_MSE={val_metrics.get('Reblur_MSE', 0):.6f}")
        elif stage == 'restoration_fixed_physics' and is_best.get('psnr', False):
            best_path = os.path.join(checkpoints_dir, 'best_stage2_restoration.pt')
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best Stage 2 model saved: PSNR={val_metrics.get('PSNR', 0):.2f}")
        elif stage == 'joint' and is_best.get('combined', False):
            best_path = os.path.join(checkpoints_dir, 'best_stage3_joint.pt')
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best Stage 3 model saved: PSNR={val_metrics.get('PSNR', 0):.2f}")
        elif stage == 'restoration_only' and is_best.get('psnr', False):
            best_path = os.path.join(checkpoints_dir, 'best_restoration_only.pt')
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best restoration-only model saved: PSNR={val_metrics.get('PSNR', 0):.2f}")
        if current_epoch % save_interval == 0:
            periodic_path = os.path.join(checkpoints_dir, f'checkpoint_epoch{current_epoch:03d}.pt')
            trainer.save_checkpoint(periodic_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f'  ✓ Periodic checkpoint saved: {periodic_path}')
    final_path = os.path.join(checkpoints_dir, 'final_model.pt')
    trainer.save_checkpoint(final_path, epoch=total_epochs, stage=stage, val_metrics=val_metrics)
    print(f'\n✓ Final model saved: {final_path}')
    trainer.close_tensorboard()
    print('\n' + '=' * 60)
    print('Training Finished!')
    print('=' * 60)
    print(f'\nBest metrics achieved:')
    if use_physical_layer:
        print(f"  Stage 1 (Physics): Reblur_MSE = {trainer.best_metrics['physics_only']['reblur_mse']:.6f}")
        print(f"  Stage 2 (Restoration): PSNR = {trainer.best_metrics['restoration_fixed_physics']['psnr']:.2f}")
        print(f"  Stage 3 (Joint): PSNR = {trainer.best_metrics['joint']['psnr']:.2f}")
    else:
        print(f"  Restoration Only: PSNR = {trainer.best_metrics['restoration_only']['psnr']:.2f}")
    print(f'\nOutput directory: {output_dir}')
    if tb_log_dir:
        print(f"Run 'tensorboard --logdir {tb_log_dir}' to view training curves.")
    else:
        print('TensorBoard is disabled.')
if __name__ == '__main__':
    main()