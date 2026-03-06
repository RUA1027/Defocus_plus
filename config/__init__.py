import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path

@dataclass
class PhysicsConfig:
    n_modes: int = 36
    pupil_size: int = 64
    kernel_size: int = 33
    oversample_factor: int = 2
    ref_wavelength: float = 5.5e-07
    wavelengths: List[float] = field(default_factory=lambda : [6.2e-07, 5.5e-07, 4.5e-07])
    learnable_wavelengths: bool = False
    wavelength_bounds: List[float] = field(default_factory=lambda : [4e-07, 7e-07])

    def __post_init__(self):
        if self.kernel_size % 2 == 0:
            raise ValueError(f'kernel_size 必须为奇数，当前值: {self.kernel_size}')
        if self.n_modes < 1 or self.n_modes > 36:
            raise ValueError(f'n_modes 必须在 1-36 之间，当前值: {self.n_modes}')
        if not isinstance(self.wavelength_bounds, list) or len(self.wavelength_bounds) != 2 or self.wavelength_bounds[0] >= self.wavelength_bounds[1]:
            raise ValueError(f'wavelength_bounds 必须为 [min, max]，当前值: {self.wavelength_bounds}')
        if any((w < self.wavelength_bounds[0] or w > self.wavelength_bounds[1] for w in self.wavelengths)):
            raise ValueError('wavelengths 必须在 wavelength_bounds 范围内')

@dataclass
class OLAConfig:
    patch_size: int = 128
    stride: int = 64
    pad_to_power_2: bool = True

    def __post_init__(self):
        if self.stride > self.patch_size:
            raise ValueError(f'stride ({self.stride}) 不能大于 patch_size ({self.patch_size})')

@dataclass
class MLPConfig:
    hidden_dim: int = 64
    use_fourier: bool = True
    fourier_scale: int = 5
    a_max_mlp: float = 3.0

@dataclass
class AberrationNetConfig:
    n_coeffs: int = 36
    a_max: float = 2.0
    mlp: MLPConfig = field(default_factory=MLPConfig)

@dataclass
class RestorationNetConfig:
    n_channels: int = 3
    n_classes: int = 3
    base_filters: int = 64
    bilinear: bool = True
    use_coords: bool = True
    use_physics_injection: bool = True
    injection_grid_size: int = 16
    channel_multipliers: List[int] = field(default_factory=lambda : [1, 2, 4, 8, 8])

@dataclass
class OptimizerConfig:
    type: str = 'adamw'
    lr_restoration: float = 0.0001
    lr_optics: float = 1e-05
    weight_decay: float = 0.01

@dataclass
class LossConfig:
    lambda_sup: float = 0.0
    lambda_fft: float = 0.1
    lambda_coeff: float = 0.01
    lambda_smooth: float = 0.01
    lambda_image_reg: float = 0.001

@dataclass
class GradientClipConfig:
    restoration: float = 5.0
    optics: float = 1.0

@dataclass
class StageScheduleConfig:
    stage1_epochs: int = 50
    stage2_epochs: int = 200
    stage3_epochs: int = 50

@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gradient_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    stage_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    smoothness_grid_size: int = 16
    accumulation_steps: int = 1
    stage_schedule: StageScheduleConfig = field(default_factory=StageScheduleConfig)

@dataclass
class AugmentationConfig:
    random_flip: bool = True
    random_rotate90: bool = False

@dataclass
class DataConfig:
    data_root: str = 'data/dd_dp_dataset_png'
    batch_size: int = 4
    image_height: int = 1120
    image_width: int = 1680
    crop_size: int = 512
    val_crop_size: int = 1024
    num_workers: int = 4
    repeat_factor: int = 100
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

@dataclass
class PSFGridConfig:
    rows: int = 5
    cols: int = 5
    coord_range: List[float] = field(default_factory=lambda : [-0.9, 0.9])
    colormap: str = 'inferno'

@dataclass
class CoeffMapsConfig:
    grid_size: int = 128
    indices: List[int] = field(default_factory=lambda : [3, 4, 5, 6])
    colormap: str = 'viridis'

@dataclass
class VisualizationConfig:
    psf_grid: PSFGridConfig = field(default_factory=PSFGridConfig)
    coeff_maps: CoeffMapsConfig = field(default_factory=CoeffMapsConfig)

@dataclass
class ExperimentConfig:
    name: str = 'default'
    seed: int = 42
    device: str = 'cuda'
    use_physical_layer: bool = True
    epochs: int = 300
    save_interval: int = 20
    log_interval: int = 1
    output_dir: str = 'results'
    run_name: Optional[str] = None
    use_timestamp: bool = True
    timestamp_format: str = '%m%d_%H%M'
    checkpoints_subdir: str = 'checkpoints'
    tensorboard: 'TensorBoardConfig' = field(default_factory=lambda : TensorBoardConfig())

@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = 'runs'
    append_run_name: bool = False
    log_images: bool = True
    image_log_interval: int = 10

@dataclass
class CheckpointConfig:
    save_best_per_stage: bool = True
    stage1_metric: str = 'reblur_mse'
    stage2_metric: str = 'psnr'
    stage3_metric: str = 'combined'
    save_interval: int = 10
    log_interval: int = 1
    output_dir: str = 'results'

@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ola: OLAConfig = field(default_factory=OLAConfig)
    aberration_net: AberrationNetConfig = field(default_factory=AberrationNetConfig)
    restoration_net: RestorationNetConfig = field(default_factory=RestorationNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        print(f'配置已保存到: {path}')

def _dataclass_to_dict(obj) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for (k, v) in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj

def _dict_to_dataclass(cls, data: Dict[str, Any]):
    if data is None:
        return cls()
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for (field_name, field_type) in field_types.items():
        if field_name in data:
            value = data[field_name]
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[field_name] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value
    return cls(**kwargs)

def _apply_overrides(config_dict: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for override in overrides:
        if '=' not in override:
            raise ValueError(f'无效的覆盖格式: {override}，应为 key=value')
        (key_path, value_str) = override.split('=', 1)
        keys = key_path.split('.')
        try:
            if '.' in value_str:
                value = float(value_str)
            elif value_str.lower() in ('true', 'false'):
                value = value_str.lower() == 'true'
            elif value_str.startswith('[') and value_str.endswith(']'):
                value = yaml.safe_load(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    return config_dict

def load_config(config_path: Optional[str]=None, overrides: Optional[List[str]]=None) -> Config:
    if config_path is None:
        default_path = Path(__file__).parent / 'default.yaml'
        if default_path.exists():
            config_path = str(default_path)
        else:
            print('未找到默认配置文件，使用内置默认值')
            return Config()
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    if overrides:
        config_dict = _apply_overrides(config_dict, overrides)
    config = _build_config_from_dict(config_dict)
    print(f'✓ 配置已加载: {config_path}')
    return config

def _build_config_from_dict(data: Dict[str, Any]) -> Config:
    physics = _dict_to_dataclass(PhysicsConfig, data.get('physics', {}))
    ola = _dict_to_dataclass(OLAConfig, data.get('ola', {}))
    ab_data = data.get('aberration_net', {})
    mlp = _dict_to_dataclass(MLPConfig, ab_data.get('mlp', {}))
    aberration_net = AberrationNetConfig(n_coeffs=ab_data.get('n_coeffs', 36), a_max=ab_data.get('a_max', 2.0), mlp=mlp)
    restoration_net = _dict_to_dataclass(RestorationNetConfig, data.get('restoration_net', {}))
    tr_data = data.get('training', {})
    optimizer = _dict_to_dataclass(OptimizerConfig, tr_data.get('optimizer', {}))
    loss = _dict_to_dataclass(LossConfig, tr_data.get('loss', {}))
    gradient_clip = _dict_to_dataclass(GradientClipConfig, tr_data.get('gradient_clip', {}))
    stage_schedule = _dict_to_dataclass(StageScheduleConfig, tr_data.get('stage_schedule', {}))
    training = TrainingConfig(optimizer=optimizer, loss=loss, gradient_clip=gradient_clip, stage_weights=tr_data.get('stage_weights', {}), smoothness_grid_size=tr_data.get('smoothness_grid_size', 16), accumulation_steps=tr_data.get('accumulation_steps', 1), stage_schedule=stage_schedule)
    d_data = data.get('data', {})
    augmentation = _dict_to_dataclass(AugmentationConfig, d_data.get('augmentation', {}))
    data_config = DataConfig(data_root=d_data.get('data_root', 'data/dd_dp_dataset_png'), batch_size=d_data.get('batch_size', 2), image_height=d_data.get('image_height', 1120), image_width=d_data.get('image_width', 1680), crop_size=d_data.get('crop_size', 512), val_crop_size=d_data.get('val_crop_size', 1024), num_workers=d_data.get('num_workers', 4), repeat_factor=d_data.get('repeat_factor', 100), augmentation=augmentation)
    vis_data = data.get('visualization', {})
    psf_grid = _dict_to_dataclass(PSFGridConfig, vis_data.get('psf_grid', {}))
    coeff_maps = _dict_to_dataclass(CoeffMapsConfig, vis_data.get('coeff_maps', {}))
    visualization = VisualizationConfig(psf_grid=psf_grid, coeff_maps=coeff_maps)
    exp_data = data.get('experiment', {})
    tensorboard = _dict_to_dataclass(TensorBoardConfig, exp_data.get('tensorboard', {}))
    experiment = ExperimentConfig(name=exp_data.get('name', 'default'), seed=exp_data.get('seed', 42), device=exp_data.get('device', 'cuda'), use_physical_layer=exp_data.get('use_physical_layer', True), epochs=exp_data.get('epochs', 300), save_interval=exp_data.get('save_interval', 20), log_interval=exp_data.get('log_interval', 1), output_dir=exp_data.get('output_dir', 'results'), run_name=exp_data.get('run_name'), use_timestamp=exp_data.get('use_timestamp', True), timestamp_format=exp_data.get('timestamp_format', '%m%d_%H%M'), checkpoints_subdir=exp_data.get('checkpoints_subdir', 'checkpoints'), tensorboard=tensorboard)
    ckpt_data = data.get('checkpoint', {})
    checkpoint = CheckpointConfig(save_best_per_stage=ckpt_data.get('save_best_per_stage', True), stage1_metric=ckpt_data.get('stage1_metric', 'reblur_mse'), stage2_metric=ckpt_data.get('stage2_metric', 'psnr'), stage3_metric=ckpt_data.get('stage3_metric', 'combined'), save_interval=ckpt_data.get('save_interval', 10), log_interval=ckpt_data.get('log_interval', 1), output_dir=ckpt_data.get('output_dir', 'results'))
    return Config(physics=physics, ola=ola, aberration_net=aberration_net, restoration_net=restoration_net, training=training, data=data_config, visualization=visualization, experiment=experiment, checkpoint=checkpoint)

def get_default_config() -> Config:
    return Config()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='配置管理工具')
    parser.add_argument('--config', '-c', type=str, default=None, help='配置文件路径')
    parser.add_argument('--print', '-p', action='store_true', help='打印配置')
    parser.add_argument('--save', '-s', type=str, default=None, help='保存配置到指定路径')
    parser.add_argument('overrides', nargs='*', help='覆盖参数，格式: key1.key2=value')
    args = parser.parse_args()
    config = load_config(args.config, args.overrides if args.overrides else None)
    if args.print:
        print('\n' + '=' * 60)
        print('当前配置:')
        print('=' * 60)
        print(config)
    if args.save:
        config.save(args.save)