from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, Optional


if TYPE_CHECKING:
    UncertaintyMode = Literal["disabled", "l2reg", "l1reg", "dino", "dino+mssim"]
else:
    # NOTE: OmegaConf does not support Literal types
    # https://github.com/omry/omegaconf/issues/422
    UncertaintyMode = str


@dataclass
class Config:
    source_path: str
    model_path: str
    sh_degree: int = 3
    images: str = "images"
    data_device: str = "cuda"
    eval: bool = False
    kernel_size: float = 0.1

    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

    num_sky_gaussians: int = 0
    use_background_model: bool = False
    background_lr: float = 0.001

    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002

    use_gof_abs_gradient: bool = True
    

    appearance_n_fourier_freqs: int = 4
    n_gaussian_features: int = 4
    embedding_lr: float = 0.005
    embedding_regularization: float = 0.0

    appearance_enabled: bool = True
    enable_exposure_mlp: bool = True
    exposure_mlp_lr: float = 0.0001
    appearance_embedding_dim: int = 32
    appearance_embedding_lr: float = 0.001
    appearance_mlp_lr: float = 0.0005
    appearance_embedding_regularization: float = 0.0
    appearance_embedding_optim_lr: float = 0.1
    appearance_embedding_optim_iters: int = 128
    appearance_optim_type: str = "dssim+l1-scaled"
    """Either 'mse', 'dssim+l1'"""
    appearance_separate_tuned_color: bool = True
    appearance_use_raw_colors: bool = False
    appearance_model_sh: bool = False
    appearance_model_2D: str = "disabled"
    appearance_conv_lr: float = 0.0005
    """either disabled, vast-gaussians"""

    appearance_init_fourier: bool = True

    # Uncertainty model
    uncertainty_mode: UncertaintyMode = "dino"
    uncertainty_backbone: str = "dinov2_vits14_reg"
    uncertainty_regularizer_weight: float = 0.5
    uncertainty_clip_min: float = 0.1
    uncertainty_mask_clip_max: Optional[float] = None
    uncertainty_dssim_clip_max: float = 1.0  # 0.05 -> 0.005
    uncertainty_lr: float = 0.001
    uncertainty_dropout: float = 0.5
    uncertainty_dino_max_size: Optional[int] = None
    uncertainty_scale_grad: bool = False
    uncertainty_center_mult: bool = False
    uncertainty_after_opacity_reset: int = 1000
    uncertainty_protected_iters: int = 500
    uncertainty_preserve_sky: bool = False

    uncertainty_warmup_iters: int = 0
    uncertainty_warmup_start: int = 2000
