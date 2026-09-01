from dataclasses import dataclass


def next_power_of_2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


@dataclass
class Hyperparams:
    emb_dim: int = 512
    max_n_ions: int = 128
    pad_to_closest_pow2: bool = True
    zero_padded_ions: bool = False
    sort_ions_by_intensity: bool = True
    ion_input: str = "mz"
    mz_from: float = 10.0
    mz_to: float = 1000.0
    mz_min_wavelength: float = 0.0005
    mz_max_wavelength: float = 50000.0
    intensity_min_wavelength: float = 1e-5
    intensity_max_wavelength: float = 10.0

    tf_num_layers: int = 2
    tf_dim_ff: int = 512
    num_heads: int = 16
    tf_dropout: float = 0.2
    pos_min_wavelength: float = 0.1
    pos_max_wavelength: float = 10000.0
    masked_mean_pool: bool = True

    fp_hidden_channels: int = 4096
    fp_num_layers: int = 2
    fp_dropout: float = 0.1
    fp_size: int = 4096
    fp_radius: int = 2

    loss_type: str = "binary_cross_entropy"
    batch_size: int = 128
    max_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    val_fraction: float = 0.1
    seed: int = 1

    def padded_max_n_ions(self):
        return next_power_of_2(self.max_n_ions) if self.pad_to_closest_pow2 else self.max_n_ions
