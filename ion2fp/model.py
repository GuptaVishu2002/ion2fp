import torch
import torch.nn as nn

from .encoders import FloatEncoder


class TransformerEncoder(torch.nn.Module):
    def __init__(
            self,
            d_model: int = 128,
            num_heads: int = 8,
            dim_feedforward: int = 1024,
            n_layers: int = 1,
            dropout: float = 0,
            **kwargs: dict,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.dropout = dropout

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
            self,
            embeddings,
            padding_mask,
            **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)


class IonsToSpectralEncoder(nn.Module):
    def __init__(
            self,
            max_num_ions_per_spectrum: int = 100,
            num_heads: int = 1,
            emb_dim: int = 512,
            tf_num_layers: int = 2,
            tf_dim_ff: int = 1024,
            tf_dropout: int = 0.2,
            masked_mean_pool: bool = False,
            zero_padded_ions: bool = False,
    ):
        super().__init__()
        self.TransformEncoder = TransformerEncoder(
            d_model=emb_dim,
            num_heads=num_heads,
            dim_feedforward=tf_dim_ff,
            n_layers=tf_num_layers,
            dropout=tf_dropout
        )
        self.positional_encoder = FloatEncoder(
            d_model=emb_dim,
            min_wavelength=0.1,
            max_wavelength=10000,
        )
        self.max_num_ions_per_spectrum = max_num_ions_per_spectrum
        self.masked_mean_pool = masked_mean_pool
        self.zero_padded_ions = zero_padded_ions

    def forward(self, x, padding_mask):
        if self.zero_padded_ions:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)
        positions = torch.tensor([[[i] for i in range(self.max_num_ions_per_spectrum)]] * x.shape[0], dtype=torch.float32, device=x.device)
        positional_encoding = self.positional_encoder(positions).squeeze(2)
        x = x + positional_encoding
        transformer_output = self.TransformEncoder(x, ~padding_mask)
        if self.masked_mean_pool:
            mask = padding_mask.unsqueeze(-1).to(transformer_output.dtype)
            x_conv = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x_conv = transformer_output.mean(dim=1)
        return x_conv


class SineIonEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            ion_input: str = "mz",
            min_mz_wavelength: float = 0.0005,
            max_mz_wavelength: float = 50000.0,
            min_intensity_wavelength: float = 1e-5,
            max_intensity_wavelength: float = 10.0,
    ):
        super().__init__()
        if ion_input not in ("mz", "intensity", "both"):
            raise ValueError(f"ion_input must be 'mz', 'intensity', or 'both', got {ion_input!r}")
        self.ion_input = ion_input
        self.mz_encoder = None
        self.int_encoder = None
        if ion_input == "both":
            assert d_model % 2 == 0, "d_model must be even (split evenly between m/z and intensity, as in PeakEncoder)"
            half_dim = d_model // 2
            self.mz_encoder = FloatEncoder(
                d_model=half_dim,
                min_wavelength=min_mz_wavelength,
                max_wavelength=max_mz_wavelength,
            )
            self.int_encoder = FloatEncoder(
                d_model=half_dim,
                min_wavelength=min_intensity_wavelength,
                max_wavelength=max_intensity_wavelength,
            )
        elif ion_input == "mz":
            self.mz_encoder = FloatEncoder(
                d_model=d_model,
                min_wavelength=min_mz_wavelength,
                max_wavelength=max_mz_wavelength,
            )
        else:
            self.int_encoder = FloatEncoder(
                d_model=d_model,
                min_wavelength=min_intensity_wavelength,
                max_wavelength=max_intensity_wavelength,
            )

    def forward(self, mzs, intensities):
        if self.ion_input == "both":
            return torch.cat([self.mz_encoder(mzs), self.int_encoder(intensities)], dim=2)
        if self.ion_input == "mz":
            return self.mz_encoder(mzs)
        return self.int_encoder(intensities)


class FeedForwardHead(nn.Module):
    def __init__(self, hp):
        super().__init__()
        n_hidden = max(hp.fp_num_layers - 1, 0)
        dims = [hp.emb_dim] + [hp.fp_hidden_channels] * n_hidden + [hp.fp_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(hp.fp_dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class IonToFingerprintModel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.ion_embedding = SineIonEmbedding(
            d_model=hp.emb_dim,
            ion_input=hp.ion_input,
            min_mz_wavelength=hp.mz_min_wavelength,
            max_mz_wavelength=hp.mz_max_wavelength,
            min_intensity_wavelength=hp.intensity_min_wavelength,
            max_intensity_wavelength=hp.intensity_max_wavelength,
        )
        self.ions_to_spectral = IonsToSpectralEncoder(
            max_num_ions_per_spectrum=hp.padded_max_n_ions(),
            num_heads=hp.num_heads,
            emb_dim=hp.emb_dim,
            tf_num_layers=hp.tf_num_layers,
            tf_dim_ff=hp.tf_dim_ff,
            tf_dropout=hp.tf_dropout,
            masked_mean_pool=hp.masked_mean_pool,
            zero_padded_ions=hp.zero_padded_ions,
        )
        self.ffn = FeedForwardHead(hp)

    def forward(self, mzs, intensities, padding_mask):
        ion_embeddings = self.ion_embedding(mzs, intensities)
        spectral_embedding = self.ions_to_spectral(ion_embeddings, padding_mask)
        logits = self.ffn(spectral_embedding)
        return torch.sigmoid(logits)
