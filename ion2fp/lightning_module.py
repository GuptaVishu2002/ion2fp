from dataclasses import fields

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .config import Hyperparams
from .losses import compute_loss
from .model import IonToFingerprintModel

_HP_FIELDS = {f.name for f in fields(Hyperparams)}


class FingerprintLightningModule(pl.LightningModule):
    def __init__(self, **hp_kwargs):
        super().__init__()
        hp_kwargs = {k: v for k, v in hp_kwargs.items() if k in _HP_FIELDS}
        self.save_hyperparameters(hp_kwargs)
        self.hp = Hyperparams(**hp_kwargs)
        self.model = IonToFingerprintModel(self.hp)

    def forward(self, mzs, intensities, padding_mask):
        return self.model(mzs, intensities, padding_mask)

    def _step(self, batch):
        pred = self(batch["mzs"], batch["intensities"], batch["mask"])
        loss = compute_loss(pred, batch["fingerprint"], self.hp.loss_type)
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch["mzs"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch["mzs"].shape[0])
        cos_sim = F.cosine_similarity(pred, batch["fingerprint"]).mean()
        self.log("val_cosine_sim", cos_sim, on_step=False, on_epoch=True, batch_size=batch["mzs"].shape[0])
        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch["mzs"], batch["intensities"], batch["mask"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
