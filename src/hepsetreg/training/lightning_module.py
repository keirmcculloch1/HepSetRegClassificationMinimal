"""
LightningModule wiring a backbone+head model into a training loop: a plain
pointwise regression loss (MSE by default, via `losses.RegressionLoss`)
against EventDataset5's (already z-scored) "target".

Model-agnostic: works with `EventRegressor` (backbone.py + head.py) as-is,
or any `nn.Module` whose `forward(batch) -> (B, output_dim)` matches that
shape.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from hepsetreg.losses.losses import RegressionLoss

import torch.nn as nn

class EventClassificationModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.0, scheduler_fn=None):
        super().__init__()
        self.model = model

        #Loss function is cross entropy loss only
        self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_fn = scheduler_fn  # optimizer -> LR scheduler, or None for a constant LR
        self.save_hyperparameters(ignore=["model", "scheduler_fn"])

    def forward(self, batch):
        return self.model(batch)

    def _shared_step(self, batch, stage):
        pred = self.model(batch)

        #Slice the targets to avoid shape issues
        #This assumes that there is only one target feature
        loss = self.loss_fn(pred, batch["target"][:,0])
        
        self.log(
            f"{stage}_loss", loss,
            on_step=(stage == "train"), on_epoch=True, prog_bar=True,
            batch_size=batch["target"].shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        """Returns the (still z-scored) prediction and target for one batch,
        moved to CPU. The run_*.py predict() functions concatenate these
        across all batches from Trainer.predict()'s returned list, then
        unscale once at the end -- Trainer.predict() already handles model
        eval-mode, no_grad, and per-batch device transfer, so there's no
        manual loop or .to(device) needed on the caller's side."""
        pred = self.model(batch)
        return {"pred": pred.cpu(), "target": batch["target"].cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      fused=True)  # Requires PyTorch 2.0+ and CUDA)
        if self.scheduler_fn is None:
            return optimizer
        scheduler = self.scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
