"""
Train or run inference for the system-level (ttbar_mass/pt/DR) classification
task -- combines train.py and inference.py into one script, in the same
style as run_slot_model.py / run_regression_ParT.py.

Usage:
    python -m hepsetreg.runs.run_classification --train --config configs/run_classification.yaml
    python -m hepsetreg.runs.run_classification --predict --config configs/run_classification.yaml
"""

import sys

from hepsetreg.config import check_args

# Checked here, before the heavy imports below, so a bad command line
# (missing --train/--predict or --config) fails in milliseconds rather than
# only being reported after paying the full torch/pytorch_lightning/... import cost.
if __name__ == "__main__":
    check_args("usage: python -m hepsetreg.runs.run_classification --train | --predict --config <path.yaml>")

import logging

import awkward as ak
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.backbones.head import EventClassifier, ClassificationHead
from hepsetreg.config import load_config
from hepsetreg.datasets.EventDatasetBase import ClassificationEventDataModule, ClassificationEventDataset, _identity_collate
from hepsetreg.training.lightning_module import EventClassificationModule

MODEL_KEYS = ["d_model", "nhead", "num_layers","num_bins"]


def build_model(config):
    """Builds the model architecture from a config dict (see MODEL_KEYS and
    config["features"]). train() and predict() both call this with the same
    model settings, so a saved checkpoint's weights always match the model
    they're loaded into."""
    features = config["features"]
    backbone = TransformerBackbone(
        n_global_features=len(features["global"]),
        n_jet_features=len(features["jet"]),
        n_lepton_features=len(features["lepton"]),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )
    head = ClassificationHead(d_model=config["d_model"], output_dim=config["num_bins"])
    return EventClassifier(backbone, head)


def train(config):
    # Not a hyperparameter -- an environment setting. PyTorch defaults to
    # using every visible CPU core for every op, even tiny ones; on a
    # many-core shared machine that causes severe thread-oversubscription
    # overhead (measured ~20x slower elsewhere in this project for small
    # per-op work). Capping it avoids that regardless of model size.
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("medium")  # allows TF32 matmuls on Ampere+ GPUs

    bin_edges = np.linspace(config["min_bin"],config["max_bin"],config["num_bins"]+1)

    dm = ClassificationEventDataModule(
        config["train_path"], config["val_path"], features=config["features"], bin_edges = bin_edges, batch_size=config["batch_size"]
    )

    model = build_model(config)
    if config["compile_model"]:
        model = torch.compile(model)
    # halve the learning rate when val_loss stops improving
    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config["lr_patience"],
    )

    module = EventClassificationModule(
        model, learning_rate=config["learning_rate"], scheduler_fn=scheduler_fn,
    )

    # stop once val_loss hasn't improved for early_stop_patience epochs, and
    # keep the best-val_loss checkpoint rather than the last one
    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=config["early_stop_patience"])
    best_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    # config["run_name"] names the run's folder (tb_logs/event_classification/<run_name>)
    # instead of the default auto-incrementing version_0, version_1, ...
    logger = TensorBoardLogger(save_dir="tb_logs", name="event_classification", version=config.get("run_name"))
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"], accelerator=config["accelerator"],
        logger=logger, precision="bf16-mixed", callbacks=[early_stop, best_checkpoint],
    )
    trainer.fit(module, datamodule=dm)

    # Save the z-score stats, max_jets, and model architecture alongside the
    # checkpoint: predict() needs all three to scale new data and rebuild
    # the model exactly the way this run did -- it can't guess or refit them.
    model_config = {key: config[key] for key in MODEL_KEYS}
    model_config["features"] = config["features"]
    torch.save(
        {"stats": dm.stats, "max_jets": dm.train_dataset.max_jets, "model_config": model_config},
        f"{logger.log_dir}/stats.pt",
    )
    print(f"done training. checkpoint + stats saved under {logger.log_dir}")


def predict(config):
    torch.set_num_threads(4)  # see the comment in train()

    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    saved = torch.load(config["stats"], map_location="cpu")
    stats, max_jets, model_config = saved["stats"], saved["max_jets"], saved["model_config"]

    model = build_model(model_config)
    module = EventClassificationModule(model)
    # if the checkpoint was saved from a torch.compile(model)-wrapped run,
    # its keys are prefixed "_orig_mod." -- strip it so they match this
    # plain, uncompiled model's keys.
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in checkpoint["state_dict"].items()}
    module.load_state_dict(state_dict)

    bin_edges = np.linspace(config["min_bin"],config["max_bin"],config["num_bins"]+1)

    target_features = model_config["features"]["target"]
    ds = EventDataset(config["data_path"], features=model_config["features"], bin_edges = bin_edges,stats=stats, max_jets=max_jets)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=_identity_collate)

    # Trainer.predict() puts the module in eval mode, disables gradients,
    # and moves both module and batches to the accelerator -- no manual
    # .eval()/.to(device)/no_grad()/per-batch transfer loop needed.
    trainer = pl.Trainer(accelerator=config["accelerator"], devices=1, logger=False)
    outputs = trainer.predict(module, dataloaders=loader)  # list of per-batch {"pred", "target"} dicts

    """
    preds_scaled = torch.cat([out["pred"] for out in outputs])
    targets_scaled = torch.cat([out["target"] for out in outputs])

    t_mean, t_std = stats["target"]
    t_mean, t_std = torch.as_tensor(t_mean), torch.as_tensor(t_std)
    preds = preds_scaled * (t_std + 1e-8) + t_mean
    targets = targets_scaled * (t_std + 1e-8) + t_mean
    """

    pred_logits = torch.cat([out["pred"] for out in outputs])
    targets = torch.cat([out["target"] for out in outputs])

    pred_probs = torch.softmax(pred_logits,dim=1)
    preds = torch.argmax(pred_probs,dim=1)

    mae = (preds - targets).abs().mean(dim=0)  # mean over events, per target feature
    for name, value in zip(target_features, mae):
        print(f"{name:12s} MAE: {value:.4f}")

    # save predictions + targets to parquet, as two record groups -- e.g.
    # arr.targets.ttbar_mass, arr.predictions.ttbar_mass
    # .contiguous(): a column slice of a 2D tensor is a strided,
    # non-contiguous view, which pyarrow rejects with "ndarray is not contiguous"
    target_fields = {name: ak.Array(targets[:, i].contiguous().numpy()) for i, name in enumerate(target_features)}
    pred_fields = {name: ak.Array(preds[:, i].contiguous().numpy()) for i, name in enumerate(target_features)}
    out = ak.zip({"targets": ak.zip(target_fields), "predictions": ak.zip(pred_fields)}, depth_limit=1)
    ak.to_parquet(out, config["output"])
    print(f"saved predictions to {config['output']}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        train(load_config("train"))
    elif "--predict" in sys.argv:
        predict(load_config("predict"))
    else:
        print("usage: python -m hepsetreg.runs.run_classification --train | --predict --config <path.yaml>")
        sys.exit(1)
