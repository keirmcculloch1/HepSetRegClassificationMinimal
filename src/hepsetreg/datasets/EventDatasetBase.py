"""
EventDatasetBase: takes PreCachedDataset's "scale everything once, up front"
idea all the way -- jets are also PADDED to a fixed `max_jets` slots per
event at construction, using vectorized awkward calls, not a per-batch
Python loop.

Every item this Dataset returns already has a FIXED shape (unlike
EventDataset5's ragged-per-event `jet` tensor), so a DataLoader can stack a
batch with PyTorch's own default collation -- no custom `collate_fn`, no
per-batch padding cost during training at all.

Feature names are NOT defined here: the caller passes a `features` dict
(keys "global", "jet", "lepton", "target") built in its own run_*.py. A
field missing from the parquet file raises KeyError naming the field.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


def zscore(x, mean, std, eps=1e-8):
    """Z-transform: (x - mean) / std, feature-wise."""
    return (x - mean) / (std + eps)


def _require_fields(record, names, where):
    missing = [name for name in names if name not in record.fields]
    if missing:
        raise KeyError(f"{where}: field(s) {missing} not found; available fields: {record.fields}")


def load_arrays(parquet_path, features):
    """Reads (global, jet, lepton, target) from `parquet_path`.

    `features` is a dict with keys "global", "jet", "lepton", "target" giving
    the field names as they appear in the file. Lepton features are read as
    `el_<name>` and `mu_<name>` and concatenated, except "is_electron", which
    is derived from the flavour each lepton came from.
    """
    events = ak.from_parquet(parquet_path)
    inputs = events["inputs"]
    targets = events["targets"]

    lepton_names = [f for f in features["lepton"] if f != "is_electron"]
    _require_fields(inputs, features["global"] + features["jet"], f"{parquet_path} inputs")
    _require_fields(inputs, [f"el_{f}" for f in lepton_names] + [f"mu_{f}" for f in lepton_names], f"{parquet_path} inputs")
    if "is_electron" in features["lepton"]:
        _require_fields(inputs, ["el_pt", "mu_pt"], f"{parquet_path} inputs")
    _require_fields(targets, features["target"], f"{parquet_path} targets")

    global_arr = np.stack(
        [ak.to_numpy(inputs[f]).astype(np.float32) for f in features["global"]], axis=1
    )  # (n_events, n_global_features)

    jet_arr = ak.zip({f: inputs[f] for f in features["jet"]})

    lepton_fields = {}
    for f in features["lepton"]:
        if f == "is_electron":
            lepton_fields[f] = ak.concatenate([ak.ones_like(inputs["el_pt"]), ak.zeros_like(inputs["mu_pt"])], axis=1)
        else:
            lepton_fields[f] = ak.concatenate([inputs[f"el_{f}"], inputs[f"mu_{f}"]], axis=1)
    lepton_arr = ak.zip(lepton_fields)

    if features["target"]:
        target_arr = np.stack(
            [ak.to_numpy(targets[f]).astype(np.float32) for f in features["target"]], axis=1
        )
    else:
        target_arr = np.zeros((len(events), 0), dtype=np.float32)
    return global_arr, jet_arr, lepton_arr, target_arr


class ClassificationEventDataset(Dataset):
    """
    Parameters
    ----------
    source : str or (global, jet, lepton, target) tuple
        Parquet path, or the arrays already split out (see `load_arrays`).
    features : dict
        Keys "global", "jet", "lepton", "target" -- field names to use.
    indices, stats, keep_unscaled_data : see EventDataset5.EventDataset.
    max_jets : int or None
        Fixed jet-slot count every event is padded/truncated to. None (the
        default) auto-detects the largest jet count actually present in
        `source` -- no wasted padding, no risk of silently truncating a
        real jet.
    """

    def __init__(self, source, *, features, indices=None, stats=None, keep_unscaled_data=False, max_jets=None):
        global_arr, jet_arr, lepton_arr, target_arr = (
            load_arrays(source, features) if isinstance(source, str) else source
        )

        if indices is None:
            indices = np.arange(len(global_arr))
        indices = np.asarray(indices)
        n_events = len(indices)

        global_arr = global_arr[indices]
        target_arr = target_arr[indices]
        jet_sub = jet_arr[indices]
        lepton_sub = lepton_arr[indices]
        self.keep_unscaled_data = keep_unscaled_data

        # fit/reuse stats on the RAGGED jets (real objects only) -- must
        # happen before padding, or the padded zeros would skew the mean/std
        self.stats = stats or self._compute_stats(global_arr, jet_sub, lepton_sub, target_arr, features)
        g_mean, g_std = self.stats["global"]
        j_mean, j_std = self.stats["jet"]
        l_mean, l_std = self.stats["lepton"]
        t_mean, t_std = self.stats["target"]

        # ---- global / target: already fixed-size, zscore the whole array at once ----
        self.global_tensor = torch.from_numpy(zscore(global_arr, g_mean, g_std))
        self.target_tensor = (torch.from_numpy(target_arr)).to(torch.int64)
        if keep_unscaled_data:
            self.global_unscaled_tensor = torch.from_numpy(global_arr)
            self.target_unscaled_tensor = (torch.from_numpy(target_arr)).to(torch.int64)

        # ---- leptons: always exactly 2/event -> flatten, zscore, reshape ----
        lep_flat = np.stack(
            [ak.to_numpy(ak.flatten(lepton_sub[f])).astype(np.float32) for f in features["lepton"]], axis=1
        )
        self.lepton_tensor = torch.from_numpy(zscore(lep_flat, l_mean, l_std)).reshape(
            n_events, 2, len(features["lepton"])
        )
        # always exactly 2 real leptons/event -> mask is trivially all-True,
        # but TransformerBackbone.forward reads batch["lepton_mask"]
        # unconditionally, so it still needs to be present.
        self.lepton_mask = torch.ones(n_events, 2, dtype=torch.bool)
        if keep_unscaled_data:
            self.lepton_unscaled_tensor = torch.from_numpy(lep_flat).reshape(n_events, 2, len(features["lepton"]))

        # ---- jets: ragged -> pad/truncate to `max_jets` fixed slots, ONCE ----
        counts = ak.num(jet_sub, axis=1).to_numpy()
        self.max_jets = max_jets if max_jets is not None else int(counts.max())

        # One vectorized pad+fill+convert call per feature (not per event) --
        # ak.pad_none pads every event's jet list up to max_jets with None
        # (truncating longer ones, since clip=True); ak.fill_none turns those
        # into 0.0.
        jet_dense = np.stack(
            [
                ak.to_numpy(ak.fill_none(ak.pad_none(jet_sub[f], self.max_jets, clip=True), 0.0)).astype(np.float32)
                for f in features["jet"]
            ],
            axis=-1,
        )  # (n_events, max_jets, n_jet_features)

        mask = np.arange(self.max_jets)[None, :] < counts[:, None]  # True = real jet
        self.jet_mask = torch.from_numpy(mask)

        if keep_unscaled_data:
            self.jet_unscaled_tensor = torch.from_numpy(jet_dense.copy())

        # In-place zscore: jets are the biggest tensor here, so avoid
        # allocating a second full-size (n_events, max_jets, n_features)
        # array just to hold the scaled copy.
        jet_dense -= j_mean
        jet_dense /= j_std + 1e-8
        jet_dense[~mask] = 0.0  # padded slots stay exactly 0, not zscore(0)
        self.jet_tensor = torch.from_numpy(jet_dense)

    @staticmethod
    def _compute_stats(global_arr, jet_arr, lepton_arr, target_arr, features):
        jet_flat = np.stack(
            [ak.to_numpy(ak.flatten(jet_arr[f])).astype(np.float32) for f in features["jet"]], axis=1
        )
        lep_flat = np.stack(
            [ak.to_numpy(ak.flatten(lepton_arr[f])).astype(np.float32) for f in features["lepton"]], axis=1
        )
        return {
            "global": (global_arr.mean(0), global_arr.std(0)),
            "jet": (jet_flat.mean(0), jet_flat.std(0)),
            "lepton": (lep_flat.mean(0), lep_flat.std(0)),
            "target": (target_arr.mean(0), target_arr.std(0)),
        }

    def to(self, device):
        """Move every cached tensor onto `device` (e.g. `torch.device('cuda')`)
        in place, so `__getitem__`/`__getitems__` slice directly out of VRAM
        instead of paying a host-to-device copy on every batch. Returns
        `self` for chaining. All the CPU-side work (padding, z-scoring) has
        already happened by construction time -- this is a one-time bulk
        transfer of the finished tensors, not a repeated per-batch cost.
        """
        self.global_tensor = self.global_tensor.to(device)
        self.target_tensor = self.target_tensor.to(device)
        self.lepton_tensor = self.lepton_tensor.to(device)
        self.lepton_mask = self.lepton_mask.to(device)
        self.jet_tensor = self.jet_tensor.to(device)
        self.jet_mask = self.jet_mask.to(device)
        if self.keep_unscaled_data:
            self.global_unscaled_tensor = self.global_unscaled_tensor.to(device)
            self.target_unscaled_tensor = self.target_unscaled_tensor.to(device)
            self.lepton_unscaled_tensor = self.lepton_unscaled_tensor.to(device)
            self.jet_unscaled_tensor = self.jet_unscaled_tensor.to(device)
        return self

    def __len__(self):
        return self.global_tensor.shape[0]

    def __getitem__(self, idx):
        item = {
            "global": self.global_tensor[idx],
            "jet": self.jet_tensor[idx],  # (max_jets, n_jet_features), fixed shape
            "jet_mask": self.jet_mask[idx],  # (max_jets,)
            "lepton": self.lepton_tensor[idx],
            "lepton_mask": self.lepton_mask[idx],  # (2,), always True
            "target": self.target_tensor[idx],
        }
        if self.keep_unscaled_data:
            item["global_unscaled"] = self.global_unscaled_tensor[idx]
            item["jet_unscaled"] = self.jet_unscaled_tensor[idx]
            item["lepton_unscaled"] = self.lepton_unscaled_tensor[idx]
            item["target_unscaled"] = self.target_unscaled_tensor[idx]
        return item

    def __getitems__(self, indices):
        """Batched fetch: when a `Dataset` defines this, `DataLoader` calls
        it ONCE per batch with the whole index list, instead of calling
        `__getitem__` once per index (see
        `torch.utils.data._utils.fetch._MapDatasetFetcher`). Since
        everything here is already a dense tensor, a batch is just one
        vectorized fancy-index per field -- no Python-level loop over the
        batch at all, unlike the default `[dataset[i] for i in indices]` +
        `default_collate` path (measured ~13x faster at batch_size=4096).

        Returns the batch already collated as a dict of stacked tensors --
        `EventDataModule` pairs this with an identity `collate_fn` so
        nothing tries to re-collate it.
        """
        # index tensor must live on the same device as the data it indexes
        # (matters once .to(device) has moved everything onto a GPU)
        idx = torch.as_tensor(indices, device=self.global_tensor.device)
        batch = {
            "global": self.global_tensor[idx],
            "jet": self.jet_tensor[idx],
            "jet_mask": self.jet_mask[idx],
            "lepton": self.lepton_tensor[idx],
            "lepton_mask": self.lepton_mask[idx],
            "target": self.target_tensor[idx],
        }
        if self.keep_unscaled_data:
            batch["global_unscaled"] = self.global_unscaled_tensor[idx]
            batch["jet_unscaled"] = self.jet_unscaled_tensor[idx]
            batch["lepton_unscaled"] = self.lepton_unscaled_tensor[idx]
            batch["target_unscaled"] = self.target_unscaled_tensor[idx]
        return batch


def _identity_collate(batch):
    """No-op collate: `EventDataset.__getitems__` already returns a fully
    batched dict. A plain module-level function (not a lambda) so it can be
    pickled to DataLoader workers when num_workers>0."""
    return batch


class ClassificationEventDataModule(pl.LightningDataModule):
    """Separate train/val parquet files. Fits the Z-score stats (inputs AND
    targets) on the train file only, and reuses those exact stats for val
    (no leakage from val into the scaling). `max_jets` is likewise fit on
    train and reused for val, so both splits pad to the same jet-slot count.

    No `collate_fn` is passed to either DataLoader: every item this
    Dataset's `__getitem__` returns is already a fixed shape, so PyTorch's
    own default collation handles batching.

    `num_workers` defaults to 0 -- with EventDatasetBase, `__getitem__` is
    pure tensor indexing (even cheaper than EventDataset5's
    PreCachedDataset), so the case for multiprocess workers is weaker than
    ever; a prior attempt at num_workers>0 + per-worker thread-capping on
    EventDataset5's DataLoader made real training slower, not faster (the
    IPC cost of shipping batches back from worker processes outweighed the
    now-tiny per-item work). Left configurable, but not defaulted on.

    `persistent_workers` only has an effect when `num_workers>0` (PyTorch
    raises otherwise if left True at num_workers=0, so it's silently
    ignored here instead). If you do use num_workers>0, this matters more
    than usual: without it, DataLoader tears down and re-forks worker
    processes every epoch, and forking after CUDA is already initialized in
    the main process (which Lightning does before the first dataloader
    iteration) can hang -- this is what caused an earlier "epochs never
    start" issue. persistent_workers=True forks workers once, up front,
    avoiding repeated forks into that danger zone.

    `device` : None (default) keeps the dataset on CPU. Set it to a CUDA
    device to move every cached tensor onto the GPU right after
    construction (see `EventDataset.to`), so batches are sliced directly
    out of VRAM with no per-batch host-to-device copy at all. CUDA tensors
    can't be handed across DataLoader worker process boundaries, so this
    forces num_workers=0/persistent_workers=False regardless of what's
    passed in -- there's also no point pinning memory that's already on the
    GPU, so pin_memory is disabled too in that case.
    """

    def __init__(
        self, train_path, val_path, *, features, batch_size=256, num_workers=0,
        persistent_workers=False, keep_unscaled_data=False, max_jets=None,
        device=None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.features = features
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else None
        if self.device is not None and self.device.type == "cuda":
            num_workers = 0
            persistent_workers = False
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.keep_unscaled_data = keep_unscaled_data
        self.max_jets = max_jets

    def setup(self, stage=None):
        self.train_dataset = ClassificationEventDataset(
            self.train_path, features=self.features,
            keep_unscaled_data=self.keep_unscaled_data, max_jets=self.max_jets,
        )  # fits stats (and, if not given, max_jets) on train
        self.stats = self.train_dataset.stats
        self.val_dataset = ClassificationEventDataset(
            self.val_path, features=self.features, stats=self.stats,
            keep_unscaled_data=self.keep_unscaled_data, max_jets=self.train_dataset.max_jets,
        )  # reuse them
        if self.device is not None:
            self.train_dataset.to(self.device)
            self.val_dataset.to(self.device)
        print("Setup done")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=self.device is None, collate_fn=_identity_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=self.device is None, collate_fn=_identity_collate,
        )
