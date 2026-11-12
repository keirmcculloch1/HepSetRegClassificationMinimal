"""
Regression head: maps the TransformerBackbone's pooled event embedding down
to the target vector (see TARGET_FEATURES in EventDataset5.py).
"""

from __future__ import annotations

import torch

import torch.nn as nn

import numpy as np

import awkward as ak

import itertools

class RegressionHead(nn.Module):
    """Simple MLP mapping a pooled embedding to the regression targets.

    Hidden width halves each layer (floored at `min_neurons`).
    """

    def __init__(
        self,
        d_model,
        output_dim,
        n_layers=3,
        start_neurons=128,
        dropout=0.05,
        min_neurons=8,
    ):
        super().__init__()
        layers = []
        prev_dim = d_model
        for i in range(n_layers):
            hidden_dim = max(start_neurons // (2**i), min_neurons)
            layers += [nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def to_nested_list(x: torch.Tensor, sizes: list[int]):
    """
    x:     (N, sum(sizes)) tensor, features concatenated along dim 1
    sizes: [i, j, k, ...] classes per feature (length M)
    returns: list of N lists, each with M tensors of shape (sizes[m],)
    """
    N, M = x.shape[0], len(sizes)
    assert x.shape[1] == sum(sizes)

    flat = x.contiguous().view(-1)           # row-major: row0 feats, row1 feats, ...
    pieces = flat.split(sizes * N)           # single call -> N*M views
    return [pieces[n * M:(n + 1) * M] for n in range(N)]  # tuple slices are cheap

class ClassificationHead(nn.Module):
    """Simple MLP mapping a pooled embedding to the classification targets.

    Hidden width halves each layer (floored at `min_neurons`).
    """

    def __init__(
        self,
        d_model,
        num_features,
        classes_per_feature,
        n_layers=3,
        start_neurons=128,
        dropout=0.05,
        min_neurons=8,
    ):
        super().__init__()
        layers = []
        prev_dim = d_model

        #Save classes_per_feature to use in forward()
        self.classes_per_feature = classes_per_feature

        #For N features each with (i,j,k...) classes the output dimension is i+j+k...
        output_dim = np.sum(classes_per_feature)

        for i in range(n_layers):
            hidden_dim = max(start_neurons // (2**i), min_neurons)
            layers += [nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #Reshape output into an awkward array with length equal to the number of features
        #Each sub-array is a number of logits corresponding to the number of classes for that feature
        return torch.split(self.net(x),self.classes_per_feature,dim=-1)


class EventRegressor(nn.Module):
    """Wires a TransformerBackbone + RegressionHead together.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, output_dim) predictions.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch):
        pooled = self.backbone(batch)
        return self.head(pooled)

class EventClassifier(nn.Module):
    """Wires a TransformerBackbone + ClassificationHead together.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, output_dim) predictions.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch):
        pooled = self.backbone(batch)
        return self.head(pooled)


class SetEventRegressor(nn.Module):
    """Wires a TransformerBackbone(_CLS) + CrossAttentionDecoder + RegressionHead
    together for per-query set prediction.

    `RegressionHead` is applied independently -- with weights shared across
    queries -- to each of the decoder's N query embeddings: `nn.Linear` (and
    the `GELU`/`Dropout` in between) only act on the last dimension, so
    feeding it a `(B, N, d_model)` tensor instead of `(B, d_model)` naturally
    yields `(B, N, output_dim)`, no per-query loop required. Intended for use
    with a permutation-invariant loss (e.g. Hungarian matching) over the N
    predictions.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, n_queries, output_dim)
    predictions.
    """

    def __init__(self, backbone, decoder, head):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.head = head

    def forward(self, batch):
        _, encoded, valid_mask = self.backbone(batch, return_tokens=True)
        queries = self.decoder(encoded, valid_mask)  # (B, N, d_model)
        return self.head(queries)  # (B, N, output_dim)


class SlotEventRegressor(nn.Module):
    """Wires a TransformerBackbone(_CLS) + SlotAttention + RegressionHead
    together for per-slot set prediction.

    Same idea as `SetEventRegressor`, but the query embeddings come from
    `SlotAttention` (slots *compete* for input tokens via softmax-over-slots
    attention, refined over a few GRU iterations) instead of independent
    cross-attention queries. `RegressionHead` is again applied with weights
    shared across slots, so `(B, n_slots, d_model)` in gives
    `(B, n_slots, output_dim)` out -- intended for use with a
    permutation-invariant loss over the n_slots predictions.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, n_slots, output_dim)
    predictions.
    """

    def __init__(self, backbone, slot_attention, head):
        super().__init__()
        self.backbone = backbone
        self.slot_attention = slot_attention
        self.head = head

    def forward(self, batch):
        _, encoded, valid_mask = self.backbone(batch, return_tokens=True)
        slots, _ = self.slot_attention(encoded, valid_mask)  # (B, n_slots, d_model)
        return self.head(slots)  # (B, n_slots, output_dim)


if __name__ == "__main__":
    import torch

    from hepsetreg.backbones.backbone import TransformerBackbone
    from hepsetreg.backbones.decoder import CrossAttentionDecoder
    from hepsetreg.backbones.slot_attention import SlotAttention

    # feature/target counts only -- see any run_*.py for the actual field names
    n_global_features, n_jet_features, n_lepton_features, n_target_features = 6, 9, 10, 3

    d_model = 32
    backbone = TransformerBackbone(
        n_global_features=n_global_features,
        n_jet_features=n_jet_features,
        n_lepton_features=n_lepton_features,
        d_model=d_model,
        nhead=4,
        num_layers=2,
    )
    head = RegressionHead(d_model=d_model, output_dim=n_target_features, n_layers=2, start_neurons=32)
    model = EventRegressor(backbone, head)

    batch_size, n_jets, n_leptons = 8, 5, 2
    dummy_batch = {
        "global": torch.randn(batch_size, n_global_features),
        "jet": torch.randn(batch_size, n_jets, n_jet_features),
        "jet_mask": torch.ones(batch_size, n_jets, dtype=torch.bool),
        "lepton": torch.randn(batch_size, n_leptons, n_lepton_features),
        "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
    }
    preds = model(dummy_batch)
    print(f"pooled predictions shape: {tuple(preds.shape)}")

    n_queries = 5
    decoder = CrossAttentionDecoder(d_model=d_model, n_queries=n_queries, nhead=4, num_layers=2)
    set_head = RegressionHead(d_model=d_model, output_dim=n_target_features, n_layers=2, start_neurons=32)
    set_model = SetEventRegressor(backbone, decoder, set_head)
    set_preds = set_model(dummy_batch)
    print(f"per-query predictions shape: {tuple(set_preds.shape)}")

    n_slots = 5
    slot_attention = SlotAttention(n_slots=n_slots, d_model=d_model, n_iters=3)
    slot_head = RegressionHead(d_model=d_model, output_dim=n_target_features, n_layers=2, start_neurons=32)
    slot_model = SlotEventRegressor(backbone, slot_attention, slot_head)
    slot_preds = slot_model(dummy_batch)
    print(f"per-slot predictions shape: {tuple(slot_preds.shape)}")
