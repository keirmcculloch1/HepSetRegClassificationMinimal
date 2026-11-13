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

import numpy as np
import awkward as ak
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.backbones.head import EventClassifier, ClassificationHead
from hepsetreg.config import load_config
from hepsetreg.datasets.EventDatasetBase import ClassificationEventDataModule, ClassificationEventDataset, _identity_collate, load_arrays
from hepsetreg.training.lightning_module import EventClassificationModule

import sklearn.metrics as metrics

import scipy.stats

import scipy.special

#Include binning parameters in model keys for predictions
MODEL_KEYS = ["d_model", "nhead", "num_layers", "binning"]

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
    #Create a list of the number of classes for each feature
    #Eg. if you target ttbar_mass and ttbar_pt with 100 classes for each it would be (100,100)
    classes_per_feature = [config["binning"][target_feature]["num_bins"] for target_feature in features["target"]]

    head = ClassificationHead(d_model=config["d_model"], num_features=len(features["target"]), classes_per_feature=classes_per_feature)
    return EventClassifier(backbone, head)

#Calculates the bin edges for a given target feature
def calculate_bin_edges(config,feature_name):

    #Load binning parameters
    leftmost_bin_edge = config["binning"][feature_name]["left_edge"]
    rightmost_bin_edge = config["binning"][feature_name]["right_edge"]
    number_of_bins = config["binning"][feature_name]["num_bins"]

    #Equally spaced bins
    return np.linspace(leftmost_bin_edge,rightmost_bin_edge,number_of_bins+1)

def unbin(bins,config,feature_name):

    min_value = config["binning"][feature_name]["left_edge"]
    max_value = config["binning"][feature_name]["right_edge"]
    number_of_bins = config["binning"][feature_name]["num_bins"]

    return min_value + (bins/number_of_bins + 0.5) * (max_value-min_value)

def frequencies(bin_values,config,feature_name):

    number_of_bins = config["binning"][feature_name]["num_bins"]

    return np.bincount(bin_values,minlength=number_of_bins)

#Sum of top-k accuracy scores with each contribution weighted by k
#Eg. for top-1 accuracy of 0.1 and top-2 accuracy of 0.22 the total would be 0.1+(0.12/2)=0.16
def weighted_top_k_accuracy_sum(targets,prediction_logits):

    #Total number of classes
    num_classes = len(prediction_logits[0])

    #Prevent issues when one of the classes doesn't appear
    labels = np.arange(num_classes)

    #Final output for this function, weighted sum of contributions
    total_accuracy_score = 0

    #Used to calculate differences
    previous_accuracy_score = 0

    #Loop from 1 to the number of classes
    for i in range(1,num_classes+1):

        #Calculate top-i accuracy
        current_accuracy_score = metrics.top_k_accuracy_score(targets,prediction_logits,labels=labels,k=i)

        #Calculate contribution (difference)
        accuracy_difference = current_accuracy_score - previous_accuracy_score

        #Add weighted contribution
        total_accuracy_score += current_accuracy_score/i

        #Update previous accuracy score
        previous_accuracy_score = current_accuracy_score

    return total_accuracy_score

#Average rank of the target amongst the prediction logits
#Divided by the number of classes
def normalised_average_target_rank(targets,prediction_logits):

    #Total number of classes
    num_classes = len(prediction_logits[0])

    #Unsqueeze turns targets from shape N_events to N_events x 1
    #Gather gets the logit corresponding to the true target
    #Resulting in an N_events x 1 array of the logit value of the target classes
    target_logits = torch.gather(prediction_logits, 1, targets.unsqueeze(1))

    #Pytorch broadcasts target_logits from N_events x 1 to N_events x N_classes, filling each row with the same value
    #The number of prediction logits greater than the logit corresponding to the target gives the ranking of the target
    #Eg. if 2 logits are larger than the target logit then the target is ranked #3 (hence the +1)
    ranks = (prediction_logits > target_logits).sum(dim=1) + 1

    #Take average of the ranks of the targets
    mean_rank = ranks.float().mean()

    #Divide by the number of classes
    return mean_rank / num_classes

#Kolmogorov-Smirnov (KS) metric
#Takes bin frequencies and equally spreads the number of values inside a bin across that bin to produce the CDF
#Calculates the KS based off of this "continuous" CDF with the true continuous targets
def ks_binned(continuous_targets, pred_frequencies, bin_edges):

    #Normalise cumulative sum and add a 0 value to the beginning
    cdf_vals = np.concatenate([[0.0], np.cumsum(pred_frequencies) / np.sum(pred_frequencies)])

    #Create the CDF by interpolating the frequencies across the bins
    cdf = lambda x: np.interp(x, bin_edges, cdf_vals)

    #Calculate KS metric from the observations (targets) against the CDF (predictions)
    res = scipy.stats.ks_1samp(continuous_targets, cdf)

    return res.statistic

def train(config):
    # Not a hyperparameter -- an environment setting. PyTorch defaults to
    # using every visible CPU core for every op, even tiny ones; on a
    # many-core shared machine that causes severe thread-oversubscription
    # overhead (measured ~20x slower elsewhere in this project for small
    # per-op work). Capping it avoids that regardless of model size.
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("medium")  # allows TF32 matmuls on Ampere+ GPUs

    #Calculate bin edges for each target feature
    features = config["features"]
    feature_bin_edges = [calculate_bin_edges(config,target_feature) for target_feature in features["target"]]

    dm = ClassificationEventDataModule(
        config["train_path"], config["val_path"], features=config["features"], feature_bin_edges = feature_bin_edges, batch_size=config["batch_size"]
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

    #Calculate bin edges for each target feature
    features = config["features"]
    feature_bin_edges = [calculate_bin_edges(config,target_feature) for target_feature in features["target"]]

    target_features = model_config["features"]["target"]
    ds = ClassificationEventDataset(config["data_path"], features=model_config["features"], feature_bin_edges = feature_bin_edges, stats=stats, max_jets=max_jets)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=_identity_collate)

    # Trainer.predict() puts the module in eval mode, disables gradients,
    # and moves both module and batches to the accelerator -- no manual
    # .eval()/.to(device)/no_grad()/per-batch transfer loop needed.
    trainer = pl.Trainer(accelerator=config["accelerator"], devices=1, logger=False)
    outputs = trainer.predict(module, dataloaders=loader)  # list of per-batch {"pred", "target"} dicts

    #Prediction logits (values can be outside of 0-1 range, don't represent a probability yet)
    #pred_logits = [ torch.cat( [out["pred"][i] for out in outputs], dim=0) for i in range(len(target_features)) ]
    pred_logits = [torch.cat([out["pred"][i] for out in outputs]) for i in range(len(target_features))]

    #Binned target values
    targets = [torch.cat([out["target"][:,i] for out in outputs]) for i in range(len(target_features))]

    #Original continuous target values for peformance metrics
    #continuous_targets = np.array(load_arrays(config["data_path"],model_config["features"])[3])
    continuous_targets = np.array([ds.target_tensor.numpy()[:,i] for i in range(len(target_features))])

    #Frequency with which each bin appears
    target_frequencies = [frequencies(targets[i],config,feature) for i,feature in enumerate(target_features)]
    #Filter out zero frequency values to prevent divide by zero errors
    filter_zero_map = [feature_freq != 0 for feature_freq in target_frequencies]
    #Apply filter
    filtered_target_frequencies = [t_feature_freq[filter_zero_map[i]] for i,t_feature_freq in enumerate(target_frequencies)]

    #Prediction probabilities for performance metrics (not necessary for calculating predictions)
    #pred_probs = [torch.softmax(x, dim=-1) for x in pred_logits]
    pred_probs = [[torch.softmax(event, dim=-1) for event in feature] for feature in pred_logits]

    #Bin prediction is highest value logit index
    #preds = torch.stack( [torch.argmax(x, dim=-1) for x in pred_logits], dim=1)
    preds = torch.tensor([[torch.argmax(event, dim=-1) for event in feature] for feature in pred_logits])

    #Unbinned prediction values for comparison to continuous target values
    #unbinned_preds = np.dstack([unbin(preds[i],config,feature) for i,feature in enumerate(target_features)])[0]
    unbinned_preds = np.array([unbin(preds[i],config,feature) for i,feature in enumerate(target_features)])

    #Frequency with which each bin appears
    pred_frequencies = [frequencies(preds[i],config,feature) for i,feature in enumerate(target_features)]
    #Apply the same filter as the targets
    #The predictions can have zero frequency bins without divide by zero errors
    filtered_pred_frequencies = [p_feature_freq[filter_zero_map[i]] for i,p_feature_freq in enumerate(pred_frequencies)]

    #mae = (preds - targets).abs().float().mean(dim=0)  # mean over events, per target feature
    #mae = ak.mean(np.abs(preds - targets), axis=0)
    #mae = (unbinned_preds - continuous_targets).abs().float().mean(dim=0)
    mae = np.mean(np.abs(unbinned_preds - continuous_targets),axis=0)

    #Accuracy is the proportion of events guess correctly
    #accuracy = (preds == targets).float().mean()
    #accuracy = ak.mean(preds == targets)
    accuracy = np.mean(preds == targets)

    for name, value in zip(target_features, mae):
        print(f"{name:12s} MAE: {value:.4f}")

    #Different ways of expressing the target data depending on the metric
    target_datasets = {
        "binned" : targets,
        "continuous" : continuous_targets,
        "frequencies" : target_frequencies,
        "filtered frequencies" : filtered_target_frequencies,
    }

    #Different ways of expressing the predictions depending on the metric
    prediction_datasets = {
        "logits" : pred_logits,
        "probabilities" : pred_probs,
        "binned" : preds,
        "unbinned" : unbinned_preds,
        "frequencies" : pred_frequencies,
        "filtered frequencies" : filtered_pred_frequencies,
    }

    #List of metrics to calculate and save
    performance_metrics = [
        {"function" : metrics.accuracy_score, "name" : "accuracy", "targets" : "binned", "predictions" : "binned","kwargs" : lambda i : {}},
        {"function" : metrics.precision_score, "name" : "precision", "targets" : "binned", "predictions" : "binned", "kwargs" : lambda i : {"average" : "weighted"}},
        {"function" : metrics.recall_score, "name" : "recall", "targets" : "binned", "predictions" : "binned", "kwargs" : lambda i : {"average" : "weighted"}},
        {"function" : metrics.f1_score, "name" : "f1_score", "targets" : "binned", "predictions" : "binned", "kwargs" : lambda i : {"average" : "weighted"}},
        {"function" : metrics.top_k_accuracy_score, "name" : "top 2 accuracy", "targets" : "binned", "predictions" : "logits", "kwargs" : lambda i : {"k" : 2, "labels" : np.arange(len(pred_logits[i][0]))}},
        {"function" : lambda t, p : scipy.stats.chisquare(p,t).statistic/(len(t)-1), "name" : "reduced scipy_chi2", "targets" : "filtered frequencies", "predictions" : "filtered frequencies", "kwargs" : lambda i : {}},
        {"function" : ks_binned, "name" : "kolmogorov-smirnov", "targets" : "continuous", "predictions" : "frequencies", "kwargs" : lambda i : {"bin_edges" : feature_bin_edges[i]}},
        {"function" : lambda t, p : np.sum(scipy.special.kl_div(p,t)), "name" : "kullback-leibler", "targets" : "filtered frequencies", "predictions" : "filtered frequencies", "kwargs" : lambda i : {}},
        {"function" : weighted_top_k_accuracy_sum, "name" : "weighted top k accuracy sum", "targets" : "binned", "predictions" : "logits", "kwargs" : lambda i : {}},
        {"function" : normalised_average_target_rank, "name" : "normalised average target rank", "targets" : "binned", "predictions" : "logits", "kwargs" : lambda i : {}},
    ]

    #Save performance metrics to a file instead of cluttering with too many print statements
    run_name = config.get("run_name")
    with open(run_name+"_metrics.txt","w") as writefile:
        writefile.write("Performance metrics:\n")

        #Seperate metrics for each feature
        for i,feature in enumerate(target_features):
            writefile.write(f"\t{feature}\n")

            for perf_metric in performance_metrics:

                score = perf_metric["function"](target_datasets[perf_metric["targets"]][i],prediction_datasets[perf_metric["predictions"]][i],**perf_metric["kwargs"](i))

                writefile.write(f"\t\t{perf_metric['name']}: {score:.4f}\n")

    # save predictions + targets to parquet, as two record groups -- e.g.
    # arr.targets.ttbar_mass, arr.predictions.ttbar_mass
    # .contiguous(): a column slice of a 2D tensor is a strided,
    # non-contiguous view, which pyarrow rejects with "ndarray is not contiguous"
    target_fields = {name: ak.Array(targets[i].contiguous().numpy()) for i, name in enumerate(target_features)}
    pred_fields = {name: ak.Array(preds[i].contiguous().numpy()) for i, name in enumerate(target_features)}

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
