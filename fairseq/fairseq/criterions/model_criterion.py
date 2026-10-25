# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


logger = logging.getLogger(__name__)


@dataclass
class ModelCriterionConfig(FairseqDataclass):
    loss_weights: Dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "weights for the loss terms"},
    )
    log_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "additional output keys to log"},
    )


@register_criterion("model", dataclass=ModelCriterionConfig)
class ModelCriterion(FairseqCriterion):
    """
    This criterion relies on the model to supply losses.
    The losses should be a dictionary of name -> scalar returned by
    the model either by including it in the net_output dict or by
    implementing a get_losses(net_output, sample) method. The final loss is
    a scaled sum of all losses according to weights in loss_weights.
    If no weights are provided, then all losses are scaled by 1.0.

    The losses will be automatically logged. Additional keys from
    net_output dict can be logged via the log_keys parameter.
    """

    # class attributes:
    compute_extra_test_metrics = False
    topks = (1, 3, 5, 10)

    def __init__(self, task, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.loss_weights = loss_weights
        self.log_keys = log_keys

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"]) # forward call 

        scaled_losses = {}

        if hasattr(model, "get_losses"):
            losses = model.get_losses(net_output, sample)
        elif isinstance(net_output, dict) and "losses" in net_output:
            losses = net_output["losses"]
        else:
            raise Exception("Could not retrieve losses")

        for lk, p in losses.items(): # iterate over every key (lk) of the losses dict 
            try:
                coef = 1.0 if len(self.loss_weights) == 0 else self.loss_weights[lk]
            except KeyError:
                logger.error(
                    f"weight for loss {lk} is not in loss_weights ({self.loss_weights})"
                )
                raise
            if coef != 0 and p is not None:
                scaled_losses[lk] = coef * p.float()

        loss = sum(scaled_losses.values())

        if "sample_size" in net_output:
            sample_size = net_output["sample_size"]
        else:
            sample_size = loss.numel()

        if reduce and loss.numel() > 1:
            loss = loss.sum()

        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "_world_size": 1,
        }

        # Compute accuracy if we are in validation step (also works for testing)
        if not model.training:
            # we need the model's predictions and the ground truth
            if "logits" in net_output and "gloss_labels" in sample["net_input"]: 
                logits = net_output["logits"]
                preds = net_output["logits"].argmax(dim=-1).detach().cpu() + 1 # target range is 1 - GlossarySize
                targets = sample["net_input"]["gloss_labels"]
                logging_output["preds"] = preds
                logging_output["target"] = targets

                # --- NEW ---
                if self.compute_extra_test_metrics:
                    # logits is a tensor (Batch_Size, Glossray_Size)
                    k = min(max(self.topks), logits.size(-1)) # we can compute up to Glossary_Size top-k accuracy
                    logging_output["topk_preds"] = (
                        logits.topk(k, dim=-1).indices.detach().cpu() + 1 # native method of the torch.Tensor class. 
                    ) # topk returns a tuple w/ the k largest logits of the specified dimension and the indices (gloss IDs), but we only keep the IDs. 
                # -----------
                

        for lk in self.log_keys:
            if lk in net_output and net_output[lk] is not None:
                if not torch.is_tensor(net_output[lk]) or net_output[lk].numel() == 1:
                    logging_output[lk] = float(net_output[lk])
                else:
                    for i, v in enumerate(net_output[lk]):
                        logging_output[f"{lk}_{i}"] = float(v)

        if len(scaled_losses) > 1:
            for lk, l in scaled_losses.items():
                if l.numel() > 1:
                    l = l.sum()
                logging_output[f"loss_{lk}"] = l.item()

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3) # 1/N Σ J(y, y_pred) 
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        # logging_outputs is a list of logging_output (with metrics from all the batches).
        # this makes sure that this block is only executed during validation (otw preds wouldn't exist)
        if len(logging_outputs) > 0 and "preds" in logging_outputs[0]:
            # Concatenate the tensors from all the batches:
            all_preds = torch.cat([log["preds"] for log in logging_outputs if "preds" in log]).cpu() # final size (N, ), where N is the number of samples of the validation set
            all_targets = torch.cat([log["target"] for log in logging_outputs if "target" in log]).cpu()

            if len(all_preds) > 0 and len(all_targets) > 0:

                # Compute micro and macro accuracy on the validation set
                if not ModelCriterion.compute_extra_test_metrics: 
                    # 1. Micro Accuracy (total correct predictions / total number of predictions)
                    acc_micro = (all_preds == all_targets).float().mean().item() # convert true to 1 and false to 0, then compute the mean.  
                    
                    # 2. Macro Accuracy (mean of correct predictions per gloss)
                    unique_classes = torch.unique(all_targets)
                    class_accuracies = []   # list of accuracy per gloss

                    for c in unique_classes: # per gloss
                        class_mask = (all_targets == c) # True: ground truth is c - False: gt is anything else 
                        class_acc = (all_targets[class_mask] == all_preds[class_mask]).float().mean() # we filter the samples that are not class c  
                        class_accuracies.append(class_acc)

                    acc_macro = torch.stack(class_accuracies).mean().item() if class_accuracies else 0.0 

                    # Register the metrics in fairseq (to visualize them in wandb)
                    metrics.log_scalar("val_acc_micro", float(acc_micro), round=4)
                    metrics.log_scalar("val_acc_macro", float(acc_macro), round=4)

                # Compute the top-k accuracies and the confusion matrix on the test set
                else:
                    all_topk_preds = torch.cat( # shape (Total_Batch_Size, K)
                        [
                            log["topk_preds"]
                            for log in logging_outputs
                            if "topk_preds" in log
                        ]
                    ).cpu()

                    wandb_metrics = {
                        # "val_acc_micro": float(acc_micro),
                        # "val_acc_macro": float(acc_macro),
                    }
 
                    if len(all_topk_preds) > 0: # more than 1 sample
                        for k in ModelCriterion.topks:
                            k = min(k, all_topk_preds.size(-1))
                            correct_topk = (
                                all_topk_preds[:, :k] == all_targets.unsqueeze(1) # unsqueeze to avoid shape like (N,)
                            ).any(dim=1) # checks for every row if across the k columns there's a True and if so returns True (N, )
                            # micro-accuracy:
                            topk_acc = correct_topk.float().mean().item() # convert to 0s and 1s, compute the % of samples that had the correct gloss among the k most probable preds. item(): from PyTorch Tensor to Python float 
                            metrics.log_scalar(
                                f"test_acc_top{k}", float(topk_acc), round=4
                            )
                            wandb_metrics[f"test_acc_top{k}"] = float(topk_acc)

                    try:
                        import wandb
                        from sklearn.metrics import confusion_matrix
                        import os
                        import numpy as np

                        cm = wandb.plot.confusion_matrix(
                                            y_true=all_targets.numpy(), preds=all_preds.numpy()
                                        )
                        if wandb.run is not None:
                            wandb_metrics["confusion_matrix"] = cm
                            wandb.log(wandb_metrics)
                            logger.info("Metrics and Confusion Matrix logged W&B.")

                        cm_sklearn = confusion_matrix(y_true=all_targets.numpy(), y_pred=all_preds.numpy())

                        base_dir = os.environ.get("WANDB_DIR", ".")
                        target_dir = os.path.join(base_dir, "metrics")
                        os.makedirs(target_dir, exist_ok=True)
                        output_path = os.path.join(target_dir, "confusion_matrix.npy")
                        # save the confusion matrix:
                        np.save(output_path, cm_sklearn)
                        output_path = os.path.join(target_dir, "topk_pred.npy")
                        # save the topk predictions of the whole test split.
                        np.save(output_path, all_topk_preds)
                    except Exception as e:
                        logger.warning(f"Error loading to W&B: {e}")
            # -----------

        builtin_keys = {        # this keys won't be summed in the block below
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "_world_size",
            "preds",
            "target",
            "topk_preds",
        }

        world_size = utils.item(
            sum(log.get("_world_size", 0) for log in logging_outputs)
        )

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss_"):
                    metrics.log_scalar(k, val / sample_size, sample_size, round=3)
                else:
                    metrics.log_scalar(k, val / world_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
