# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class GlueConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="sentence_classification_head",
        metadata={"help": "name of the classification head to use"},
    )
    regression_target: bool = field(
        default=False,
    )


@register_criterion("glue", dataclass=GlueConfig)
class GlueCriterion(FairseqCriterion):
    def __init__(self, cfg: GlueConfig, task):
        super().__init__(task)
        self.num_classes = task.cfg.num_classes
        self.classification_head_name = cfg.classification_head_name
        self.regression_target = cfg.regression_target

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        is_training = model.training

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )

        label_targets = sample["target"]
        gpt_targets = sample["gpt_targets"]
        sample_size = label_targets.numel()
        nsentences = sample_size

        if not self.regression_target:
            if model.ft_type == 2:
                lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                loss = F.nll_loss(lprobs, label_targets.view(-1), reduction="sum")
                label_targets = label_targets.view(-1)
                pred_label = logits.argmax(dim=1)
            else:
                pad_id = self.task.dictionary.pad()
                length = sample["net_input"]["mlm_src_tokens"][:,1:].size(-1)
                loss_mask = torch.ones_like(gpt_targets) * pad_id
                loss_mask[:, :length] = sample["net_input"]["mlm_src_tokens"][:,1:]
                loss_mask = (loss_mask == pad_id) * (gpt_targets != pad_id)

                lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

                if is_training:
                    loss = - torch.gather(lprobs, -1, gpt_targets.unsqueeze(-1)).squeeze(-1) * loss_mask.float()
                    loss = loss.sum()
                    sample_size = loss_mask.int().sum()
                    nsentences = len(label_targets)
                else:
                    loss = - torch.gather(lprobs, -1, gpt_targets.unsqueeze(-1)).squeeze(-1) * loss_mask.float()
                    loss = loss.sum(-1)

                    pred_label = torch.argmin(loss.view(-1, self.num_classes), dim=1)
                    label_targets = label_targets.view(-1, self.num_classes)[:,0]
                    loss = loss.sum()
                    sample_size = len(label_targets)
                    nsentences = len(label_targets)
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            task_loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {}
        # mha & ffn regularization update
        if (
            hasattr(model.args, "mha_reg_scale_factor")
            and model.args.mha_reg_scale_factor != 0.0
        ):
            mha_reg_loss = model._get_adaptive_head_loss()
            loss += mha_reg_loss
            logging_output.update({"mha_reg_loss": mha_reg_loss})
        if (
            hasattr(model.args, "ffn_reg_scale_factor")
            and model.args.ffn_reg_scale_factor != 0.0
        ):
            ffn_reg_loss = model._get_adaptive_ffn_loss()
            loss += ffn_reg_loss
            logging_output.update({"ffn_reg_loss": ffn_reg_loss})

        logging_output.update(
            {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
            }
        )
        if not self.regression_target and not is_training:
            logging_output["ncorrect"] = (pred_label == label_targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mha_reg_loss_sum = sum(log.get("mha_reg_loss", 0) for log in logging_outputs)
        ffn_reg_loss_sum = sum(log.get("ffn_reg_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if mha_reg_loss_sum:
            metrics.log_scalar(
                "mha_reg_loss",
                mha_reg_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ffn_reg_loss_sum:
            metrics.log_scalar(
                "ffn_reg_loss",
                ffn_reg_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
