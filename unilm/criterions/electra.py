# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from unilm.criterions.unilm import UniLmConfig

@dataclass
class ElectraConfig(UniLmConfig):
    mlm_only: bool = field(
        default=False, metadata={"help": "use mlm objective only"}
    )
    seq2seq_only: bool = field(
        default=False, metadata={"help": "use seq2seq objective only"}
    )
    weight: float = field(
        default=50.0, metadata={"help": "weight to combine generator loss and discriminator loss"}
    )
    tpu: bool = II("common.tpu")


@register_criterion("electra", dataclass=ElectraConfig)
class ElectraLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.cfg = cfg
        self.mask_idx = task.mask_idx

    def mask_lm_loss(self, model, sample, reduce):
        masked_tokens = sample["src_tokens"].eq(self.mask_idx)
        sample_size = masked_tokens.int().sum()

        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        input_tokens, generator_logits, discriminator_logits = model(
            src_tokens=sample["src_tokens"],
            masked_tokens=masked_tokens)[:3]

        targets = sample["targets"]
        generator_targets = targets[targets.ne(self.padding_idx)]
        src_tokens = sample["src_tokens"].clone()
        src_tokens[masked_tokens] = generator_targets
        discriminator_targets = src_tokens.eq(input_tokens).long() 

        non_pad_tokens = sample["src_tokens"].ne(self.padding_idx)
        discriminator_logits = discriminator_logits[non_pad_tokens]
        discriminator_targets = discriminator_targets[non_pad_tokens]
        discriminator_sample_size = non_pad_tokens.int().sum()

        generator_loss = modules.cross_entropy(
            generator_logits.view(-1, generator_logits.size(-1)),
            generator_targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        discriminator_loss = F.nll_loss(
            F.log_softmax(
                discriminator_logits.view(-1, discriminator_logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            discriminator_targets.view(-1),
            reduction='sum',
            ignore_index=-100,
        )

        loss = generator_loss + self.cfg.weight * discriminator_loss * sample_size / discriminator_sample_size

        with torch.no_grad():
            positive = discriminator_logits[:, 1] >= discriminator_logits[:, 0]
            negative = discriminator_logits[:, 1] < discriminator_logits[:, 0]
            tp = (positive & (discriminator_targets == 1)).long().sum()
            fp = (positive & (discriminator_targets == 0)).long().sum()
            tn = (negative & (discriminator_targets == 0)).long().sum()
            fn = (negative & (discriminator_targets == 1)).long().sum()

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "mlm_loss": loss.clone() if self.tpu else loss.data.clone(),
            "generator_loss": generator_loss if self.tpu else generator_loss.data,
            "discriminator_loss": discriminator_loss if self.tpu else discriminator_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "discriminator_sample_size": discriminator_sample_size,
            "tp": tp.data,
            "fp": fp.data,
            "tn": tn.data,
            "fn": fn.data,
        }
        return loss, sample_size, logging_output
    
    def seq2seq_loss(self, model, sample, reduce):
        features = model(src_tokens=sample["src_tokens"], tgt_tokens=sample["tgt_tokens"])[0]
        features = features[:, sample["src_tokens"].size(1):, :]
        logits = model.output_layer(features)
        targets = sample["targets"]
        sample_size = targets.ne(self.padding_idx).int().sum()

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "seq2seq_loss": loss.clone() if self.tpu else loss.data.clone(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, sample_size, logging_outputs = 0, 0, {}

        if "mlm" in sample and not self.cfg.seq2seq_only:
            mlm_loss, mlm_sample_size, mlm_logging_output = self.mask_lm_loss(
                model, sample["mlm"], reduce=reduce
            )
            loss += mlm_loss
            sample_size = mlm_sample_size
            logging_outputs.update(mlm_logging_output)
        
        if "seq2seq" in sample and not self.cfg.mlm_only:
            seq2seq_loss, seq2seq_sample_loss, seq2seq_logging_output = self.seq2seq_loss(
                model, sample["seq2seq"], reduce=reduce
            )
            if "mlm" in sample:
                loss += seq2seq_loss * sample_size / seq2seq_sample_loss
                logging_outputs["loss"] += seq2seq_logging_output["loss"]
                logging_outputs["ntokens"] += seq2seq_logging_output["ntokens"]
                logging_outputs["nsentences"] += seq2seq_logging_output["nsentences"]
                logging_outputs["seq2seq_loss"] = seq2seq_logging_output["seq2seq_loss"]
            else:
                logging_outputs.update(seq2seq_logging_output)
                loss, sample_size = seq2seq_loss, seq2seq_sample_size
        
        return loss, sample_size, logging_outputs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        if "mlm_loss" in logging_outputs[0]:
            mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "mlm_loss", mlm_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if "seq2seq_loss" in logging_outputs[0]:
            seq2seq_loss_sum = sum(log.get("seq2seq_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "seq2seq_loss", seq2seq_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if "discriminator_sample_size" in logging_outputs[0]:
            discriminator_sample_size = sum(log.get("discriminator_sample_size", 0) for log in logging_outputs)
            generator_loss_sum = sum(log.get("generator_loss", 0) for log in logging_outputs)
            discriminator_loss_sum = sum(log.get("discriminator_loss", 0) for log in logging_outputs)
            tp = sum(log.get("tp", 0) for log in logging_outputs)
            fp = sum(log.get("fp", 0) for log in logging_outputs)
            tn = sum(log.get("tn", 0) for log in logging_outputs)
            fn = sum(log.get("fn", 0) for log in logging_outputs)
            replace_acc = tn / (tn + fp + 1e-5)
            non_replace_acc = tp / (tp + fn + 1e-5)
            replace_rate = (tn + fp) / (tn + fp + tp + fn + 1e-5)
            metrics.log_scalar(
                "generator_loss", generator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "discriminator_loss", discriminator_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
            )
            metrics.log_scalar(
                "replace_acc", replace_acc, tn + fp, round=4
            )
            metrics.log_scalar(
                "non_replace_acc", non_replace_acc, tp + fn, round=4
            )
            metrics.log_scalar(
                "replace_rate", replace_rate, tn + fp + tp + fn, round=4
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
