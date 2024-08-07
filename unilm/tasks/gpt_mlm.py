# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Optional
import logging
from argparse import Namespace
import json
import torch

from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import FairseqTask, register_task
# from unilm.data.lm_loader import LMLoader
from unilm.data.spm_lm_loader import SpmLmLoader as LMLoader
from unilm.tasks.gpt_base import GPTPretrainingConfig, GPTTask


logger = logging.getLogger(__name__)


@dataclass
class MLMGPTPretrainingConfig(GPTPretrainingConfig):
    mlm_cut_length: int = field(
        default=128,
        metadata={"help": "max input length for mlm"},
    )

    mlm_tokens_proportion: float = field(
        default=0.23,
        metadata={
            "help": "proportion of tokens produced by mlm"
        },
    )


@register_task("mlm_gpt_pretraining", dataclass=MLMGPTPretrainingConfig)
class MLMGPTTask(GPTTask):

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if "tnlg" in self.cfg.data:
            self.datasets[split] = {
                # 'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                # 'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub-noarvix-nopubmed.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub-noarvix-nopubmed-mtnlg.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        else:
            self.datasets[split] = {
                'data': json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        self.datasets[split] = Namespace(**self.datasets[split])

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):
        return LMLoader(
            self.cfg,
            dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
        )
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample['text'])
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
