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
from unilm.data.lm_loader_v2 import MultiSourceLmLoader
from unilm.data.basic_loader import ConcatLoader

from unilm.data.vl.vl_loader import WdsLoaderConfig
try:
    from unilm.data.vl.itlv_lm import ItlvLoader
    from unilm.data.vl.vlm_loader import VlmLoader, VlmLoader4ArrowDataset
except ImportError:
    print('Please install webdataset: pip install webdataset for VL dataset')

from unilm.data.vl2.wild_loader import WildLoader
from deepspeed.runtime.engine import DeepSpeedEngine


logger = logging.getLogger(__name__)


@dataclass
class ImageGPTPretrainingConfig(GPTPretrainingConfig, WdsLoaderConfig):
    max_image_num: int = field(default=5, metadata={"help": ""})
    image_token_length: int = field(default=64, metadata={"help": ""})
    wild_data_dir: str = field(default="", metadata={"help": ""})


@register_task("image_gpt_pretraining_wild", dataclass=ImageGPTPretrainingConfig)
class ImageGPTWildTask(GPTTask):
    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg, dictionary, tokenizer)
        self.vlm_model = None

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint=from_checkpoint)
        self.vlm_model = model
        return model

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
        wild_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.wild_data_dir}/json/train.json')),
            'data_dir': self.cfg.wild_data_dir,
            'shuffle': True})

        vl_loader = WildLoader(
            self.cfg,
            wild_dataset,
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
            no_prefetch=False,
        )

        lm_loader = LMLoader(
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

        concat_loader = ConcatLoader([vl_loader, lm_loader])
        return concat_loader

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        json_split_name = self.cfg.train_json_split_name if split == "train" else split
        self.datasets[split] = {
            'data': json.load(open(f'{self.cfg.data}/json/{json_split_name}.json')),
            'data_dir': self.cfg.data,
            'shuffle': True if split == 'train' else False, }
        self.datasets[split] = Namespace(**self.datasets[split])

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

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                vl_loss, sample_size, logging_output = criterion(model, sample["vl"], loss_name="image")
        if ignore_grad:
            vl_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            if isinstance(model, DeepSpeedEngine):
                model.backward(vl_loss)
            else:
                optimizer.backward(vl_loss)
        
        agg_loss += vl_loss.detach().item()
        agg_sample_size += sample_size
        agg_logging_output.update(logging_output)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                gpt_loss, sample_size, logging_output = criterion(model, sample["gpt"], loss_name="gpt")
        if ignore_grad:
            gpt_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            if isinstance(model, DeepSpeedEngine):
                model.backward(gpt_loss)
            else:
                optimizer.backward(gpt_loss)

        agg_loss += gpt_loss.detach().item()
        agg_sample_size += sample_size
        for key, value in logging_output.items():
            if key not in agg_logging_output:
                agg_logging_output[key] = value
            else:
                agg_logging_output[key] += value

        return agg_loss, agg_sample_size, agg_logging_output

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


@register_task("image_gpt_pretraining_vlm_only_wild", dataclass=ImageGPTPretrainingConfig)
class OnlyVLMImageGPTWildTask(ImageGPTWildTask):

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
        vl_loader = WildLoader(
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

        return vl_loader

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
                loss, sample_size, logging_output = criterion(
                    model, sample['image'])
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
