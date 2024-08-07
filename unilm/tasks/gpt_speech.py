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
from unilm.data.speech.speech_loader import SpeechLoader
from unilm.tasks.gpt_base import GPTPretrainingConfig, GPTTask
from unilm.data.basic_loader import ConcatLoader


logger = logging.getLogger(__name__)


@dataclass
class SpeechGPTPretrainingConfig(GPTPretrainingConfig):
    audio_data_dir: str = field(default="", metadata={"help": ""})
    audio_segment_size: int = field(default=2, metadata={"help": ""})
    audio_tokens_size: int = field(default=0, metadata={"help": ""})
    audio_root_path: str = field(default="", metadata={"help": ""})
    audio_downsample_rate: int = field(default=2, metadata={"help": "downsample rate, fbank size / audio_hidden_size"})

@register_task("speech_gpt_pretraining", dataclass=SpeechGPTPretrainingConfig)
class SpeechGPTTask(GPTTask):
    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg, dictionary, tokenizer)

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
        speech_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.audio_data_dir}/json/train.json')),
            'data_dir': self.cfg.audio_data_dir,
            'shuffle': True})

        speech_loader = SpeechLoader(
            self.cfg,
            speech_dataset,
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

        concat_loader = ConcatLoader([speech_loader, lm_loader])
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
                audio_loss, sample_size, logging_output = criterion(model, sample["audio"], loss_name="audio")
        if ignore_grad:
            audio_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(audio_loss)
        
        agg_loss += audio_loss.detach().item()
        agg_sample_size += sample_size
        agg_logging_output.update(logging_output)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                gpt_loss, sample_size, logging_output = criterion(model, sample["gpt"], loss_name="gpt")
        if ignore_grad:
            gpt_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
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

@register_task("only_speech_gpt_pretraining", dataclass=SpeechGPTPretrainingConfig)
class OnlySpeechGPTTask(SpeechGPTTask):
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
                audio_loss, sample_size, logging_output = criterion(model, sample["audio"], loss_name="audio")
        if ignore_grad:
            audio_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(audio_loss)
        
        agg_loss += audio_loss.detach().item()
        agg_sample_size += sample_size
        agg_logging_output.update(logging_output)

        return agg_loss, agg_sample_size, agg_logging_output
