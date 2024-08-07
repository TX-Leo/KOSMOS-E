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
from unilm.data.text_tuning_lm_loader import SpmLmLoader as InstruLMLoader
from unilm.data.spm_lm_loader import SpmLmLoader as LMLoader
from unilm.tasks.gpt_base import GPTPretrainingConfig, GPTTask
from unilm.data.lm_loader_v2 import MultiSourceLmLoader
from unilm.data.basic_loader import MixLoader

from unilm.data.vl.vl_loader import WdsLoaderConfig
try:
    from unilm.data.vl.itlv_lm import ItlvLoader
    from unilm.data.vl.vlm_loader import VlmLoader, VlmLoader4ArrowDataset
except ImportError:
    print('Please install webdataset: pip install webdataset for VL dataset')

from unilm.data.vl2.wild_loader_v2 import WildLoader
from unilm.data.vl2.laion2b_loader import Laion2BLoader
from unilm.data.vl2.laion2b_obj_loader import Laion2BObjLoader
from unilm.data.vl2.laion2b_obj_ref_loader import Laion2BObjRefLoader
from unilm.data.vl2.laion2b_obj_tune_loader import Laion2BObjTuneLoader

# from unilm.data.vl2.vl_tuning_loader import InstructVLLoader
from unilm.data.vl2.vl_tuning_loader_v2 import InstructVLLoader

from deepspeed.runtime.engine import DeepSpeedEngine


logger = logging.getLogger(__name__)


@dataclass
class ImageGPTPretrainingConfig(GPTPretrainingConfig, WdsLoaderConfig):
    max_image_num: int = field(default=5, metadata={"help": ""})
    image_token_length: int = field(default=64, metadata={"help": ""})
    wild_data_dir: str = field(default="", metadata={"help": ""})
    wild_batch_size: int = field(default=4, metadata={"help": ""})
    wild_image_from_local: bool = field(default=False, metadata={"help": ""})
    laion_data_dir: str = field(default="", metadata={"help": ""})
    laion_batch_size: int = field(default=32, metadata={"help": ""})
    instru_data_dir: str = field(default="", metadata={"help": ""})
    instru_batch_size: int = field(default=8, metadata={"help": ""})
    input_resolution: int = field(default=224, metadata={"help": ""})
    vl_instru_data_dir: str = field(default="", metadata={"help": ""})
    vl_instru_batch_size: int = field(default=32, metadata={"help": ""})

    # grounding setting
    quantized_size: int = field(default=16, metadata={"help": "used to discrete the continuous coordinates"})
    locate_special_token: int = field(default=0, metadata={"help": "used to discrete the continuous coordinates"})
    phrase_mode: str = field(default="phrase", metadata={"help": "mode in phrase,expression"})
    simplest_grounding_prompt: bool = field(default=False, metadata={"help": "use simplest_grounding_prompt or not"})
    training_image_only_resize: int = field(default=0, metadata={"help": "only use resize transform during pretraining"})

    # some parameters to filter the bounding box used for pretraining
    box_score_threshold: float = field(default=0.65, metadata={"help": "filter the box with low confidence"})
    mix_no_object_prob: float = field(default=0., metadata={"help": "prob of using the image-text pairs that w/o box"})
    use_object_bbox_prob: float = field(default=1., metadata={"help": "prob of using the image-text pairs that w box"})
    
    # tuning setting
    region_caption_template_prob: float = field(default=0., metadata={"help": "as description"})
    region_caption_template_data: str = field(default='refcocog', metadata={"help": "as description"})
    vl_instru_dataset: str = field(default="llava,refcoco,lvis", metadata={"help": ""})
    
    flickr_tuning_mode: str = field(default="grounding,caption,pron", metadata={"help": ""})
    flickr_tuning_mode_prob: str = field(default="1,0,0", metadata={"help": ""})
    flickr_caption_template: float = field(default=1, metadata={"help": "use tempalte for caption  on flickr or not"})
    flickr_caption_ignore_eos_gra: float = field(default=0, metadata={"help": "ignore eos gradient when using flickr caption"})
    text_tuning_template: float = field(default=0, metadata={"help": "use tempalte for tuning text or not"})
    llava_conversation_multiturn: float = field(default=1, metadata={"help": "enable llava_conversation_multiturn"})
    llava_tuning_splits: str = field(default="detail,conversation,complex", metadata={"help": ""})
    llava_question_template: float = field(default=0, metadata={"help": "use tempalte for question-answer on llava or not"})
    vcr_tuning_mode: str = field(default="qa", metadata={"help": "vcr dataset tuning mode"})
    
    tuning_image_only_resize: float = field(default=0, metadata={"help": "only use resize during runing phase"})
    
    # for referring exp built on laion
    refer_use_single_obj_prob: float = field(default=1, metadata={"help": "the prob of using image-caption that only contains one"})
    refer_ignore_eos_grad: float = field(default=0, metadata={"help": "ignore eos gradient when using refer data"})
    refer_image_only_resize: float = field(default=1, metadata={"help": "only use resize for refer data"})
    refer_use_simple_template: float = field(default=1, metadata={"help": "Use simple templates for refer data"})
    refer_use_short_expression_prob: float = field(default=1, metadata={"help": "the prob of using short expression"})
    refer_use_exp_w_multibox: float = field(default=1, metadata={"help": "Use the expression for multi boxes"})
    refer_use_exp_start_w_a_the: float = field(default=0, metadata={"help": "Use the expression with the begaining of a/an/the"})
    
    # for using laion tuning dataloader
    enable_laion_tune_dataloader: bool = field(default=False, metadata={"help": "Enable the laion tuning dataloader"})
    laion_tune_min_box: int = field(default=2, metadata={"help": "minium box for laion tuning data"})
    laion_tune_image_only_resize: float = field(default=1, metadata={"help": "only use resize for refer data"})
    laion_tune_use_single_box_prob: float = field(default=0, metadata={"help": "the prob of using image-txt with low box( < laion_tune_min_box)"})
    laion_tune_use_single_box_mode: str = field(default="box", metadata={"help": "gradient on box or all"})
    laion_tune_use_caption_template: float = field(default=0, metadata={"help": "laion_tune_use_caption_template"})
    
    data_weights: str = field(default="4,32,1,1,32,0", metadata={"help": "wild,laion,gpt,inst,vl_inst,refer"})
    
    
@register_task("gpt_visual_text_tuning", dataclass=ImageGPTPretrainingConfig)
class VLGPTObjTuningTask(GPTTask):
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
            max_sentences=self.cfg.wild_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )

        laion_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.laion_data_dir}/json/train.json')),
            'data_dir': self.cfg.laion_data_dir,
            'shuffle': True})

        if self.cfg.enable_laion_tune_dataloader:
            lain_vl_loader = Laion2BObjTuneLoader(
                self.cfg,
                laion_dataset,
                self.dictionary,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=self.cfg.laion_batch_size,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
                no_prefetch=False,
            )
        else:    
            lain_vl_loader = Laion2BObjLoader(
                self.cfg,
                laion_dataset,
                self.dictionary,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=self.cfg.laion_batch_size,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
                no_prefetch=False,
            )
        
        lain_vl_refer_loader = Laion2BObjRefLoader(
            self.cfg,
            laion_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.laion_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )
        
        vl_instru_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.vl_instru_data_dir}/json/train.json')),
            'data_dir': self.cfg.vl_instru_data_dir,
            'shuffle': True})

        vl_instru_loader = InstructVLLoader(
            self.cfg,
            vl_instru_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.vl_instru_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )

        instru_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.instru_data_dir}/json/train.json')),
            'data_dir': self.cfg.instru_data_dir,
            'shuffle': True})

        instru_lm_loader = InstruLMLoader(
            self.cfg,
            instru_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.instru_batch_size,
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

        data_weight = [float(x) for x in self.cfg.data_weights.split(',')]
        data_weight = [x / sum(data_weight) for x in data_weight]
        logger.info(f"data weights: {data_weight}")
        # interleaved, caption, pt, instru, vl instru, refering
        # import pdb; pdb.set_trace()
        if len(data_weight) < 6:
            data_weight.extend([0,]*(6 - len(data_weight)))
        concat_loader = MixLoader([vl_loader, lain_vl_loader, lm_loader, 
                                   instru_lm_loader, vl_instru_loader, lain_vl_refer_loader], data_weight)
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

        loss_name = None
        loss_key = None
        if 'vl' in sample:
            loss_name = "image_wild"
            loss_key = 'vl'
        elif 'vl_laion' in sample:
            loss_name = "image_laion"
            loss_key = 'vl_laion'
        elif 'gpt' in sample:
            loss_name = "gpt"
            loss_key = 'gpt'
        elif 'gpt_tune' in sample:
            loss_name = "gpt_tune"
            loss_key = 'gpt_tune'
        elif 'vl_tune' in sample:
            loss_name = "image_tune"
            loss_key = 'vl_tune'
        else:
            assert False, "Unknown loss key"

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                vl_loss, sample_size, logging_output = criterion(model, sample[loss_key], loss_name=loss_name)
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

        return agg_loss, agg_sample_size, agg_logging_output

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

