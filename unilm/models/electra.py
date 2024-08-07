import math
import logging
import copy
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel, FairseqIncrementalDecoder, register_model, register_model_architecture
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttention,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from torch import Tensor
from fairseq.utils import safe_getattr, safe_hasattr
from omegaconf import II
from unilm.models.squad import SQuADHead
from unilm.models.unilm import LMHead, ClassificationHead, UniLMBody, UniLMModelConfig, UniLMModel

DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)

@dataclass
class ElectraModelConfig(UniLMModelConfig):
    generator_encoder_layers: int = field(default=6, metadata={"help": "num generator's encoder layers"})

@register_model("electra", dataclass=ElectraModelConfig)
class ElectraModel(UniLMModel):

    def __init__(self, args, discriminator, discriminator_lm_head, generator, generator_lm_head):
        super(UniLMModel, self).__init__()
        self.args = args
        self.discriminator = discriminator
        self.discriminator_lm_head = discriminator_lm_head
        self.generator = generator
        self.generator_lm_head = generator_lm_head
        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        args.max_target_positions = safe_getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        embed_tokens = cls.build_embedding(
            args, 
            task.source_dictionary, 
            args.encoder_input_dim
        )

        discriminator = UniLMBody(
            args, 
            task.source_dictionary, 
            embed_tokens
        )

        discriminator_lm_head = cls.build_lm_head(
            args, 
            args.encoder_embed_dim, 
            2, 
            args.activation_fn, 
            weight=None
        )

        if task.cfg._name == 'pretraining':
            generator_args = copy.copy(args)
            generator_args.encoder_layers = args.generator_encoder_layers
            generator_args.dropout = 0.0
            generator_args.attention_dropout = 0.0
            generator_args.activation_dropout = 0.0
            generator_args.pooler_dropout = 0.0
            generator_args.task_moe = False

            generator = UniLMBody(
                generator_args, 
                task.source_dictionary, 
                embed_tokens
            )

            generator_lm_head = cls.build_lm_head(
                generator_args, 
                generator_args.encoder_embed_dim, 
                len(task.dictionary), 
                generator_args.activation_fn, 
                weight=embed_tokens.weight
            )
        else:
            generator, generator_lm_head = None, None

        return cls(args, discriminator, discriminator_lm_head, generator, generator_lm_head)
    
    def output_layer(self, features):
        return self.generator_lm_head(features)
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if self.generator is None:
            keys_to_delete = []
            for k in state_dict.keys():
                if k.startswith('generator'):
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del state_dict[k]

    def forward(self, src_tokens=None, tgt_tokens=None, incremental_state=None, classification_head_name=None, masked_tokens=None, features_only=False, **kwargs):
        if tgt_tokens is not None or features_only:
            x, extra = self.discriminator(src_tokens, tgt_tokens, incremental_state, return_all_hiddens=True)
            return x, extra

        if classification_head_name is not None:
            x, extra = self.discriminator(src_tokens, None, incremental_state, return_all_hiddens=True)
            x = self.classification_heads[classification_head_name](x)
            return x, extra
        
        generator_x, generator_extra = self.generator(src_tokens, None, incremental_state, return_all_hiddens=True)
        generator_logits = self.generator_lm_head(generator_x, masked_tokens=masked_tokens)

        with torch.no_grad():
            sampled_probs = torch.softmax(generator_logits.view(-1, generator_logits.size(-1)), -1, dtype=torch.float32)
            sampled_tokens = torch.multinomial(sampled_probs, 1).view(-1)
            input_tokens = src_tokens.clone()
            input_tokens[masked_tokens] = sampled_tokens
        
        discriminator_x, discriminator_extra = self.discriminator(input_tokens, None, incremental_state, return_all_hiddens=True)
        discriminator_logits = self.discriminator_lm_head(discriminator_x)

        return input_tokens, generator_logits, discriminator_logits, generator_extra, discriminator_extra


@register_model_architecture("electra", "electra_base")
def base_unilm_architecture(args):
    if safe_hasattr(args, "encoder_final_norm"):
        args.no_encoder_final_norm = not args.encoder_final_norm

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")

    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # args.add_bos_token = safe_getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_encoder_input_output_embed = safe_getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.encoder_output_dim = safe_getattr(
        args, "encoder_output_dim", args.encoder_embed_dim
    )
    args.encoder_input_dim = safe_getattr(args, "encoder_input_dim", args.encoder_embed_dim)

    # Model training is not stable without this
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', False)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", True)

    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.generator_encoder_layers = safe_getattr(args, "generator_encoder_layers", 4)