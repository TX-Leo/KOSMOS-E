from dataclasses import dataclass, field
from typing import Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

import logging
import numpy as np
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.utils import safe_getattr, safe_hasattr

from fairseq.modules import LayerNorm
from fairseq.models import (
  BaseFairseqModel,
  register_model,
  register_model_architecture,
)
from fairseq.models.roberta import (
    roberta_large_architecture,
    roberta_base_architecture,
    RobertaEncoder,
    RobertaModel,
)
from fairseq.models.transformer_lm import (
  TransformerLanguageModelConfig,
  TransformerLanguageModel,
  base_gpt3_architecture,
)
from unilm.models.gpt import GPTmodel, GPTModelConfig
from unilm.models.unilm import LMHead, ClassificationHead, UniLMBody, UniLMModelConfig, UniLMModel
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
)

DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


@dataclass
class ELEGPTModelConfig(GPTModelConfig):

    no_connector: bool = field(
        default=False,
        metadata={
            "help": "whether to add connector"
        },
    )
    connector_type: str = field(
        default='simple',
        metadata={
            "help": "which type connector to use, simple or complex"
        },
    )
    remain_tokens: int = field(
        default=300, 
        metadata={
            "help": "at least k tokens to produce gpt loss"
        },
    )
    ele_model_path: str = field(
        default="",
        metadata={"help": "mlm checkpoint path"},
    )
    mlm_dict: str = field(
        default="",
        metadata={"help": "mlm dict path"},
    )
    mlm_tokens_per_sample: int = field(
        default=512,
        metadata={"help": "mlm max length"},
    )
    freeze_gpt: bool = field(
        default=False,
        metadata={
            "help": "freeze gpt parameters"
        },
    )
    update_last_layer: bool = field(
        default=False,
        metadata={
            "help": "freeze gpt parameters"
        },
    )
    update_last_two_layers: bool = field(
        default=False,
        metadata={
            "help": "update the last two lyaers of encoder, update_last_layer and update_last_two_layers cannot be True at the same time"
        },
    )
    rand_last_layer: bool = field(
        default=False,
        metadata={
            "help": "freeze gpt parameters"
        },
    )

    ft_type: int = field(
        default=3,
        metadata={
            "help": "fine-tuning type: \
            1: gpt only \
            2: roberta only \
            3: roberta + gpt \
            4: roberta + gpt(freeze) \
            5: roberta(freeze) + gpt "
        },
    )
    rel_pos_buckets: int = field(
        default=0, metadata={"help": ""}
    )
    max_rel_pos: int = field(
        default=0, metadata={"help": ""}
    )
    pooler_dropout: float = field(
        default=0.1,
        metadata={"help": "mlm max length"},
    )

    pretrained_ckpt_path: str = field(
        default="",
        metadata={"help": "model checkpoint path"},
    )
    deepnorm_encoder: bool = field(
        default=False,
        metadata={
            "help": "whether to use deepnorm for encoder"
        },
    )


@register_model("elegptmodel", dataclass=ELEGPTModelConfig)
class ELEGPTmodel(BaseFairseqModel):

    def __init__(self, args, gpt_model, mlm_model, connector=None, bos=0, eos=2):
        super().__init__()
        self.args = args
        self.gpt_model = gpt_model
        self.mlm_model = mlm_model
        self.connector = connector
        self.bos = bos
        self.eos = eos
        self.classification_heads = nn.ModuleDict()
        self.ft_type = args.ft_type

        if args.freeze_gpt:
            for p in self.gpt_model.parameters():
                p.requires_grad = False

    @classmethod
    def build_model(cls, args, task):
        if hasattr(task, "all_dict"):
            task.dictionary = task.all_dict
        gpt_model = GPTmodel.build_model(args, task)
        logger.info("gpt args is {}".format(args))

        mlm_args = copy.deepcopy(args)
        mlm_task = task
        mlm_task.dictionary.add_symbol("<mask>")
        for i in range(100):
            mlm_task.dictionary.add_symbol(f"<mask_{i}>")
        mlm_task.dictionary.pad_to_multiple_(8)
        logger.info("Electra dictionary: {} types".format(len(mlm_task.dictionary)))

        mlm_args.layernorm_embedding = True
        mlm_args.no_scale_embedding = True
        mlm_args.dropout = 0.1
        mlm_args.attention_dropout = 0.1
        mlm_args.tokens_per_sample = mlm_args.mlm_tokens_per_sample
        mlm_args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        logger.info("mlm args is {}".format(mlm_args))
        mlm_model = UniLMBody(mlm_args, task.source_dictionary, embed_tokens = Embedding(len(mlm_task.dictionary), mlm_args.encoder_input_dim, mlm_task.dictionary.pad()))

        if args.ele_model_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.ele_model_path)
            tmp_state = state["model"]
            keys_all = list(tmp_state.keys())
            for key in keys_all:
                if "discriminator." in key:
                    if "11" in key and args.rand_last_layer: # NOTE hard codes
                        assert False, "hard codes"
                        tmp_state.pop(key)
                    else:
                        tmp_state[key[len(key.split('.')[0]+'.'):]] = tmp_state[key]
                        tmp_state.pop(key)
                else:
                    tmp_state.pop(key)
            mlm_model.load_state_dict(tmp_state, strict=False)

        if args.no_connector:
            connector = None
        elif args.connector_type == "simple":
            connector = SimpleConnector(mlm_args.encoder_embed_dim, 
                                        args.decoder_embed_dim,)
        elif args.connector_type == "complex":
            connector = ComplexConnector(mlm_args.encoder_embed_dim, 
                                        args.decoder_embed_dim,
                                        args.activation_fn)

        model = cls(args, gpt_model, mlm_model, connector, task.dictionary.bos_index, task.dictionary.eos_index)

        if args.pretrained_ckpt_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_ckpt_path)
            model.load_state_dict(state["model"], strict=True, args=args)

        for p in model.mlm_model.parameters():
            p.requires_grad = False

        assert (not args.update_last_layer) or (not args.update_last_two_layers), "update_last_layer and update_last_two_layers cannot be True at the same time"
        
        if args.update_last_layer:
            for n, p in model.mlm_model.named_parameters():
                if f"layers.{mlm_args.encoder_layers-1}" in n:
                    p.requires_grad = True

        if args.update_last_two_layers:
            for n, p in model.mlm_model.named_parameters():
                if f"layers.{mlm_args.encoder_layers-2}" in n or f"layers.{mlm_args.encoder_layers-1}" in n:
                    p.requires_grad = True
            
        return model

    def forward(self, src_tokens, mlm_src_tokens=None, gpt_input_mask=None, gpt_loss_mask=None, mlm_mask=None, classification_head_name=None, **kwargs):

        if classification_head_name is None:
            # pre-training

            if mlm_src_tokens is not None:
                # mlm
                mlm_output, _ = self.mlm_model(mlm_src_tokens)
                mlm_output = mlm_output[mlm_mask]

                # linear projection layer
                mlm_output = self.connector(mlm_output)
            else:
                mlm_output = None

            # gpt 
            x, extra = self.gpt_model(src_tokens, mlm_features=mlm_output, gpt_input_mask=gpt_input_mask)

            # loss mask
            extra["loss_mask"] = gpt_loss_mask
            return x, extra

        mlm_mask = torch.ones_like(mlm_src_tokens) * (mlm_src_tokens!=1)
        mlm_mask[:,0:1] = 0
        mlm_mask = mlm_mask.bool().cuda()

        gpt_input_mask = torch.zeros_like(src_tokens).bool().cuda()
        gpt_input_mask[:,:len(mlm_mask[0])] = mlm_mask

        # fine-tuning
        if self.ft_type == 1:
            x, extra = self.gpt_model(src_tokens, mlm_features=None)
        elif self.ft_type == 2:
            x, extra = self.mlm_model(mlm_src_tokens, features_only=True)
            x = self.classification_heads[classification_head_name](x)
        else:
            mlm_output, _ = self.mlm_model(mlm_src_tokens, features_only=True)
            mlm_output = mlm_output[mlm_mask]
            mlm_output = self.connector(mlm_output)

            x, extra = self.gpt_model(src_tokens, mlm_features=mlm_output, gpt_input_mask=gpt_input_mask)
        
        return x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        self.classification_heads[name] = ClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args.ft_type
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        ft_type
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.ft_type = ft_type

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SimpleConnector(nn.Module):
    """Connector model of GPT and MLM."""

    def __init__(self, intput_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(intput_dim, output_dim)

    def forward(self, features, **kwargs):
        
        x = self.dense(features)
        return x


class ComplexConnector(nn.Module):
    """Connector model of GPT and MLM."""

    def __init__(self, input_dim, output_dim, activation_fn):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.predict = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):

        x = self.dense(features)
        x = self.activation_fn(x)

        x = self.predict(x)
        return x


@register_model_architecture("elegptmodel", "elegptmodel_small")
def gptmodel_small(args):
    # 125M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    base_gpt3_architecture(args)
    base_unilm_architecture(args)

@register_model_architecture("elegptmodel", "elegptmodel_medium")
def gptmodel_medium(args):
    # 355M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.1)
    base_gpt3_architecture(args)
    base_unilm_architecture(args)

@register_model_architecture("elegptmodel", "elegptmodel_large")
def gptmodel_large(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)

@register_model_architecture("elegptmodel", "elegptmodel_huge")
def gptmodel_medium(args):
    # 1.4B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 48)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 24)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.1)

    # encoder use large ele
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    args.generator_encoder_layers = safe_getattr(args, "generator_encoder_layers", 6)
    base_gpt3_architecture(args)
    base_unilm_architecture(args)

@register_model_architecture("elegptmodel", "elegptmodel_xl")
def gptmodel_medium(args):
    # 1.4B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.1)

    # encoder use large ele
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    args.generator_encoder_layers = safe_getattr(args, "generator_encoder_layers", 6)
    base_gpt3_architecture(args)
    base_unilm_architecture(args)

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

