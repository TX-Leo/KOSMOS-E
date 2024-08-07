# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d

import pdb

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        return_hidden_only: bool = False,
        hidden_feature_type: str = 'masked',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        if return_hidden_only and hidden_feature_type == 'mask':
            return self.get_instance_feature(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )
        elif return_hidden_only and hidden_feature_type == 'hidden':
            return self.get_hidden_states(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )
        else:
            masks, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # pdb.set_trace()
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    
    def get_instance_feature(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # pdb.set_trace()
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # NOTE:
        #    In the original version, the default batchsize of image embedding is set as 1, then the image embedding 
        #    is expand to align with sparse_prompt_embeddings, which batchsize is not 1 as usual (same as grid points).
        #    So, it may induce huge GPU cost when image batchsize is not 1 and we align the points size and image batchsize 
        #    at the same time. 
        #    For the fast adaptation, we iteratively resample the image embedding
        collect_resample_feature = []
        for image_embedding in image_embeddings:
            # pdb.set_trace()
            # Expand per-image data in batch direction to be per-mask
            src = torch.repeat_interleave(image_embedding[None], tokens.shape[0], dim=0)
            src = src + dense_prompt_embeddings
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
            b, c, h, w = src.shape

            # Run the transformer
            hs, src = self.transformer(src, pos_src, tokens)      
            mask_tokens_out = hs[:, 1, :]
            src = src.transpose(1, 2).view(b, c, h, w) # [64, 256, 64, 64]
            upscaled_embedding = self.output_upscaling(src) # [64, 32, 256, 256]
            hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out).unsqueeze(1)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # [64, 1, 256, 256]
            
            masks = F.interpolate(
                masks,
                (image_embedding.shape[-2], image_embedding.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ) # [64, 1, 64, 64]
            
            # pdb.set_trace()
            masks = (masks > 0.).type_as(image_embedding) # [64, 1, 64, 64]
                        
            masked_image_embedding = (image_embedding * masks).sum(dim=[-2, -1])
            norm_masked_image_embedding = masked_image_embedding / masks.sum(dim=[-2, -1]).clamp(1)
            collect_resample_feature.append(norm_masked_image_embedding) 
        collect_resample_feature = torch.stack(collect_resample_feature, dim=0)
        
        # pdb.set_trace()
        return collect_resample_feature
    
    def get_hidden_states(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # pdb.set_trace()
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # NOTE:
        #    In the original version, the default batchsize of image embedding is set as 1, then the image embedding 
        #    is expand to align with sparse_prompt_embeddings, which batchsize is not 1 as usual (same as grid points).
        #    So, it may induce huge GPU cost when image batchsize is not 1 and we align the points size and image batchsize 
        #    at the same time. 
        #    For the fast adaptation, we iteratively resample the image embedding
        collect_resample_feature = []
        for image_embedding in image_embeddings:
            # pdb.set_trace()
            # Expand per-image data in batch direction to be per-mask
            src = torch.repeat_interleave(image_embedding[None], tokens.shape[0], dim=0)
            src = src + dense_prompt_embeddings
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
            b, c, h, w = src.shape

            # Run the transformer
            hs, src = self.transformer(src, pos_src, tokens, ignore_final_attn=True)
            # iou_token_out = hs[:, 0, :]
            # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
            
            collect_resample_feature.append(hs[:, 1, :])
        collect_resample_feature = torch.stack(collect_resample_feature, dim=0)
        return collect_resample_feature


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
