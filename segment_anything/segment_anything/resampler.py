# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import math
import os
import torchvision.transforms as T
import torch.nn.functional as F

from segment_anything.modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide
from .utils.amg import build_point_grid

import pdb

class SamResampler(nn.Module):
    reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
    def __init__(
        self,
        sam_model: Sam,
        sample_mode: str = 'grid',
        sample_grid_points: int = 256, 
        sample_final_keep: int = 0,
        sample_final_nms_thr: float = 0.,
        feature_type: str = 'mask',
        norm_output: bool = False,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        
        self.model_input_size = sam_model.image_encoder.img_size
        assert sample_mode in ['grid', 'random']
        self.sample_mode = sample_mode
        
        if sample_mode == 'grid':
            grid_points_each_side = int(math.sqrt(sample_grid_points))
            grid_points = build_point_grid(grid_points_each_side)
            grid_points *= sam_model.image_encoder.img_size
        else:
            grid_points = torch.rand((sample_grid_points, 2))
            
        self.grid_points = torch.as_tensor(grid_points, dtype=torch.float)[:, None, :]
        self.grid_points_labels = torch.ones(grid_points.shape[0], dtype=torch.int)[:, None]
        
        # not implement now
        self.random_points = None
        self.random_points_labels = None
        
        self.sample_final_keep = sample_final_keep if sample_final_keep else sample_grid_points
        self.sample_final_nms_thr = sample_final_nms_thr
        
        self.feature_type = feature_type
        self.norm_output = norm_output
        
        self.reset_image()

    def set_image(
        self,
        image: torch.Tensor,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """ 
        # revert it to numpy.array, then re-norm it, then using the imagenet - mean/var to normalize it
        input_image_reverse = image.to(self.device) * self.reverse_norm_std.to(self.device) + self.reverse_norm_mean.to(self.device)
        input_image_reverse = input_image_reverse.clamp(0, 1).type_as(image)
        # assert input_image_reverse.min() >= 0.
        # assert input_image_reverse.max() <= 1.
        
        # visualize the image to check
        # pdb.set_trace() 
        # to_pil = T.ToPILImage()
        # image = to_pil(input_image_reverse[0])
        # image.save(os.path.join('output/debug', "test.jpg"))
    
        input_image_torch = F.interpolate(input_image_reverse, 
                                          size=(self.model_input_size, self.model_input_size), 
                                          mode="bilinear")
        input_image_torch = (input_image_torch * 255.).clamp(0, 255)
        self.set_torch_image(input_image_torch, image.shape[:2])

    # @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def forward(self, image) -> torch.Tensor:
        # pdb.set_trace()
        self.set_image(image)
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        
        # pdb.set_trace()
        if self.sample_mode == 'grid':
            resampler_feature = self.sample_torch(
                self.grid_points.to(self.device),
                self.grid_points_labels.to(self.device),
                None,
                None,
                False,
            )
            return resampler_feature
          
        if self.sample_mode == 'random':
            random_points = torch.rand_like(self.grid_points).clamp(0.005, 0.995) * self.model_input_size
            resampler_feature = self.sample_torch(
                  random_points.to(self.device),
                  self.grid_points_labels.to(self.device),
                  None,
                  None,
                  False,
              )
            return resampler_feature
        
    # @torch.no_grad()
    def sample_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> torch.Tensor:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # pdb.set_trace()
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # pdb.set_trace()
        # Sampler feaure based on the points
        resampler_feature = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            return_hidden_only=True,
            hidden_feature_type=self.feature_type,
        )
        # pdb.set_trace()
        if self.norm_output:
            # process the zero-norm
            # return F.normalize(resampler_feature.transpose(0, 1), dim=-1)
            eps = 1e-8
            norm = torch.norm(resampler_feature, dim=-1, keepdim=True)
            norm = torch.max(norm, eps * torch.ones_like(norm))
            resampler_feature = resampler_feature / norm
            return F.normalize(resampler_feature.transpose(0, 1), dim=-1)
        else:
            return resampler_feature.transpose(0, 1) # [token, bsz, dim]

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
