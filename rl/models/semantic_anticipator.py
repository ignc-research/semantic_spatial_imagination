#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

from rl.models.unet import (
    UNetEncoder,
    UNetDecoder,
    MiniUNetEncoder,
    LearnedRGBProjection,
    MergeMultimodal,
    ResNetRGBEncoder,
)

from einops import rearrange

def softmax_2d(x):
    b, h, w = x.shape
    x_out = F.softmax(rearrange(x, "b h w -> b (h w)"), dim=1)
    x_out = rearrange(x_out, "b (h w) -> b h w", h=h)
    return x_out

def padded_resize(x, size):
    """For an image tensor of size (bs, c, h, w), resize it such that the
    larger dimension (h or w) is scaled to `size` and the other dimension is
    zero-padded on both sides to get `size`.
    """
    h, w = x.shape[2:]
    top_pad = 0
    bot_pad = 0
    left_pad = 0
    right_pad = 0
    if h > w:
        left_pad = (h - w) // 2
        right_pad = (h - w) - left_pad
    elif w > h:
        top_pad = (w - h) // 2
        bot_pad = (w - h) - top_pad
    x = F.pad(x, (left_pad, right_pad, top_pad, bot_pad))
    x = F.interpolate(x, size, mode="bilinear", align_corners=False)
    return x

# ================================ Anticipation base ==================================
class BaseModel(nn.Module):
    """The basic model for semantic anticipator
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "sigmoid":
            self.normalize_channel_0 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "softmax":
            self.normalize_channel_0 = softmax_2d

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "sigmoid":
            self.normalize_channel_1 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "softmax":
            self.normalize_channel_1 = softmax_2d

        self._create_gp_models()

    def forward(self, x):
        final_outputs = {}
        gp_outputs = self._do_gp_anticipation(x)
        final_outputs.update(gp_outputs)

        return final_outputs

    def _create_gp_models(self):
        raise NotImplementedError

    def _do_gp_anticipation(self, x):
        raise NotImplementedError

    def _normalize_decoder_output(self, x_dec):
        x_dec_c0 = self.normalize_channel_0(x_dec[:, 0])
        # x_dec_c1 = self.normalize_channel_1(x_dec[:, 1])
        return torch.stack([x_dec_c0], dim=1)

# SemAnt Model
class SemAntRGBD(BaseModel):
    """
    Anticipated using rgb and depth projection.
    """
    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_encoder = UNetEncoder(2, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth encoder branch
        self.gp_depth_proj_encoder = unet_encoder

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = unet_decoder

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, infeats, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, infeats, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        x_depth_proj_enc = self.gp_depth_proj_encoder(
            x["ego_map_gt"]
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)  # (bs, 2, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"sem_estimate": x_dec}

        return outputs

class SemAntGroundTruth(BaseModel):
    """
    Outputs the GT anticipated occupancy
    """

    def _create_gp_models(self):
        pass

    def _do_gp_anticipation(self, x):
        x_dec = x["ego_map_gt_anticipated"]  # (bs, 2, H, W)
        outputs = {"sem_estimate": x_dec}

        return outputs

class SemAnticipator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        model_type = cfg.type
        self._model_type = model_type
        cfg.defrost()
        if model_type == "sem_rgbd":
            self.main = SemAntRGBD(cfg)
        elif model_type == "occant_ground_truth":
            self.main = SemAntGroundTruth(cfg)
        else:
            raise ValueError(f"Invalid model_type {model_type}")
        cfg.freeze()

    def forward(self, x):
        return self.main(x)

    @property
    def use_gp_anticipation(self):
        return self.main.use_gp_anticipation

    @property
    def model_type(self):
        return self._model_type

class SemAnticipationWrapper(nn.Module):

    def __init__(self, model, V, input_hw):
        """[summary]

        Args:
            model (SemAnticipator): the model used in the Wrapper
            V (int): Outoput dismension will be (V,V)
            input_hw ([h,w]): input height and width
        """
        super().__init__()
        self.main = model
        self.V = V
        self.input_hw = input_hw
        self.keys_to_interpolate = [
            "ego_map_hat",
            "sem_estimate",
            "depth_proj_estimate",  # specific to RGB Model V2
        ]

    def forward(self, x):
        x["rgb"] = padded_resize(x["rgb"], self.input_hw[0])
        if "ego_map_gt" in x:
            x["ego_map_gt"] = F.interpolate(x["ego_map_gt"], size=self.input_hw)
        x_full = self.main(x)
        for k in x_full.keys():
            if k in self.keys_to_interpolate:
                x_full[k] = F.interpolate(
                    x_full[k], size=(self.V, self.V), mode="bilinear"
                )
        return x_full