#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import time
from einops.einops import asnumpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import string
from occant_utils.map import (
    subtract_pose,
)

from mapper.mapper import Mapper

class MapperDataParallelWrapper(Mapper):
    def forward(self, *args, **kwargs):
        outputs = self.predict_deltas(*args, **kwargs)
        return outputs

    def _estimate_ego_map(self, x):
        return self.projection_unit(x)

from einops import rearrange


def simple_mapping_loss_fn(pt_hat, pt_gt, weight):
    occupied_hat = pt_hat[:, 0]  # (T*N, V, V)
    occupied_gt = pt_gt[:, 0]  # (T*N, V, V)
    occupied_weight = weight[:, 0]
    occupied_mapping_loss = F.binary_cross_entropy(occupied_hat, occupied_gt,weight=occupied_weight)
    
    mapping_loss = occupied_mapping_loss

    return mapping_loss


def pose_loss_fn(pose_hat, pose_gt):
    trans_loss = F.smooth_l1_loss(pose_hat[:, :2], pose_gt[:, :2])
    rot_loss = F.smooth_l1_loss(pose_hat[:, 2], pose_gt[:, 2])
    pose_loss = 0.5 * (trans_loss + rot_loss)

    return pose_loss, trans_loss, rot_loss


class MapUpdateBase(nn.Module):
    def __init__(
        self,
        mapper,
        label_id="ego_map_gt_anticipated",
        lr=None,
        eps=None,
        max_grad_norm=None,
        pose_loss_coef=2.0,
        semantic_anticipator_type="sem_rgbd",
        freeze_projection_unit=False,
        bias_factor=10.0,
    ):
        super().__init__()

        self.mapper = mapper

        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, mapper.parameters()), lr=lr, eps=eps,
        )
        mapper_cfg = self.mapper.config
        if mapper_cfg.use_data_parallel and len(mapper_cfg.gpu_ids) > 0:
            self.device = mapper_cfg.gpu_ids[0]
        else:
            self.device = next(mapper.parameters()).device
        self.pose_loss_coef = pose_loss_coef
        self.freeze_projection_unit = freeze_projection_unit
        self.semantic_anticipator_type = semantic_anticipator_type
        self.bias_factor = bias_factor
        self.label_id = label_id

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts):
        raise NotImplementedError

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

    def after_step(self):
        pass

    def load_state_dict(self, loaded_state_dict):
        """Intelligent state dict assignment. Load state-dict only for keys
        that are available and have matching parameter sizes.
        """
        src_state_dict = self.state_dict()
        matching_state_dict = {}
        offending_keys = []
        for k, v in loaded_state_dict.items():
            if k in src_state_dict.keys() and v.shape == src_state_dict[k].shape:
                matching_state_dict[k] = v
            else:
                offending_keys.append(k)
        src_state_dict.update(matching_state_dict)
        super().load_state_dict(src_state_dict)
        if len(offending_keys) > 0:
            print("=======> MapUpdate: list of offending keys in load_state_dict")
            for k in offending_keys:
                print(k)


def map_update_fn(ps_args):
    # Unpack args
    mapper = ps_args[0]
    mapper_rollouts = ps_args[1]
    optimizer = ps_args[2]
    num_update_batches = ps_args[3]
    batch_size = ps_args[4]
    freeze_projection_unit = ps_args[5]
    bias_factor = ps_args[6]
    semantic_anticipator_type = ps_args[7]
    pose_loss_coef = ps_args[8]
    max_grad_norm = ps_args[9]
    label_id = ps_args[10]

    # Perform update
    losses = {
        "total_loss": 0,
        "mapping_loss": 0,
        "trans_loss": 0,
        "rot_loss": 0,
    }

    if isinstance(mapper, nn.DataParallel):
        mapper_config = mapper.module.config
    else:
        mapper_config = mapper.config

    img_mean = mapper_config.NORMALIZATION.img_mean
    img_std = mapper_config.NORMALIZATION.img_std
    start_time = time.time()
    # Debugging
    map_update_profile = {"data_sampling": 0.0, "pytorch_update": 0.0}
    for i in range(num_update_batches):
        start_time_sample = time.time()
        observations = mapper_rollouts.sample(batch_size)
        map_update_profile["data_sampling"] += time.time() - start_time_sample
        # Labels
        # Pose labels
        start_time_pyt = time.time()

        # Map labels
        pt_gt = observations[f"{label_id}_at_t"]  # (bs, V, V, 2)
        pt_gt = rearrange(pt_gt, "b h w c -> b c h w")  # (bs, 2, V, V)

        # Forward pass
        mapper_inputs = observations
        mapper_outputs = mapper(mapper_inputs)
        pt_hat = mapper_outputs["pt"]

        # Compute losses
        # -------- mapping loss ---------
        pt_gt_np = asnumpy(pt_gt)
        ego_map_gt_dilation_at_t_np = asnumpy(observations["ego_map_gt_dilation_at_t"])
        alpha = (pt_gt_np > -1).sum()/ (pt_gt_np > 0).sum()
        alpha = min(alpha,30)
        gama = (ego_map_gt_dilation_at_t_np > -1).sum()/ (ego_map_gt_dilation_at_t_np > 0).sum()
        gama = min(gama,10)
        weight = ((alpha - 1)*(pt_gt > 0) + (gama - 1)*(rearrange(observations["ego_map_gt_dilation_at_t"], "b h w c -> b c h w")> 0))  + 1

        mapping_loss = simple_mapping_loss_fn(pt_hat, pt_gt, weight)

        if freeze_projection_unit:
            mapping_loss = mapping_loss.detach()

        optimizer.zero_grad()
        total_loss = mapping_loss 

        # Backward pass
        total_loss.backward()

        # Update
        nn.utils.clip_grad_norm_(mapper.parameters(), max_grad_norm)
        optimizer.step()

        losses["total_loss"] += total_loss.item()
        losses["mapping_loss"] += mapping_loss.item()

        map_update_profile["pytorch_update"] += time.time() - start_time_pyt
        time_per_step = (time.time() - start_time) / (60 * (i + 1))

    for k in losses.keys():
        losses[k] /= num_update_batches

    return losses


def map_update_worker(
    remote, parent_remote, ps_args, update_completed,
):

    # Unpack args
    mapper = ps_args[0]
    mapper_rollouts = ps_args[1]
    optimizer = ps_args[2]
    num_update_batches = ps_args[3]
    batch_size = ps_args[4]
    freeze_projection_unit = ps_args[5]
    bias_factor = ps_args[6]
    semantic_anticipator_type = ps_args[7]
    pose_loss_coef = ps_args[8]
    max_grad_norm = ps_args[9]
    label_id = ps_args[10]

    # Close parent remote
    parent_remote.close()

    while True:
        cmd, data = remote.recv()
        if cmd == "update":
            # Unset update completed
            update_completed.clear()
            # Ensure that rollouts is filled
            if mapper_rollouts.step > batch_size or mapper_rollouts.memory_filled:
                pass
            else:
                update_completed.set()
                remote.send({})
                continue
            # Perform update
            losses = map_update_fn(ps_args)
            remote.send(losses)
            # Set update completed event
            update_completed.set()
        elif cmd == "close":
            remote.close()
            break


class MapUpdate(MapUpdateBase):
    def __init__(
        self,
        mapper,
        label_id="ego_map_gt_anticipated",
        lr=None,
        eps=None,
        max_grad_norm=None,
        pose_loss_coef=2.0,
        semantic_anticipator_type="sem_rgbd",
        freeze_projection_unit=False,
        num_update_batches=1,
        batch_size=32,
        mapper_rollouts=None,
    ):

        super().__init__(
            mapper,
            label_id=label_id,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            pose_loss_coef=pose_loss_coef,
            semantic_anticipator_type=semantic_anticipator_type,
            freeze_projection_unit=freeze_projection_unit,
        )

        self.num_update_batches = num_update_batches
        self.batch_size = batch_size
        mapper_cfg = self.mapper.config
        # ========================== Create update worker =====================
        # This is necessary for CUDA tensors and models.
        self.mp_ctx = mp.get_context("forkserver")
        self.update_worker_dict = {}
        remote, work_remote = self.mp_ctx.Pipe()
        # Create an event to signal update completion
        update_completed = self.mp_ctx.Event()
        # Initially set this to True. The process will unset this
        # event when it receives the update signal, and set it back when
        # the update was completed.
        update_completed.set()
        # Make a copy of the mapper and make it shared.
        self.mapper_copy = MapperDataParallelWrapper(
            mapper_cfg, copy.deepcopy(self.mapper.projection_unit),
        )
        self.mapper_copy.load_state_dict(self.mapper.state_dict())
        if mapper_cfg.use_data_parallel and len(mapper_cfg.gpu_ids) > 0:
            self.mapper_copy.to(self.mapper.config.gpu_ids[0])
            self.mapper_copy = nn.DataParallel(
                self.mapper_copy,
                device_ids=self.mapper.config.gpu_ids,
                output_device=self.mapper.config.gpu_ids[0],
            )
        else:
            self.mapper_copy.to(next(self.mapper.parameters()).device)

        self.mapper_copy.share_memory()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.mapper_copy.parameters()),
            lr=lr,
            eps=eps,
        )

        # Create worker arguments --- very dirty way
        ps_args = (
            self.mapper_copy,
            mapper_rollouts,
            self.optimizer,
            self.num_update_batches,
            self.batch_size,
            self.freeze_projection_unit,
            self.bias_factor,
            self.semantic_anticipator_type,
            self.pose_loss_coef,
            self.max_grad_norm,
            self.label_id,
        )

        ps = self.mp_ctx.Process(
            target=map_update_worker,
            args=(work_remote, remote, ps_args, update_completed),
        )
        ps.daemon = True
        ps.start()
        work_remote.close()
        self.update_worker_dict["remote"] = remote
        self.update_worker_dict["work_remote"] = work_remote
        self.update_worker_dict["process"] = ps
        self.update_worker_dict["update_completed"] = update_completed
        # Keep track of when the first update is sent
        self._first_update_sent = False

    def update(self, rollouts):
        # Wait for the previous update to complete
        self.update_worker_dict["update_completed"].wait()
        # Get losses from previous update
        if self._first_update_sent:
            losses = self.update_worker_dict["remote"].recv()
        else:
            losses = {}
        # Copy the state dict from mapper_copy to mapper
        if self.mapper.config.use_data_parallel:
            self.mapper.load_state_dict(self.mapper_copy.module.state_dict())
        else:
            self.mapper.load_state_dict(self.mapper_copy.state_dict())
        # Start the next update
        self.update_worker_dict["remote"].send(("update", None))
        self._first_update_sent = True
        return losses
