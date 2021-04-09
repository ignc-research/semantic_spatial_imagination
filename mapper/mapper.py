import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
from gym import spaces
from einops import rearrange
from occant_utils.map import (
    add_pose,
    subtract_pose,
    process_image,
    transpose_image,
    bottom_row_padding,
    bottom_row_cropping,
    spatial_transform_map, 
)
EPS_MAPPER = 1e-8
from semantic_utils.common import safe_mkdir


class Mapper(nn.Module):
    def __init__(self, config, projection_unit):
        super().__init__()
        self.config = config
        self.map_config = {"size": config.map_size, "scale": config.map_scale}
        V = self.map_config["size"]
        s = self.map_config["scale"]
        
        self.img_mean_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_mean), "c -> () c () ()"
        )
        self.img_std_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_std), "c -> () c () ()"
        )

        self.projection_unit = projection_unit
        if self.config.freeze_projection_unit:
            for p in self.projection_unit.parameters():
                p.requires_grad = False

        # Cache to store pre-computed information
        self._cache = {}

    def forward(self, x, masks=None):
        outputs = self.predict_deltas(x, masks=masks)
        mt_1 = x["map_at_t_1"]
        if masks is not None:
            mt_1 = mt_1 * masks.view(-1, 1, 1, 1)
        with torch.no_grad():
            mt = self._register_map(mt_1, outputs["pt"], outputs["xt_hat"])
        outputs["mt"] = mt

        return outputs

    def predict_deltas(self, x, masks=None):
        # Transpose multichannel inputs

        st = process_image(x["rgb_at_t"], self.img_mean_t, self.img_std_t)
        dt = transpose_image(x["depth_at_t"])
        ego_map_gt_at_t = transpose_image(x["ego_map_gt_at_t"])

        pu_inputs_t = {
            "rgb": st,
            "depth": dt,
            "ego_map_gt": ego_map_gt_at_t,
        }

        pu_outputs = self.projection_unit(pu_inputs_t)
        pu_outputs_t = {k: v[:] for k, v in pu_outputs.items()}
        pt = pu_outputs["sem_estimate"]

        #debug 
        #safe_mkdir('data/debug/data')
        #randomID = ''.join(random.choices(string.#ascii_uppercase + string.digits, k=20))
        
        # torch.save(x, 'data/debug/data/x_t_'+randomID+'.pt')
        # torch.save(pt, 'data/debug/data/pu_output_t_'+randomID+'.pt')

        all_pose_outputs = None

        outputs = {
            "pt": pt,
            "all_pu_outputs": pu_outputs_t,
            "all_pose_outputs": all_pose_outputs,
        }
        if "ego_map_hat" in pu_outputs_t:
            outputs["ego_map_hat_at_t"] = pu_outputs_t["ego_map_hat"]
        return outputs

    def _bottom_row_spatial_transform(self, p, dx, invert=False):
        """
        Inputs:
            p - (bs, 2, V, V) local map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        NOTE: The agent stands at the central column of the last row in the
        ego-centric map and looks forward. But the rotation happens about the
        center of the map.  To handle this, first zero-pad pt_1 and then crop
        it after transforming.

        Conventions:
            The origin is at the bottom-center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction
        """
        V = p.shape[2]
        p_pad = bottom_row_padding(p)
        p_trans_pad = self._spatial_transform(p_pad, dx, invert=invert)
        # Crop out the original part
        p_trans = bottom_row_cropping(p_trans_pad, V)

        return p_trans

    def _spatial_transform(self, p, dx, invert=False):
        """
        Applies the transformation dx to image p.
        Inputs:
            p - (bs, 2, H, W) map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        Conventions:
            The origin is at the center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction

        Note: These denote transforms in an agent's position. Not the image directly.
        For example, if an agent is moving upward, then the map will be moving downward.
        To disable this behavior, set invert=False.
        """
        s = self.map_config["scale"]
        # Convert dx to map image coordinate system with X as rightward and Y as downward
        dx_map = torch.stack(
            [(dx[:, 1] / s), -(dx[:, 0] / s), dx[:, 2]], dim=1
        )  # anti-clockwise rotation
        p_trans = spatial_transform_map(p, dx_map, invert=invert)

        return p_trans

    def _register_map(self, m, p, x):
        """
        Given the locally computed map, register it to the global map based
        on the current position.

        Inputs:
            m - (bs, F, M, M) global map
            p - (bs, F, V, V) local map
            x - (bs, 3) in global coordinates
        """
        V = self.map_config["size"]
        s = self.map_config["scale"]
        M = m.shape[2]
        Vby2 = (V - 1) // 2 if V % 2 == 1 else V // 2
        Mby2 = (M - 1) // 2 if M % 2 == 1 else M // 2
        # The agent stands at the bottom-center of the egomap and looks upward
        left_h_pad = Mby2 - V + 1
        right_h_pad = M - V - left_h_pad
        left_w_pad = Mby2 - Vby2
        right_w_pad = M - V - left_w_pad
        # Add zero padding to p so that it matches size of global map
        p_pad = F.pad(
            p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0
        )
        # Register the local map
        p_reg = self._spatial_transform(p_pad, x)
        # Aggregate
        m_updated = self._aggregate(m, p_reg)

        return m_updated

    def _aggregate(self, m, p_reg):
        """
        Inputs:
            m - (bs, 2, M, M) - global map
            p_reg - (bs, 2, M, M) - registered egomap
        """
        reg_type = self.config.registration_type
        beta = self.config.map_registration_momentum
        if reg_type == "max":
            m_updated = torch.max(m, p_reg)
        elif reg_type == "overwrite":
            # Overwrite only the currently explored regions
            mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            mask = mask.unsqueeze(1)
            m_updated = m * (1 - mask) + p_reg * mask
        elif reg_type == "moving_average":
            mask_unexplored = (
                (p_reg[:, 0] <= self.config.thresh_explored).float().unsqueeze(1)
            )
            mask_unfilled = (m[:, 1] == 0).float().unsqueeze(1)
            m_ma = p_reg * (1 - beta) + m * beta
            m_updated = (
                m * mask_unexplored
                + m_ma * (1.0 - mask_unexplored) * (1.0 - mask_unfilled)
                + p_reg * (1.0 - mask_unexplored) * mask_unfilled
            )
        elif reg_type == "entropy_moving_average":
            explored_mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            log_p_reg = torch.log(p_reg + EPS_MAPPER)
            log_1_p_reg = torch.log(1 - p_reg + EPS_MAPPER)
            entropy = -p_reg * log_p_reg - (1 - p_reg) * log_1_p_reg
            entropy_mask = (entropy.mean(dim=1) < self.config.thresh_entropy).float()
            explored_mask = explored_mask * entropy_mask
            unfilled_mask = (m[:, 1] == 0).float()
            m_updated = m
            # For regions that are unfilled, write as it is
            mask = unfilled_mask * explored_mask
            mask = mask.unsqueeze(1)
            m_updated = m_updated * (1 - mask) + p_reg * mask
            # For regions that are filled, do a moving average
            mask = (1 - unfilled_mask) * explored_mask
            mask = mask.unsqueeze(1)
            p_reg_ma = (p_reg * (1 - beta) + m_updated * beta) * mask
            m_updated = m_updated * (1 - mask) + p_reg_ma * mask
        else:
            raise ValueError(
                f"Mapper: registration_type: {self.config.registration_type} not defined!"
            )

        return m_updated

    def ext_register_map(self, m, p, x):
        return self._register_map(m, p, x)

    def _safe_cat(self, d1, d2):
        """Given two dicts of tensors with same keys, the values are
        concatenated if not None.
        """
        d = {}
        for k, v1 in d1.items():
            d[k] = None if v1 is None else torch.cat([v1, d2[k]], 0)
        return d
    