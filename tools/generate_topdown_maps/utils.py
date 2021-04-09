import torch
import torch.nn.functional as F
import numpy as np

from habitat.utils.geometry_utils import quaternion_to_list
from habitat_extensions.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar

def map_ego_to_global(mapper, height, width, x, map_size=3001):
    p = torch.ones(1,1,width,height)
    M = map_size

    heightby2 = (height - 1) // 2 if height % 2 == 1 else height // 2 
    widthby2 = (width - 1) // 2 if width % 2 == 1 else width // 2

    Mby2 = (M - 1) // 2 if M % 2 == 1 else M // 2
    # The agent stands at the center of the object bounding box

    rec_pad_h = Mby2 - heightby2
    rec_pad_w = Mby2 - widthby2

    # Add zero padding to p so that it matches size of global map
    p_pad = F.pad(
        p, (rec_pad_h + (1 - height % 2), rec_pad_h, rec_pad_w + (1 - width % 2), rec_pad_w), "constant", 0
    )
    # Register the local map
    p_reg = mapper._spatial_transform(p_pad, x)
        
    return(p_reg)

def pos_real_to_map(pos, episode):

    agent_position = pos

    origin = np.array(episode.start_position, dtype=np.float32)

    rotation_world_start = quaternion_from_coeff(episode.start_rotation)

    agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )

    rotation_world_start = quaternion_from_coeff(episode.start_rotation)

    direction_vector = np.array([0, 0, -1])
    heading_vector = quaternion_rotate_vector(rotation_world_start.inverse(), direction_vector)
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]


    return np.array(
            [-agent_position[2], agent_position[0], phi], dtype=np.float32,
        )  