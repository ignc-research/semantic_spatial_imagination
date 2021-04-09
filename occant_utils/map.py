import torch
import torch.nn.functional as F
import cv2
import numpy as np
from einops import rearrange


def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size

    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer

    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(0)
        .expand(crop_size, -1)
        .contiguous()
        .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(1)
        .expand(-1, crop_size)
        .contiguous()
        .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
    ) / Wby2
    crop_grid[:, :, :, 1] = (
        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
    ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode)

    return h_cropped

def spatial_transform_map(p, x, invert=True, mode="bilinear"):
    """
    Inputs:
        p     - (bs, f, H, W) Tensor
        x     - (bs, 3) Tensor (x, y, theta) transforms to perform
    Outputs:
        p_trans - (bs, f, H, W) Tensor
    Conventions:
        Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.

    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    device = p.device
    H, W = p.shape[2:]

    trans_x = x[:, 0]
    trans_y = x[:, 1]
    # Convert translations to -1.0 to 1.0 range
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

    trans_x = trans_x / Wby2
    trans_y = trans_y / Hby2
    rot_t = x[:, 2]

    sin_t = torch.sin(rot_t)
    cos_t = torch.cos(rot_t)

    # This R convention means Y axis is downwards.
    A = torch.zeros(p.size(0), 3, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    # Since this is a source to target mapping, and F.affine_grid expects
    # target to source mapping, we have to invert this for normal behavior.
    Ainv = torch.inverse(A)

    # If target to source mapping is required, invert is enabled and we invert
    # it again.
    if invert:
        Ainv = torch.inverse(Ainv)

    Ainv = Ainv[:, :2]
    grid = F.affine_grid(Ainv, p.size())
    p_trans = F.grid_sample(p, grid, mode=mode)

    return p_trans

def bottom_row_padding(p):
    V = p.shape[2]
    Vby2 = (V - 1) / 2 if V % 2 == 1 else V // 2
    left_h_pad = 0
    right_h_pad = int(V - 1)
    if V % 2 == 1:
        left_w_pad = int(Vby2)
        right_w_pad = int(Vby2)
    else:
        left_w_pad = int(Vby2) - 1
        right_w_pad = int(Vby2)

    # Pad so that the origin is at the center
    p_pad = F.pad(p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0)

    return p_pad

def bottom_row_cropping(p, map_size):
    bs = p.shape[0]
    V = map_size
    Vby2 = (V - 1) / 2 if V % 2 == 1 else V // 2
    device = p.device

    x_crop_center = torch.zeros(bs, 2).to(device)
    x_crop_center[:, 0] = V - 1
    x_crop_center[:, 1] = Vby2
    x_crop_size = V

    p_cropped = crop_map(p, x_crop_center, x_crop_size)

    return p_cropped

def add_pose(pose_a, pose_ab):
    """
    Add pose_ab (in ego-coordinates of pose_a) to pose_a
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_ab, y_ab, theta_ab = torch.unbind(pose_ab, dim=1)

    r_ab = torch.sqrt(x_ab ** 2 + y_ab ** 2)
    phi_ab = torch.atan2(y_ab, x_ab)

    x_b = x_a + r_ab * torch.cos(phi_ab + theta_a)
    y_b = y_a + r_ab * torch.sin(phi_ab + theta_a)
    theta_b = theta_a + theta_ab
    theta_b = torch.atan2(torch.sin(theta_b), torch.cos(theta_b))

    pose_b = torch.stack([x_b, y_b, theta_b], dim=1)  # (bs, 3)

    return pose_b

def subtract_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_b, y_b, theta_b = torch.unbind(pose_b, dim=1)

    r_ab = torch.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)  # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) - theta_a  # (bs, )
    theta_ab = theta_b - theta_a  # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack(
        [r_ab * torch.cos(phi_ab), r_ab * torch.sin(phi_ab), theta_ab,], dim=1
    )  # (bs, 3)

    return x_ab

def transpose_image(img):
    """
    Inputs:
        img - (bs, H, W, C) torch Tensor
    """
    img_p = img.permute(0, 3, 1, 2)  # (bs, C, H, W)
    return img_p


def process_image(img, img_mean, img_std):
    """
    Convert HWC -> CHW, normalize image.
    Inputs:
        img - (bs, H, W, C) torch Tensor
        img_mean - list of per-channel means
        img_std - list of per-channel stds

    Outputs:
        img_p - (bs, C, H, W)
    """
    C = img.shape[3]
    device = img.device

    img_p = rearrange(img.float(), "b h w c -> b c h w")
    img_p = img_p / 255.0  # (bs, C, H, W)

    if type(img_mean) == type([]):
        img_mean_t = rearrange(torch.Tensor(img_mean), "c -> () c () ()").to(device)
        img_std_t = rearrange(torch.Tensor(img_std), "c -> () c () ()").to(device)
    else:
        img_mean_t = img_mean.to(device)
        img_std_t = img_std.to(device)

    img_p = (img_p - img_mean_t) / img_std_t

    return img_p

def grow_projected_map(proj_map, local_map, iterations=2):
    """
    proj_map - (H, W, 2) map
    local_map - (H, W, 2) map

    channel 0 - 1 if occupied, 0 otherwise
    channel 1 - 1 if explored, 0 otherwise
    """
    proj_map = np.copy(proj_map)
    HEIGHT, WIDTH = proj_map.shape[:2]

    explored_local_mask = local_map[..., 1] == 1
    free_local_mask = (local_map[..., 0] == 0) & explored_local_mask
    occ_local_mask = (local_map[..., 0] == 1) & explored_local_mask

    # Iteratively expand multiple times
    for i in range(iterations):
        # Generate regions which are predictable

        # ================ Processing free space ===========================
        # Pick only free areas that are visible
        explored_proj_map = (proj_map[..., 1] == 1).astype(np.uint8) * 255
        free_proj_map = ((proj_map[..., 0] == 0) & explored_proj_map).astype(
            np.uint8
        ) * 255
        occ_proj_map = ((proj_map[..., 0] == 1) & explored_proj_map).astype(
            np.uint8
        ) * 255

        # For the first iteration, aggressively close holes
        if i == 0:
            close_kernel = np.ones((3, 3), np.uint8)
            for itr in range(5):
                free_proj_map = cv2.morphologyEx(
                    free_proj_map, cv2.MORPH_CLOSE, close_kernel
                )
            free_proj_map = (free_proj_map > 0).astype(np.uint8) * 255

        dilate_kernel = np.ones((7, 7), np.uint8)

        # Expand only GT free area
        for itr in range(2):
            free_proj_map_edges = cv2.Canny(free_proj_map, 50, 100)
            free_proj_map_edges_dilated = cv2.dilate(
                free_proj_map_edges, dilate_kernel, iterations=3
            )
            free_mask = (
                (free_proj_map_edges_dilated > 0) | (free_proj_map > 0)
            ) & free_local_mask
            free_proj_map = free_mask.astype(np.uint8) * 255

        # Dilate to include some occupied area
        free_proj_map = cv2.dilate(free_proj_map, dilate_kernel, iterations=1)
        free_proj_map = (free_proj_map > 0).astype(np.uint8)

        # Close holes
        close_kernel = np.ones((3, 3), np.uint8)
        free_proj_map = cv2.morphologyEx(free_proj_map, cv2.MORPH_CLOSE, close_kernel)

        # ================ Processing occupied space ===========================
        # For the first iteration, aggressively close holes
        if i == 0:
            close_kernel = np.ones((3, 3), np.uint8)
            for itr in range(3):
                occ_proj_map = cv2.morphologyEx(
                    occ_proj_map, cv2.MORPH_CLOSE, close_kernel
                )
            occ_proj_map = (occ_proj_map > 0).astype(np.uint8) * 255

        dilate_kernel = np.ones((3, 3), np.uint8)

        # Expand only GT occupied area
        for itr in range(1):
            occ_proj_map_edges = cv2.Canny(occ_proj_map, 50, 100)
            occ_proj_map_edges_dilated = cv2.dilate(
                occ_proj_map_edges, dilate_kernel, iterations=3
            )
            occ_mask = (
                (occ_proj_map_edges_dilated > 0) | (occ_proj_map > 0)
            ) & occ_local_mask
            occ_proj_map = occ_mask.astype(np.uint8) * 255

        dilate_kernel = np.ones((9, 9), np.uint8)
        # Expand the free space around the GT occupied area
        for itr in range(2):
            occ_proj_map_dilated = cv2.dilate(occ_proj_map, dilate_kernel, iterations=3)
            free_mask_around_occ = (occ_proj_map_dilated > 0) & free_local_mask
            occ_proj_map = ((occ_proj_map > 0) | free_mask_around_occ).astype(
                np.uint8
            ) * 255

        # Close holes
        close_kernel = np.ones((3, 3), np.uint8)
        occ_proj_map = cv2.morphologyEx(occ_proj_map, cv2.MORPH_CLOSE, close_kernel)

        # Include originally present areas in proj_map
        predictable_regions_mask = (
            (explored_proj_map > 0) | (free_proj_map > 0) | (occ_proj_map > 0)
        )

        # Create new proj_map
        proj_map = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        proj_map[predictable_regions_mask & occ_local_mask, 0] = 1
        proj_map[predictable_regions_mask, 1] = 1

    gt_map = proj_map

    return gt_map

def colorize_ego_map(ego_map):
    """
    ego_map - (V, V, 2) array where 1st channel represents prob(occupied space) an
              d 2nd channel represents prob(explored space)
    """
    explored_mask = ego_map[..., 1] > 0.5
    occupied_mask = np.logical_and(ego_map[..., 0] > 0.5, explored_mask)
    free_space_mask = np.logical_and(ego_map[..., 0] <= 0.5, explored_mask)
    unexplored_mask = ego_map[..., 1] <= 0.5

    ego_map_color = np.zeros((*ego_map.shape[:2], 3), np.uint8)

    # White unexplored map
    ego_map_color[unexplored_mask, 0] = 255
    ego_map_color[unexplored_mask, 1] = 255
    ego_map_color[unexplored_mask, 2] = 255

    # Blue occupied map
    ego_map_color[occupied_mask, 0] = 0
    ego_map_color[occupied_mask, 1] = 0
    ego_map_color[occupied_mask, 2] = 255

    # Green free space map
    ego_map_color[free_space_mask, 0] = 0
    ego_map_color[free_space_mask, 1] = 255
    ego_map_color[free_space_mask, 2] = 0

    return ego_map_color

def dilate_tensor(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    if type(size) == int:
        padding = size // 2
    else:
        padding = tuple([v // 2 for v in size])
        
    for i in range(iterations):
        x = F.max_pool2d(x, size, stride=1, padding=padding)

    return x


def erode_tensor(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    if type(size) == int:
        padding = size // 2
    else:
        padding = tuple([v // 2 for v in size])
    for i in range(iterations):
        x = -F.max_pool2d(-x, size, stride=1, padding=padding)

    return x


def morphology_close(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    x = dilate_tensor(x, size, iterations)
    x = erode_tensor(x, size, iterations)
    return x

