import os
import cv2
import json
from habitat.utils.geometry_utils import quaternion_from_coeff
import tqdm
import gzip
import glob
import argparse
import numpy as np
import gc
import torch

import habitat
import habitat_extensions

from habitat.utils.visualizations import maps
from habitat_extensions.geometry_utils import compute_heading_from_quaternion

from einops import rearrange, asnumpy
from mapper.mapper import Mapper
from config.default import get_config
from habitat.config.default import Config as CN
from tools.generate_topdown_maps.utils import (
    map_ego_to_global,
    pos_real_to_map,
)

class DummyRLEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0):
        super(DummyRLEnv, self).__init__(config, dataset)
        self._env_ind = env_ind

    def get_reward_range(self):
        return -1.0, 1.0

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_env_ind(self):
        return self._env_ind

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def get_episode_map(env:habitat.RLEnv, mapper:Mapper, M:int, config:CN, device:torch.device):
    """Given the environment and the configuration, compute the global
    top-down wall and seen area maps by sampling maps for individual locations
    along a uniform grid in the environment, and registering them.
    """
    # Initialize a global map for the episode
    global_wall_map = torch.zeros(1, 2, M, M).to(device)
    global_seen_map = torch.zeros(1, 2, M, M).to(device)

    grid_size = config.TASK.GT_EGO_MAP.MAP_SCALE
    coordinate_max = maps.COORDINATE_MAX
    coordinate_min = maps.COORDINATE_MIN
    resolution = (coordinate_max - coordinate_min) / grid_size
    grid_resolution = (int(resolution), int(resolution))

    top_down_map = maps.get_topdown_map(
        env.habitat_env.sim, grid_resolution, 20000, draw_border=False,
    )

    map_w, map_h = top_down_map.shape

    intervals = (max(int(0.5 / grid_size), 1), max(int(0.5 / grid_size), 1))
    x_vals = np.arange(0, map_w, intervals[0], dtype=int)
    y_vals = np.arange(0, map_h, intervals[1], dtype=int)
    coors = np.stack(np.meshgrid(x_vals, y_vals), axis=2)  # (H, W, 2)
    coors = coors.reshape(-1, 2)  # (H*W, 2)
    map_vals = top_down_map[coors[:, 0], coors[:, 1]]
    valid_coors = coors[map_vals > 0]

    real_x_vals = coordinate_max - valid_coors[:, 0] * grid_size
    real_z_vals = coordinate_min + valid_coors[:, 1] * grid_size
    start_y = env.habitat_env.sim.get_agent_state().position[1]

    for j in tqdm.tqdm(range(real_x_vals.shape[0]),desc='occupacy map', position=2):
        for theta in np.arange(-np.pi, np.pi, np.pi / 3.0):
            position = [
                real_x_vals[j].item(),
                start_y.item(),
                real_z_vals[j].item(),
            ]
            rotation = [
                0.0,
                np.sin(theta / 2).item(),
                0.0,
                np.cos(theta / 2).item(),
            ]

            sim_obs = env.habitat_env.sim.get_observations_at(
                position, rotation, keep_agent_at_new_pose=True,
            )
            episode = env.habitat_env.current_episode
            obs = env.habitat_env.task.sensor_suite.get_observations(
                observations=sim_obs, episode=episode, task=env.habitat_env.task
            )
            ego_map_gt = torch.Tensor(obs["ego_map_gt"]).to(device)
            ego_map_gt = rearrange(ego_map_gt, "h w c -> () c h w")
            ego_wall_map_gt = torch.Tensor(obs["ego_wall_map_gt"]).to(device)
            ego_wall_map_gt = rearrange(ego_wall_map_gt, "h w c -> () c h w")
            pose_gt = torch.Tensor(obs["pose_gt"]).unsqueeze(0).to(device)
            global_seen_map = mapper.ext_register_map(
                global_seen_map, ego_map_gt, pose_gt
            )
            global_wall_map = mapper.ext_register_map(
                global_wall_map, ego_wall_map_gt, pose_gt
            )

    global_wall_map_np = asnumpy(rearrange(global_wall_map, "b c h w -> b h w c")[0])
    global_seen_map_np = asnumpy(rearrange(global_seen_map, "b c h w -> b h w c")[0])

    return global_seen_map_np, global_wall_map_np

def get_items_list(list_path):
    idx_label = {}
    label_idx = {}
    with open(list_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        for i, line in enumerate(lines):
            if i == 0: # header
                continue
            else:
                data = line.split()
                idx = data[0]
                label = data[1]
                hex_col = data[2]
                idx_label[idx] = label
                label_idx[label] = idx
    return [idx_label, label_idx]

[_ , label_idx] = get_items_list("data/mpcat40.tsv")
  
def generate_semantic_layers(env:habitat.RLEnv,mapper:Mapper, M:int, mapper_config:CN,  global_seen_map_np:torch.Tensor):
    grid_size = mapper_config.TASK.GT_EGO_MAP.MAP_SCALE
    start_y = env.habitat_env.sim.get_agent_state().position[1]
    
    # init the semantic layers 
    semantic_layers = []
    scene = env.habitat_env.sim.semantic_scene
    obj_pos = {}
    for name in label_idx.keys():
        obj_pos[name] = {obj.id : [obj.aabb.center,obj.aabb.sizes]
                            for obj in scene.objects
                            if name == obj.category.name()}

    for objects in tqdm.tqdm(obj_pos,desc='semantic map', position=3):
        if objects == 'chair':
            obj_mapping = np.zeros((M,M))
            gc.collect()
            for object in obj_pos[objects].items():
                pos = pos_real_to_map(object[1][0], env.habitat_env.current_episode)
                obj_height = object[1][0][1]
                # Check if obj is part of the current floor. the centre of the floor is 1 meter higher than the start position and we look for object inside a 2 meters bound.
                if np.abs(start_y - obj_height + 1) > 1:
                    continue
                
                edges_w = max(int(object[1][1][0] / grid_size), 1)
                edges_l = max(int(object[1][1][2] / grid_size), 1)
                trans_obj = asnumpy(map_ego_to_global(mapper, edges_w, edges_l,torch.Tensor(pos).unsqueeze(0),M).squeeze())

                obj_mapping = (obj_mapping + trans_obj) > 0

            reduced_obj_layer = obj_mapping
            semantic_layers.append(reduced_obj_layer)
    return np.array(semantic_layers)

def main(args):
    
    config = get_config()

    mapper_config = config.RL.ANS.MAPPER
    mapper_config.defrost()
    mapper_config.map_size = 130
    mapper_config.map_scale = 0.02
    mapper_config.freeze()

    mapper = Mapper(mapper_config, None)

    M = args.global_map_size
    skip_scene = args.skip_scene
    config_path = args.config_path
    save_dir = args.save_dir
    safe_mkdir(save_dir)

    seen_map_save_root = os.path.join(save_dir, "seen_area_maps")
    wall_map_save_root = os.path.join(save_dir, "wall_maps")
    semantic_map_save_root = os.path.join(save_dir, "semantic_maps")
    json_save_path = os.path.join(save_dir, "all_maps_info.json")

    config = habitat_extensions.get_extended_config(config_path)

    scenes_list = glob.glob(f"")
    dataset_path = config.DATASET.DATA_PATH.replace("{split}", config.DATASET.SPLIT)
    with gzip.open(dataset_path, "rt") as fp:
        dataset = json.load(fp)

    num_episodes = len(dataset["episodes"])

    print("===============> Loading data per scene")
    scene_to_data = {}
    if num_episodes == 0:
        content_path = os.path.join(
            dataset_path[: -len(f"{config.DATASET.SPLIT}.json.gz")], "content"
        )
        scene_paths = glob.glob(f"{content_path}/*")
        print(f"Number of scenes found: {len(scene_paths)}")
        for scene_data_path in scene_paths:
            with gzip.open(scene_data_path, "rt") as fp:
                scene_data = json.load(fp)
            num_episodes += len(scene_data["episodes"])
            scene_id = scene_data["episodes"][0]["scene_id"].split("/")[-1]
            scene_to_data[scene_id] = scene_data["episodes"]
    else:
        for ep in dataset["episodes"]:
            scene_id = ep["scene_id"].split("/")[-1]
            if scene_id not in scene_to_data:
                scene_to_data[scene_id] = []
            scene_to_data[scene_id].append(ep)

    print("===============> Computing heights for different floors in each scene")
    scenes_to_floor_heights = {}
    for scene_id, scene_data in scene_to_data.items():
        # Identify the number of unique floors in this scene
        floor_heights = []
        for ep in scene_data:
            height = ep["start_position"][1]
            if len(floor_heights) == 0:
                floor_heights.append(height)
            # Measure height difference from all existing floors
            d2floors = map(lambda x: abs(x - height), floor_heights)
            d2floors = np.array(list(d2floors))
            if not np.any(d2floors < 0.5):
                floor_heights.append(height)
        # Store this in the dict
        scenes_to_floor_heights[scene_id] = floor_heights

    env = DummyRLEnv(config=config)
    env.seed(1234)
    _ = env.reset()
    device = torch.device("cuda:0")

    safe_mkdir(seen_map_save_root)
    safe_mkdir(wall_map_save_root)
    safe_mkdir(semantic_map_save_root)

    # Data format for saving top-down maps per scene:
    # For each split, create a json file that contains the following dictionary:
    # key - scene_id
    # value - [{'floor_height': ...,
    #           'seen_map_path': ...,
    #           'wall_map_path': ...,
    #           'world_position': ...,
    #           'world_heading': ...},
    #          .,
    #          .,
    #          .,
    #         ]
    # The floor_height specifies a single height value on that floor.
    # All other heights within 0.5m of this height will correspond to this floor.
    # The *_map_path specifies the path to a .npy file that contains the
    # corresponding map. This map is in the world coordinate system, not episode
    # centric start-view coordinate system.
    # The world_position is the (X, Y, Z) position of the agent w.r.t. which this
    # map was computed. The world_heading is the clockwise rotation (-Z to X)
    # of the agent in the world coordinates.
    # The .npy files will be stored in seen_map_save_root and wall_map_save_root.

    # Create top-down maps per scene, per floor
    per_scene_per_floor_maps = {}
    print("===============> generate meta information for gt map")
    for target_scene in tqdm.tqdm(scene_to_data.keys()):
        per_scene_per_floor_maps[target_scene] = {}
        for episode in scene_to_data[target_scene]:
            scene_id = target_scene
            start_position = episode['start_position']
            start_rotation = episode['start_rotation']
            start_height = start_position[1]
            floor_heights = scenes_to_floor_heights[scene_id]
            d2floors = map(lambda x: abs(x - start_height), floor_heights)
            d2floors = np.array(list(d2floors))
            floor_idx = np.where(d2floors < 0.5)[0][0].item()   
            if floor_idx in per_scene_per_floor_maps[scene_id]:
                continue
            start_heading = compute_heading_from_quaternion(quaternion_from_coeff(start_rotation))
            seen_map_save_path = f"{seen_map_save_root}/{scene_id}_{floor_idx}.npy"
            wall_map_save_path = f"{wall_map_save_root}/{scene_id}_{floor_idx}.npy"
            semantic_map_save_path = f"{semantic_map_save_root}/{scene_id}_{floor_idx}.npy"


            save_dict = {
            "seen_map_path": seen_map_save_path,
            "wall_map_path": wall_map_save_path,
            "semantic_map_path": semantic_map_save_path,
            "floor_height": start_height,
            "start_rotation": start_rotation,
            "world_position": start_position,
            "world_heading": start_heading,
            "scene_id": episode['scene_id']
            }

            per_scene_per_floor_maps[scene_id][floor_idx] = save_dict
            if len(per_scene_per_floor_maps[scene_id]) == len(scenes_to_floor_heights[scene_id]):
                break
    print("===============> save meta information for gt map")        
    save_json = {}
    for scene in per_scene_per_floor_maps.keys():
        scene_save_data = []
        for floor_idx, floor_data in per_scene_per_floor_maps[scene].items():
            scene_save_data.append(floor_data)
        save_json[scene] = scene_save_data

    json.dump(save_json, open(json_save_path, "w"))    

    print("===============> start to draw semantic map")  
    scene_ids = sorted(list(per_scene_per_floor_maps.keys()))
    print(scene_ids)
    start_scene = scene_ids[skip_scene]
    print(f"===============> start with scene {start_scene}")

    for target_scene in tqdm.tqdm(scene_ids[skip_scene:],desc='scenes', position=0):
        for floor_idx in per_scene_per_floor_maps[target_scene]:
            
            scene_meta_info = per_scene_per_floor_maps[target_scene][floor_idx]
            # don't regenerate maps
            
            if os.path.isfile(scene_meta_info['semantic_map_path']):
                continue
            print(scene_meta_info)

            env.habitat_env.current_episode.start_position = scene_meta_info['world_position']
            env.habitat_env.current_episode.start_rotation = scene_meta_info['start_rotation']
            env.habitat_env.current_episode.scene_id = scene_meta_info['scene_id']

            env.habitat_env.reconfigure(env.habitat_env._config)
            _ = env.habitat_env.task.reset(env.habitat_env.current_episode)

            scene_id = target_scene
            agent_state = env.habitat_env.sim.get_agent_state()
            start_position = np.array(agent_state.position)

            global_seen_map, global_wall_map = get_episode_map(
                env, mapper, M, config, device
            )

            #generate semantic layers
            global_semantic_map = generate_semantic_layers(env,mapper, M,config,global_seen_map)

            seen_map_save_path = f"{seen_map_save_root}/{scene_id}_{floor_idx}.npy"
            wall_map_save_path = f"{wall_map_save_root}/{scene_id}_{floor_idx}.npy"
            semantic_map_save_path = f"{semantic_map_save_root}/{scene_id}_{floor_idx}.npy"
            np.save(seen_map_save_path, (global_seen_map > 0))
            np.save(wall_map_save_path, (global_wall_map > 0))
            np.save(semantic_map_save_path, (global_semantic_map > 0))

            # clean the memory to avoid overflow
            global_seen_map = None
            global_wall_map = None
            global_semantic_map = None
            gc.collect()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--global-map-size", type=int, required=True)
    parser.add_argument("--skip-scene", type=int, required=True, default=0)

    args = parser.parse_args()

    main(args)
