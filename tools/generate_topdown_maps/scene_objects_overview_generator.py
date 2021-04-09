import os
import json
import torch
import pandas as pd 
import argparse
import habitat
import habitat_extensions
import tqdm
import habitat_sim
from config.default import get_config

from tools.generate_topdown_maps.gt_map_generator import DummyRLEnv, get_items_list


def get_obj_per_scene(env,label_idx):
    #Get the semantic scene of the env
    scene = env.habitat_env.sim.semantic_scene

    #Map all objects and their position/dimension in the  current floor
    obj_pos = {}
    for name in label_idx.keys():
        obj_pos[name] = {obj.id : [obj.aabb.center,obj.aabb.sizes]
                        for obj in scene.objects
                            if name == obj.category.name()}
    return obj_pos

def switch_to_next_scene(env, scene_id):
    env.habitat_env.current_episode.scene_id = scene_id
    env.habitat_env.reconfigure(env.habitat_env._config)
    _ = env.habitat_env.task.reset(env.habitat_env.current_episode)




def main(args):
    file_path = args.map_info
    with open(file_path) as json_file:
        all_maps_info = json.load(json_file)

    _ , label_idx = get_items_list("data/mpcat40.tsv")

    config_path = "tools/generate_topdown_maps/config/mp3d_train.yaml"

    minDistance = 0
    maxDistance = 2.5

    config = get_config(config_path)
    config = habitat_extensions.get_extended_config(config_path)

    try:
        env.close()
    except NameError:
        pass


    env = DummyRLEnv(config=config)
    env.seed(1234)
    device = torch.device("cuda:0")

    _ = env.reset()

    scene_objects = {}
    scene_ids = sorted(list(all_maps_info.keys()))
    
    for scene_id in scene_ids:
        scene_objects[scene_id] = {}
        
        floor_id = 0
        for scene_floor in all_maps_info[scene_id]:
            scene_objects[scene_id][floor_id] = {}
            floor_id += 1

    for scene in tqdm.tqdm(scene_ids):
        switch_to_next_scene(env, all_maps_info[scene][0]["scene_id"])
        obj_pos = get_obj_per_scene(env,label_idx)
        for floor_id in scene_objects[scene]:
            
            floor_height = all_maps_info[scene][floor_id]["floor_height"]
            floor_objects = {}
            scene_objects[scene][floor_id] = floor_objects
            
            for target_obj in obj_pos.keys():
                
                object_class = target_obj
                floor_objects[object_class] = 0
                target_objects = obj_pos[object_class]
                
                for obj_id in target_objects:
                    
                    objectHeight = target_objects[obj_id][0][1]
                    dObjectToFloor = objectHeight - floor_height

                    if dObjectToFloor > minDistance and dObjectToFloor < maxDistance: #Check if Object is within 2.5m above the floor height
                        floor_objects[object_class] += 1
                    else:
                        continue
        
        

    #Create dataframe out of scene_objects dictionary
    df_scene_objects = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in scene_objects.items()}, axis=0)
    df_scene_objects.to_csv("data/scene_object_prevelance.tsv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-info", type=str, required=True)
    args = parser.parse_args()
    main(args)