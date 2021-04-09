import os
file_path = "/home/shen/myproject/habitat/SemanticAnticipation/"
os.chdir(file_path)

from habitat.core.env import RLEnv
from habitat_baselines.common.utils import (
    batch_obs,
)
from einops import rearrange
from occant_utils.map import (
    dilate_tensor, process_image,
    transpose_image,
)
from habitat_sim.utils.common import d3_40_colors_rgb
import matplotlib.pyplot as plt
from config.default import get_config
import random
import numpy as np
import torch
from rl.models.mapnet import DepthProjectionNet
from rl.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from mapper.mapper import Mapper
from tools.semantic_anti_train.semantic_anti_trainner import (   
    SemAntExpTrainer,
    SemAnticipator,
    SemAnticipationWrapper,
)
from habitat.utils.visualizations import maps
import tqdm
from habitat.config.default import Config as CN
from einops import asnumpy
import gc
from rl.common.environments import ExpRLEnv
from tools.mit_semseg.models import ModelBuilder, SegmentationModule
from tools.mit_semseg.utils import colorEncode
import scipy.io
import PIL.Image, torchvision.transforms
import pickle
import cv2
from semantic_utils.common import safe_mkdir
import glob
# from examples.get_eval_map import get_eval_map
def save_episode_info(env,path):
    info = {
        "episode_id": env.habitat_env.current_episode.episode_id,
        "scene_id": env.habitat_env.current_episode.scene_id,
        "start_position": env.habitat_env.current_episode.start_position,
        "start_rotation": env.habitat_env.current_episode.start_rotation,
    }
    np.save(path,info)

colors = scipy.io.loadmat('tools/mit_semseg/data/color150.mat')['colors']
net_encoder = ModelBuilder.build_encoder(
    arch='hrnetv2',
    fc_dim=720,
    weights='tools/mit_semseg/ckpt/hrnetv2/encoder_epoch_30.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='c1',
    fc_dim=720,
    num_class=150,
    weights='tools/mit_semseg/ckpt/hrnetv2/decoder_epoch_30.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
objectName = "chair"
config = get_config("tools/semantic_anti_eval/config/se_check_" + objectName + "_anticipator_256.yaml")
random.seed(config.TASK_CONFIG.SEED)
np.random.seed(config.TASK_CONFIG.SEED)
torch.manual_seed(config.TASK_CONFIG.SEED)
trainer = SemAntExpTrainer(config)

ppo_cfg = trainer.config.RL.PPO
ans_cfg = trainer.config.RL.ANS
mapper_cfg = trainer.config.RL.ANS.MAPPER
occ_cfg = trainer.config.RL.ANS.SEMANTIC_ANTICIPATOR
trainer.device = (
    torch.device("cuda", 1)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
sem_cfg = ans_cfg.SEMANTIC_ANTICIPATOR
mapper_cfg = ans_cfg.MAPPER
[imgh, imgw] = ans_cfg['image_scale_hw']
sem_model = SemAnticipationWrapper(
    SemAnticipator(sem_cfg), mapper_cfg.map_size, (imgh, imgw)
)

trainer.mapper = Mapper(mapper_cfg,sem_model).to(trainer.device)


checkpoints = glob.glob(f"{trainer.config.CHECKPOINT_FOLDER}/*.pth")
ppo_cfg = trainer.config.RL.PPO
# Load lastest checkpoint
last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
checkpoint_path = last_ckpt
# Restore checkpoints to models
# ckpt_dict = trainer.load_checkpoint('data/new_checkpoints_se_256/ckpt.179.pth')
ckpt_dict = trainer.load_checkpoint(checkpoint_path)
trainer.mapper.load_state_dict(ckpt_dict["mapper"])

depth_projection_net = DepthProjectionNet(
            trainer.config.RL.ANS.SEMANTIC_ANTICIPATOR.EGO_PROJECTION
        )
trainer.mapper.eval()
segmentation_module.eval()
segmentation_module.cuda(trainer.device)

def display_map(topdown_map,ax=None):
    
    if ax==None:
        plt.figure(figsize=(20,20))
        ax = plt.subplot(1, 1, 1)
        #ax.axis("off")
        plt.imshow(topdown_map)
        plt.show(block=False)
    else:
        ax = ax
        return ax.imshow(topdown_map)

env = ExpRLEnv(config=trainer.config)

pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])


def get_eval_map(env:ExpRLEnv, trainer:SemAntExpTrainer,  M:int,depth_projection_net:DepthProjectionNet, step:int, objectName:str):
    """Given the environment and the configuration, compute the global
    top-down wall and seen area maps by sampling maps for individual locations
    along a uniform grid in the environment, and registering them.

    step (m): the length between two measure points
    """
    # Initialize a global map for the episode
    scene_name = env.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]
    dirPath = './data/debug/data/' + objectName + "/" + scene_name + '_' + env.habitat_env.current_episode.episode_id + '/' + "record/"
    safe_mkdir(dirPath)
    mapper = trainer.mapper
    config = trainer.config.TASK_CONFIG 
    device = trainer.device

    scene = env.habitat_env.sim.semantic_scene
    obj_pos = {}
    obj_pos = {obj.id : [obj.aabb.center,obj.aabb.sizes]
                            for obj in scene.objects
                            if objectName == obj.category.name()}
                            
    with open(dirPath + 'obj_pos.pkl', 'wb') as f: pickle.dump(obj_pos, f)
    objectIndexes = [int(key.split('_')[-1]) for key in obj_pos.keys()]

    global_wall_map = torch.zeros(1, 2, M, M).to(device)
    global_seen_map = torch.zeros(1, 2, M, M).to(device)
    global_object_map = torch.zeros(1, 2, M, M).to(device)
    global_se_map = torch.zeros(1, 2, M, M).to(device)
    global_se_filtered_map = torch.zeros(1, 2, M, M).to(device)
    global_se_seen_map = torch.zeros(1, 2, M, M).to(device)
    global_se_map_mit = torch.zeros(1, 2, M, M).to(device)
    global_se_map_gt = torch.zeros(1, 2, M, M).to(device)

    grid_size = config.TASK.GT_EGO_MAP.MAP_SCALE
    coordinate_max = maps.COORDINATE_MAX
    coordinate_min = maps.COORDINATE_MIN
    resolution = (coordinate_max - coordinate_min) / grid_size
    grid_resolution = (int(resolution), int(resolution))

    top_down_map = maps.get_topdown_map(
        env.habitat_env.sim, grid_resolution, 20000, draw_border=False,
    )

    map_w, map_h = top_down_map.shape

    intervals = (max(int(step / grid_size), 1), max(int(step / grid_size), 1))
    x_vals = np.arange(0, map_w, intervals[0], dtype=int)
    y_vals = np.arange(0, map_h, intervals[1], dtype=int)
    coors = np.stack(np.meshgrid(x_vals, y_vals), axis=2)  # (H, W, 2)
    coors = coors.reshape(-1, 2)  # (H*W, 2)
    map_vals = top_down_map[coors[:, 0], coors[:, 1]]
    valid_coors = coors[map_vals > 0]

    real_x_vals = coordinate_max - valid_coors[:, 0] * grid_size
    real_z_vals = coordinate_min + valid_coors[:, 1] * grid_size
    start_y = env.habitat_env.sim.get_agent_state().position[1]
    index = 0
    records = []

    totalOffsetX = np.random.uniform(-1.5, 1.5)
    totalOffsetY = np.random.uniform(-1.5, 1.5)
    for j in tqdm.tqdm(range(real_x_vals.shape[0]),desc='occupacy map', position=2):
        for theta in np.arange(-np.pi, np.pi, np.pi ):
            index +=1

            randomAngle = np.random.uniform(-np.pi,np.pi)
            randomX = np.random.uniform(-0.5, 0.5)
            randomY = np.random.uniform(-0.5, 0.5)

            position = [
                real_x_vals[j].item() + randomX + totalOffsetX,
                start_y.item(),
                real_z_vals[j].item() + randomY + totalOffsetY,
            ]
            rotation = [
                0.0,
                np.sin((theta + randomAngle) / 2).item(),
                0.0,
                np.cos((theta + randomAngle) / 2).item(),
            ]

            sim_obs = env.habitat_env.sim.get_observations_at(
                position, rotation, keep_agent_at_new_pose=True,
            )
            episode = env.habitat_env.current_episode
            obs = obs, _, _, _ = env.step(action={'action':1})
            batch = batch_obs([obs], device=trainer.device)

            semantic = batch["semantic"]
            mask = torch.zeros_like(semantic,dtype=bool)
            for objectIndex in objectIndexes:
                mask ^= (semantic == objectIndex)
            mask = rearrange(mask, "b h w  -> b h w ()")
            batch["object_mask_gt"] = mask

            ego_map_gt_b = depth_projection_net(
                    rearrange(batch["depth"], "b h w c -> b c h w")
                )
            object_depth = batch["depth"]*(mask > 0) + (mask == 0)*100
            
            ego_map_gt_object = depth_projection_net(rearrange(object_depth, "b h w c -> b c h w"))
            batch["ego_map_gt_object"] = ego_map_gt_object
            #begin use MIT Seg
            img_data = pil_to_tensor(rearrange(batch["rgb"], "b h w c -> b c h w")[0] / 255) 
            singleton_batch = {'img_data': img_data[None]}
            output_size = img_data.shape[1:]
            with torch.no_grad():
                scores = segmentation_module(singleton_batch, segSize=output_size)
            _, pred = torch.max(scores, dim=1)
            # visualize_result(pred)
            #end use MIT Seg
            batch["mit"] = pred
            #mask for chair
            
            if objectName == "chair":
                mask = (pred == 19)^(pred == 30)^(pred == 75)
            elif objectName == "bed":
                mask = (pred == 7)
            elif objectName == "table":
                mask = (pred == 15)
            else:
                mask = pred > 0
            
            mask = rearrange(mask, "b h w  -> b h w ()")
            batch["object_mask_mit"] = mask
            object_depth2 = batch["depth"]*(mask > 0) + (mask == 0)*100
            ego_map_gt_object2 = depth_projection_net(rearrange(object_depth2, "b h w c -> b c h w"))
            batch["ego_map_mit_object"] = ego_map_gt_object2
            ego_map_gt_anti = transpose_image(batch["ego_map_gt_anticipated"]).to(trainer.device)


            pu_inputs_t = {
                "rgb": process_image(batch["rgb"], trainer.mapper.img_mean_t, trainer.mapper.img_std_t).to(trainer.device),
                "depth": transpose_image(batch["depth"]).to(trainer.device),
                "ego_map_gt": transpose_image(
                        rearrange(ego_map_gt_b, "b c h w -> b h w c")
                    ).to(trainer.device),
                "ego_map_gt_anticipated": ego_map_gt_anti,
            }
            pu_inputs = trainer.mapper._safe_cat(pu_inputs_t, pu_inputs_t)

            torch.save(batch, dirPath + 'pu_inputs_t_'+str(index)+'.pt')

            pu_output = trainer.mapper.projection_unit(pu_inputs)
            se_map = asnumpy(pu_output['sem_estimate'])[0,:,:,:]

            ego_map_gt_b_np =  asnumpy(ego_map_gt_b[0,...])
            se_seen = ego_map_gt_b_np*se_map
            dilation_mask = np.ones((100, 100))

            current_mask = cv2.dilate(
                se_seen.astype(np.float32), dilation_mask, iterations=1,
            ).astype(np.float32)

            se_filtered_map = se_map
            se_seen_map = torch.Tensor(se_seen).to(device)
            se_seen_map = rearrange(se_seen_map, "c h w -> () c h w")
            se_seen_map = (se_seen_map[:,0])[None,:]

            se_map = torch.Tensor(se_map).to(device)
            se_map = rearrange(se_map, "c h w -> () c h w")

            se_filtered_map = torch.Tensor(se_filtered_map).to(device)
            se_filtered_map = rearrange(se_filtered_map, "c h w -> () c h w")

            torch.save(se_filtered_map, dirPath + 'se_filtered_map_'+str(index)+'.pt')

            ego_map_gt = torch.Tensor(obs["ego_map_gt"]).to(device)
            ego_map_gt = rearrange(ego_map_gt, "h w c -> () c h w")

            ego_wall_map_gt = torch.Tensor(obs["ego_wall_map_gt"]).to(device)
            ego_wall_map_gt = rearrange(ego_wall_map_gt, "h w c -> () c h w")



            pose_gt = torch.Tensor(obs["pose_gt"]).unsqueeze(0).to(device)

            global_se_seen_map = mapper.ext_register_map(
                global_se_seen_map, se_seen_map, pose_gt
            )

            global_se_map_gt = mapper.ext_register_map(
                global_se_map_gt, ego_map_gt_anti, pose_gt
            )

            global_seen_map = mapper.ext_register_map(
                global_seen_map, ego_map_gt_b, pose_gt
            )

            global_object_map = mapper.ext_register_map(
                global_object_map, ego_map_gt_object, pose_gt
            )

            global_wall_map = mapper.ext_register_map(
                global_wall_map, ego_wall_map_gt, pose_gt
            )
            global_se_map = mapper.ext_register_map(
                global_se_map, se_map, pose_gt
            )

            global_se_filtered_map = mapper.ext_register_map(
                global_se_filtered_map, se_filtered_map, pose_gt
            )

            global_se_map_mit = mapper.ext_register_map(
                global_se_map_mit, ego_map_gt_object2, pose_gt
            )
            se_filtered_map = None
            batch = None
            se_map = None
            ego_map_gt = None
            ego_wall_map_gt = None
            obs = None
            pu_inputs_t = None
            pu_output=None
            _ = None
            pu_inputs=None
            gc.collect()
            torch.cuda.empty_cache()

    mask = dilate_tensor(global_seen_map*(1 - global_wall_map > 0)*global_se_map,(51, 51))

    global_se_filtered_map = global_se_filtered_map*mask
    global_se_seen_map = global_se_seen_map*mask

    mask = dilate_tensor(global_seen_map*(1 - global_wall_map > 0)*global_object_map,(51, 51))
    global_se_map_gt = global_se_map_gt*mask

    global_wall_map_np = asnumpy(rearrange(global_wall_map, "b c h w -> b h w c")[0])
    global_seen_map_np = asnumpy(rearrange(global_seen_map, "b c h w -> b h w c")[0])
    global_se_map_np = asnumpy(rearrange(global_se_map, "b c h w -> b h w c")[0])
    global_se_global_se_filtered_map_np= asnumpy(rearrange(global_se_filtered_map, "b c h w -> b h w c")[0])

    global_se_map_mit_np= asnumpy(rearrange(global_se_map_mit, "b c h w -> b h w c")[0])

    global_se_map_gt_np= asnumpy(rearrange(global_se_map_gt, "b c h w -> b h w c")[0])

    global_se_seen_map_np = asnumpy(rearrange(global_se_seen_map, "b c h w -> b h w c")[0])

    global_object_map_np = asnumpy(rearrange(global_object_map, "b c h w -> b h w c")[0])

    
    return global_seen_map_np, global_wall_map_np, global_se_map_np,global_se_global_se_filtered_map_np, global_se_map_mit_np,global_se_map_gt_np,global_se_seen_map_np,global_object_map_np

def displayMapDic(global_seen_map_np, global_wall_map_np, global_se_map_np,global_se_filtered_map_np, global_se_map_mit_np, global_se_map_gt_np,global_se_seen_map_np, global_object_map_np,global_wall_map_np_dilation):

    display_map(global_seen_map_np[:,:,0]- global_wall_map_np[:,:,0])
    display_map(global_object_map_np[:,:,0]- global_wall_map_np[:,:,0])
    display_map(global_se_map_np[:,:,0]- global_wall_map_np[:,:,0])
    display_map((global_se_seen_map_np[:,:,0])*(1 - global_wall_map_np_dilation[:,:,0] > 0) - global_wall_map_np[:,:,0])
    display_map((global_se_filtered_map_np[:,:,0 ] > 0.9)*(1 - global_wall_map_np_dilation[:,:,0] > 0) - global_wall_map_np[:,:,0])
    display_map(global_se_map_mit_np[:,:,0] - global_wall_map_np[:,:,0])
    display_map((global_se_map_gt_np[:,:,0])*(1 - global_wall_map_np_dilation[:,:,0] > 0) - global_wall_map_np[:,:,0])

def iou(pre,lab):
    intersectionVal = (pre*lab > 0).sum()
    unionVal = ((pre+lab) > 0).sum()
    print(intersectionVal)
    print(unionVal)
    return np.array([intersectionVal,unionVal])

def visualize_result(pred, imgPath, colors, index=None):
    pred = pred.cpu()[0].numpy()
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    mitImage = PIL.Image.fromarray(pred_color)
    mitImage.save(imgPath +str(idx)+'/mitImage_'+ str(idx) + '.png')
    for index in [19,30,31,75]:
        mask = (pred == index)
        maskImage = PIL.Image.fromarray(pred_color*mask[:,:, np.newaxis])
        maskImage.save(imgPath + str(idx)+'/mitMaskImage_'+ str(index) + '.png')


step = 3
episode = 10
dilation_mask = np.ones((10, 10))
scenelist=[]
with open("data/datasets/exploration/se_"+objectName+"_data/v1/eval/scene_list", "r") as f:
    for line in f:
        scenelist += [line[:-1]]
iouTotal = []
for scene in range(len(scenelist)):
    _ = env.reset()
    env.habitat_env.episode_iterator._forced_scene_switch()
    iouScene = []
    scene_name = env.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]
    for i in tqdm.tqdm(range(episode)):

        global_seen_map_np, global_wall_map_np, global_se_map_np,global_se_filtered_map_np, global_se_map_mit_np, global_se_map_gt_np,global_se_seen_map_np, global_object_map_np = get_eval_map(env, trainer, 2001,depth_projection_net, step, objectName)
        global_wall_map_np_dilation = cv2.dilate(
                        global_wall_map_np.astype(np.float32), dilation_mask, iterations=1,
                    ).astype(np.float32)
        # displayMapDic(global_seen_map_np, global_wall_map_np, global_se_map_np,global_se_filtered_map_np, global_se_map_mit_np, global_se_map_gt_np,global_se_seen_map_np, global_object_map_np,global_wall_map_np_dilation)            

        ground_truth_label = (global_se_map_gt_np[:,:,0])*(1 - global_wall_map_np_dilation[:,:,0] > 0)

        pre_anti_seen = (global_se_seen_map_np[:,:,0 ] > 0)*(1 - global_wall_map_np_dilation[:,:,0] > 0)

        pre_anti = (global_se_filtered_map_np[:,:,0 ] > 0)*(1 - global_wall_map_np_dilation[:,:,0] > 0)

        pre_mit = (global_se_map_mit_np[:,:,0 ] > 0)*(1 - global_wall_map_np_dilation[:,:,0] > 0)

        pre_labeled_seg =  (global_object_map_np[:,:,0 ] > 0)*(1 - global_wall_map_np_dilation[:,:,0] > 0)

        iouEp = {
            "episode_id": env.habitat_env.current_episode.episode_id,
            "ratio_anti_seen": iou(pre_anti_seen,ground_truth_label),
            "ratio_anti": iou(pre_anti,ground_truth_label),
            "ratio_mit": iou(pre_mit,ground_truth_label),
            "ratio_labeled_seg": iou(pre_labeled_seg, ground_truth_label)
        }

        iouScene += [iouEp]
        mapsDic = {
            "global_seen_map_np":global_seen_map_np, 
            "global_wall_map_np":global_wall_map_np,
            "global_se_map_np":global_se_map_np,
            "global_se_filtered_map_np":global_se_filtered_map_np,
            "global_se_map_mit_np":global_se_map_mit_np,
            "global_se_map_gt_np":global_se_map_gt_np,
            "global_se_seen_map_np":global_se_seen_map_np,
            "global_object_map_np":global_object_map_np,
            "global_wall_map_np_dilation": global_wall_map_np_dilation
        }
        
        mapPath = "./data/debug/data/" + objectName +"/" + scene_name + '_' + env.habitat_env.current_episode.episode_id +"/" + "ep" + str(i) + "/np_file/"
        safe_mkdir(mapPath)
        save_episode_info(env, mapPath + "epInfo")
        for key in mapsDic.keys():
            np.save(mapPath + key, mapsDic[key])
            
        dataPath = './data/debug/data/' + objectName + "/" + scene_name + '_' + env.habitat_env.current_episode.episode_id + '/' + "record/"
        imgPath = './data/debug/img/' + objectName + "/" + scene_name + '_' + env.habitat_env.current_episode.episode_id +"/" + "ep" + str(i) + '/'
        file = open(dataPath + "obj_pos.pkl",'rb')
        obj_pos = pickle.load(file)
        filelist = os.listdir(dataPath)

        for id in range(len(filelist) // 2 ):
            idx = id+1
            se_filtered_map = torch.load(dataPath + 'se_filtered_map_'+str(idx)+'.pt')
            pu_inputs_t = torch.load(dataPath + 'pu_inputs_t_'+str(idx)+'.pt')
            rbg_numpy = asnumpy(pu_inputs_t["rgb"][0])/255
            semantic_numpy = asnumpy(pu_inputs_t["semantic"][0]) % 40
            indexes = np.unique(semantic_numpy)
            if pu_inputs_t["ego_map_gt_anticipated"][0,:,:,0].sum() > 0:
                safe_mkdir(imgPath + str(idx))
                visualize_result(pu_inputs_t['mit'],imgPath, colors)
                plt.imsave(imgPath + str(idx)+'/predict_'+ str(idx) + '.png', asnumpy(se_filtered_map[0,0]))
                plt.imsave(imgPath + str(idx)+'/ego_map_gt_anticipated_'+ str(idx) + '.png', asnumpy(pu_inputs_t["ego_map_gt_anticipated"][0,:,:,0]))
                plt.imsave(imgPath + str(idx)+'/rbg_image_'+ str(idx) + '.png', rbg_numpy)
                plt.imsave(imgPath + str(idx)+'/ego_map_gt_'+ str(idx) + '.png', asnumpy(pu_inputs_t["ego_map_gt"][0,:,:,0]))
                plt.imsave(imgPath + str(idx)+'/depth_image_'+ str(idx) + '.png', asnumpy(np.squeeze(pu_inputs_t["depth"][0,:,:,0])))
                plt.imsave(imgPath + str(idx)+'/object_mask_'+ str(idx) + '.png', asnumpy(np.squeeze(pu_inputs_t["object_mask_gt"][0,:,:,0])))      
                plt.imsave(imgPath + str(idx)+'/ego_map_ob_gt_'+ str(idx) + '.png', asnumpy(np.squeeze(pu_inputs_t["ego_map_gt_object"][0,0,:,:]))) 
                plt.imsave(imgPath + str(idx)+'/ego_map_ob_mit_'+ str(idx) + '.png', asnumpy(np.squeeze(pu_inputs_t["ego_map_mit_object"][0,0,:,:]))) 

        safe_mkdir('./data/debug/data/' + objectName + '/' + scene_name + '_' + env.habitat_env.current_episode.episode_id + '/iouscene/')
        np.save('./data/debug/data/' + objectName + '/' + scene_name + '_' + env.habitat_env.current_episode.episode_id + '/iouscene/' +'iou' + str(i) + '.pkl', iouScene)
        
    iouTotal += [iouScene]
    np.save('./data/debug/data/' + objectName + '/iouTotal.pkl', iouTotal)