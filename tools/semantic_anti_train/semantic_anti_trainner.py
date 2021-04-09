import os
import cv2
import time
import json
import h5py
import glob
import itertools
import gym.spaces as spaces
from collections import defaultdict, deque
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.models as tmodels
import numpy as np
from typing import Any, Dict, List, Optional

from rl.models.unet import (
    UNetEncoder,
    UNetDecoder,
    MiniUNetEncoder,
    LearnedRGBProjection,
    MergeMultimodal,
    ResNetRGBEncoder,
)
from rl.common.rollout_storage import (
    RolloutStorageExtended,
    MapLargeRolloutStorageMP,
)

import habitat
import habitat_extensions
from habitat import Config, logger

from einops import rearrange, asnumpy
from config.default import get_config

from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.utils import (
    batch_obs,
)

from rl.models.mapnet import DepthProjectionNet
from rl.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class


from mapper.mapper import Mapper
from mapper.map_update import MapUpdate
from occant_utils.map import (
    add_pose,
    dilate_tensor,
)

from rl.models.semantic_anticipator import(
    BaseModel,
    SemAntRGBD,
    SemAntGroundTruth,
    SemAnticipator,
    SemAnticipationWrapper,
)

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

class SemAntExpTrainer(BaseRLTrainer):
    r"""Trainer class for Occupancy Anticipated based exploration algorithm.
    """
    supported_tasks = ["Exp-v0"]
    frozen_mapper_types = ["ans_depth", "occant_ground_truth"]

    def __init__(self, config=None):
        if config is not None:
            self._synchronize_configs(config)
        super().__init__(config)

        # Set pytorch random seed for initialization
        torch.manual_seed(config.PYT_RANDOM_SEED)

        # initial the mapper and mapper_agent for training object

        self.mapper = None
        self.mapper_agent = None

        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

    def _synchronize_configs(self, config):
        r"""Matches configs for different parts of the model as well as the simulator. 
        """
        config.defrost()
        # Compute the EGO_PROJECTION options based on the
        # depth sensor information and agent parameters.
        map_size = config.RL.ANS.MAPPER.map_size
        map_scale = config.RL.ANS.MAPPER.map_scale
        min_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        hfov_rad = np.radians(float(hfov))
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
        vfov = np.degrees(vfov_rad).item()
        camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        height_thresholds = [0.2, 1.5]
        # Set the EGO_PROJECTION options
        ego_proj_config = config.RL.ANS.SEMANTIC_ANTICIPATOR.EGO_PROJECTION
        ego_proj_config.local_map_shape = (2, map_size, map_size)
        ego_proj_config.map_scale = map_scale
        ego_proj_config.min_depth = min_depth
        ego_proj_config.max_depth = max_depth
        ego_proj_config.hfov = hfov
        ego_proj_config.vfov = vfov
        ego_proj_config.camera_height = camera_height
        ego_proj_config.height_thresholds = height_thresholds
        config.RL.ANS.SEMANTIC_ANTICIPATOR.EGO_PROJECTION = ego_proj_config
        # Set the GT anticipation options
        wall_fov = config.RL.ANS.SEMANTIC_ANTICIPATOR.GP_ANTICIPATION.wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.WALL_FOV = wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SIZE = map_size
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SCALE = map_scale
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAX_SENSOR_RANGE = -1
        config.freeze()

    def _setup_anticipator(self, ppo_cfg: Config, ans_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params
            ans_cfg: config node for ActiveNeuralSLAM model

        Returns:
            None
        """
        try:
            os.mkdir(self.config.TENSORBOARD_DIR)
        except:
            pass

        logger.add_filehandler(os.path.join(self.config.TENSORBOARD_DIR, "run.log"))

        sem_cfg = ans_cfg.SEMANTIC_ANTICIPATOR
        mapper_cfg = ans_cfg.MAPPER
        # Create occupancy anticipation model
        [imgh, imgw] = ans_cfg['image_scale_hw']
        sem_model = SemAnticipationWrapper(
            SemAnticipator(sem_cfg), mapper_cfg.map_size, (imgh, imgw)
        )

        self.mapper = Mapper(mapper_cfg,sem_model)

        self.mapper_agent = MapUpdate(
            self.mapper,
            lr=mapper_cfg.lr,
            eps=mapper_cfg.eps,
            label_id=mapper_cfg.label_id,
            max_grad_norm=mapper_cfg.max_grad_norm,
            pose_loss_coef=mapper_cfg.pose_loss_coef,
            semantic_anticipator_type=ans_cfg.SEMANTIC_ANTICIPATOR.type,
            freeze_projection_unit=mapper_cfg.freeze_projection_unit,
            num_update_batches=mapper_cfg.num_update_batches,
            batch_size=mapper_cfg.map_batch_size,
            mapper_rollouts=self.mapper_rollouts,
        )

        if ans_cfg.model_path != "":
            self.resume_checkpoint(ans_cfg.model_path)

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "mapper_state_dict": self.mapper_agent.state_dict(),
            "mapper": self.mapper_agent.mapper.state_dict(),
            "config": self.config,
        }

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def resume_checkpoint(self, path=None):
        r"""If an existing checkpoint already exists, resume training.
        """
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        ppo_cfg = self.config.RL.PPO
        if path is None:
            if len(checkpoints) == 0:
                num_updates_start = 0
                count_steps = 0
                count_checkpoints = 0
            else:
                # Load lastest checkpoint
                last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
                checkpoint_path = last_ckpt
                # Restore checkpoints to models
                ckpt_dict = self.load_checkpoint(checkpoint_path)
                self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
                # Set the logging counts
                ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
                num_updates_start = ckpt_dict["extra_state"]["update"] + 1
                count_steps = ckpt_dict["extra_state"]["step"]
                count_checkpoints = ckpt_id + 1
                print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")
        else:
            print(f"Loading pretrained model!")
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(path)
            self.sem_model.load_state_dict(ckpt_dict["mapper_state_dict"])
            num_updates_start = 0
            count_steps = 0
            count_checkpoints = 0

        return num_updates_start, count_steps, count_checkpoints

    def _create_mapper_rollout_inputs(
        self, prev_batch, batch,
    ):
        ans_cfg = self.config.RL.ANS
        mapper_rollout_inputs = {
            # reduce memory consumption
            # "rgb_at_t_1": prev_batch["rgb"],
            # "depth_at_t_1": prev_batch["depth"],
            # "ego_map_gt_at_t_1": prev_batch["ego_map_gt"],
            # "pose_at_t_1": prev_batch["pose"],
            # "pose_gt_at_t_1": prev_batch["pose_gt"],
            
            "rgb_at_t": batch["rgb"],
            "depth_at_t": batch["depth"],
            "ego_map_gt_at_t": batch["ego_map_gt"],
            "ego_map_gt_dilation_at_t": batch["ego_map_dilation_gt"],
            "pose_at_t": batch["pose"],
            "pose_gt_at_t": batch["pose_gt"],
            "ego_map_gt_anticipated_at_t": batch["ego_map_gt_anticipated"],
        }
        
        return mapper_rollout_inputs

    def _collect_rollout_step(
        self,
        batch,
        prev_batch,
        episode_step_count,
        state_estimates,
        ground_truth_states,
        masks,
        mapper_rollouts,
    ):
        pth_time = 0.0
        env_time = 0.0

        device = self.device
        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS

        NUM_LOCAL_STEPS = ppo_cfg.num_local_steps

        for t in range(NUM_LOCAL_STEPS):

            #print(f'===> Local time: {t}, episode time: {episode_step_count[0].item()}')
            # --------------------- update mapper rollout stats -----------------------
            t_update_stats = time.time()

            mapper_rollout_inputs = self._create_mapper_rollout_inputs(
                prev_batch, batch
            )
            mapper_rollouts.insert(mapper_rollout_inputs)

            pth_time += time.time() - t_update_stats

            # ---------------------- execute environment action -----------------------
            t_step_env = time.time()

            # random agent
            functionnames = ["get_random_pos"]*self.envs.num_envs
            args = self.envs.call(functionnames)
            functionnames = ["get_obs_at"]*self.envs.num_envs
            _ = self.envs.call(functionnames,args)
            actions = [{"action":"TURN_LEFT"}]*self.envs.num_envs
            outputs = self.envs.step(actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            env_time += time.time() - t_step_env

            # -------------------- update ground-truth states -------------------------
            t_update_stats = time.time()

            masks.copy_(
                torch.tensor(
                    [[0.0] if done else [1.0] for done in dones], dtype=torch.float
                )
            )
            # Sanity check
            assert episode_step_count[0].item() <= self.config.T_EXP - 1
            assert not dones[0], "DONE must not be called during training"

            del prev_batch
            prev_batch = batch
            batch = self._prepare_batch(
                observations, prev_batch=prev_batch, device=device, actions=actions
            )

            pth_time += time.time() - t_update_stats
            
            episode_step_count += 1

        return (
            pth_time,
            env_time,
            self.envs.num_envs * NUM_LOCAL_STEPS,
            prev_batch,
            batch,
            state_estimates,
            ground_truth_states,
        )

    def _update_mapper_agent(self, mapper_rollouts):
        t_update_model = time.time()

        losses = self.mapper_agent.update(mapper_rollouts)

        return time.time() - t_update_model, losses


    def _assign_devices(self):
        # Assign devices for the simulator
        if len(self.config.SIMULATOR_GPU_IDS) > 0:
            devices = self.config.SIMULATOR_GPU_IDS
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            devices = [int(dev) for dev in visible_devices]
            # Devices need to be indexed between 0 to N-1
            devices = [dev for dev in range(len(devices))]
            if len(devices) > 1:
                devices = devices[1:]
        else:
            devices = None
        return devices

    def _create_mapper_rollouts(self, ans_cfg):

        V = ans_cfg.MAPPER.map_size
        imH, imW = ans_cfg.image_scale_hw

        mapper_observation_space = {
            "rgb_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "depth_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 1), dtype=np.float32
            ),
            "ego_map_gt_at_t": spaces.Box(
                low=0.0, high=1.0, shape=(V, V, 2), dtype=np.float32
            ),
            "ego_map_gt_dilation_at_t": spaces.Box(
                low=0.0, high=1.0, shape=(V, V, 2), dtype=np.float32
            ),
            "pose_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_gt_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "ego_map_gt_anticipated_at_t": self.envs.observation_spaces[0].spaces[
                "ego_map_gt_anticipated"
            ],
        }

        mapper_observation_space = spaces.Dict(mapper_observation_space)

        # Multiprocessing manager
        mapper_manager = mp.Manager()
        mapper_device = self.device
        if ans_cfg.MAPPER.use_data_parallel and len(ans_cfg.MAPPER.gpu_ids) > 0:
            mapper_device = ans_cfg.MAPPER.gpu_ids[0]
            
        mapper_rollouts = MapLargeRolloutStorageMP(
            ans_cfg.MAPPER.replay_size,
            mapper_observation_space,
            mapper_device,
            mapper_manager,
        )

        return mapper_rollouts

    def _prepare_batch(self, observations, prev_batch=None, device=None, actions=None):
        imH, imW = self.config.RL.ANS.image_scale_hw
        device = self.device if device is None else device
        batch = batch_obs(observations, device=device)
        if batch["rgb"].size(1) != imH or batch["rgb"].size(2) != imW:
            rgb = rearrange(batch["rgb"], "b h w c -> b c h w")
            rgb = F.interpolate(rgb, (imH, imW), mode="bilinear")
            batch["rgb"] = rearrange(rgb, "b c h w -> b h w c")
        if batch["depth"].size(1) != imH or batch["depth"].size(2) != imW:
            depth = rearrange(batch["depth"], "b h w c -> b c h w")
            depth = F.interpolate(depth, (imH, imW), mode="nearest")
            batch["depth"] = rearrange(depth, "b c h w -> b h w c")
        # Compute ego_map_gt from depth
        ego_map_gt_b = self.depth_projection_net(
            rearrange(batch["depth"], "b h w c -> b c h w")
        )
        batch["ego_map_gt"] = rearrange(ego_map_gt_b, "b c h w -> b h w c")
        #  Compute ego_map_dilation_gt from ego_map_gt
        
        batch["ego_map_dilation_gt"] = rearrange(dilate_tensor(ego_map_gt_b,31), "b c h w -> b h w c")
        
        
        if actions is None:
            # Initialization condition
            # If pose estimates are not available, set the initial estimate to zeros.
            if "pose" not in batch:
                # Set initial pose estimate to zero
                batch["pose"] = torch.zeros(self.envs.num_envs, 3).to(self.device)
            batch["prev_actions"] = torch.zeros(self.envs.num_envs, 1).to(self.device)
        else:
            # Rollouts condition
            # If pose estimates are not available, compute them from action taken.
            if "pose" not in batch:
                assert prev_batch is not None
                actions_delta = self._convert_actions_to_delta(actions)
                batch["pose"] = add_pose(prev_batch["pose"], actions_delta)
            batch["prev_actions"] = actions

        return batch

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            devices=self._assign_devices(),
        )

        # set up configurations
        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS

        # set up device
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # set up checkpoint folder
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        
        # create rollouts for mapper
        self.mapper_rollouts = self._create_mapper_rollouts(ans_cfg)

        self.depth_projection_net = DepthProjectionNet(
            ans_cfg.SEMANTIC_ANTICIPATOR.EGO_PROJECTION
        )

        self._setup_anticipator(ppo_cfg, ans_cfg)

        logger.info(
            "mapper_agent number of parameters: {}".format(
                sum(param.numel() for param in self.mapper_agent.parameters())
            )
        )


        mapper_rollouts = self.mapper_rollouts

        # ===================== Create statistics buffers =====================
        statistics_dict = {}
        # Mapper statistics
        statistics_dict["mapper"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )

        # Overall count statistics
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        # ==================== Measuring memory consumption ===================
        total_memory_size = 0
        print("=================== Mapper rollouts ======================")
        for k, v in mapper_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")

        # Resume checkpoint if available
        (
            num_updates_start,
            count_steps_start,
            count_checkpoints,
        ) = self.resume_checkpoint()
        count_steps = count_steps_start

        imH, imW = ans_cfg.image_scale_hw
        M = ans_cfg.overall_map_size
        # ==================== Create state variables =================
        state_estimates = {
            # Agent's pose estimate
            "pose_estimates": torch.zeros(self.envs.num_envs, 3).to(self.device),
            # Agent's map
            "map_states": torch.zeros(self.envs.num_envs, 2, M, M).to(self.device),
            "recurrent_hidden_states": torch.zeros(
                1, self.envs.num_envs, ans_cfg.LOCAL_POLICY.hidden_size
            ).to(self.device),
            "visited_states": torch.zeros(self.envs.num_envs, 1, M, M).to(self.device),
        }

        ground_truth_states = {
            # To measure area seen
            "visible_occupancy": torch.zeros(
                self.envs.num_envs, 2, M, M, device=self.device
            ),
            "pose": torch.zeros(self.envs.num_envs, 3, device=self.device),
            "prev_global_reward_metric": torch.zeros(
                self.envs.num_envs, 1, device=self.device
            ),
        }

        masks = torch.zeros(self.envs.num_envs, 1)
        episode_step_count = torch.zeros(self.envs.num_envs, 1, device=self.device)

        # ==================== Reset the environments =================
        observations = self.envs.reset()
        batch = self._prepare_batch(observations)
        prev_batch = batch

        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            local_reward=torch.zeros(self.envs.num_envs, 1),
            global_reward=torch.zeros(self.envs.num_envs, 1),
        )
        
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        # Useful variables
        NUM_MAPPER_STEPS = ans_cfg.MAPPER.num_mapper_steps
        NUM_LOCAL_STEPS = ppo_cfg.num_local_steps
        NUM_GLOBAL_STEPS = ppo_cfg.num_global_steps
        GLOBAL_UPDATE_INTERVAL = NUM_GLOBAL_STEPS * ans_cfg.goal_interval
        NUM_GLOBAL_UPDATES_PER_EPISODE = self.config.T_EXP // GLOBAL_UPDATE_INTERVAL
        NUM_GLOBAL_UPDATES = (
            self.config.NUM_EPISODES
            * NUM_GLOBAL_UPDATES_PER_EPISODE
            // self.config.NUM_PROCESSES
        )
        # Sanity checks
        assert (
            NUM_MAPPER_STEPS % NUM_LOCAL_STEPS == 0
        ), "Mapper steps must be a multiple of global steps interval"
        assert (
            NUM_LOCAL_STEPS == ans_cfg.goal_interval
        ), "Local steps must be same as subgoal sampling interval"
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(num_updates_start, NUM_GLOBAL_UPDATES):
                for step in range(NUM_GLOBAL_STEPS):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        prev_batch,
                        batch,
                        state_estimates,
                        ground_truth_states,
                    ) = self._collect_rollout_step(
                        batch,
                        prev_batch,
                        episode_step_count,
                        state_estimates,
                        ground_truth_states,
                        masks,
                        mapper_rollouts,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                    # Useful flags
                    UPDATE_MAPPER_FLAG = (
                        True
                        if episode_step_count[0].item() % NUM_MAPPER_STEPS == 0
                        else False
                    )


                    # ------------------------ update mapper --------------------------
                    if UPDATE_MAPPER_FLAG:
                        (
                            delta_pth_time,
                            update_metrics_mapper,
                        ) = self._update_mapper_agent(mapper_rollouts)

                        for k, v in update_metrics_mapper.items():
                            statistics_dict["mapper"][k].append(v)

                    pth_time += delta_pth_time

                    # -------------------------- log statistics -----------------------
                    for k, v in statistics_dict.items():
                        logger.info(
                            "=========== {:20s} ============".format(k + " stats")
                        )
                        for kp, vp in v.items():
                            if len(vp) > 0:
                                writer.add_scalar(f"{k}/{kp}", np.mean(vp), count_steps)
                                logger.info(f"{kp:25s}: {np.mean(vp).item():10.5f}")

                    for k, v in running_episode_stats.items():
                        window_episode_stats[k].append(v.clone())

                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }

                    deltas["count"] = max(deltas["count"], 1.0)


                    fps = (count_steps - count_steps_start) / (time.time() - t_start)
                    writer.add_scalar("fps", fps, count_steps)

                    if update > 0:
                        logger.info("update: {}\tfps: {:.3f}\t".format(update, fps))

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(update, env_time, pth_time, count_steps)
                        )

                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )

                    pth_time += delta_pth_time

                # At episode termination, manually set masks to zeros.
                if episode_step_count[0].item() == self.config.T_EXP:
                    masks.fill_(0)

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(step=count_steps, update=update),
                    )
                    count_checkpoints += 1

                if episode_step_count[0].item() == self.config.T_EXP:
                    seed = int(time.time())
                    print('change random seed to {}'.format(seed))
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    observations = self.envs.reset()
                    batch = self._prepare_batch(observations)
                    prev_batch = batch
                    # Reset episode step counter
                    episode_step_count.fill_(0)
                    # Reset states
                    for k in ground_truth_states.keys():
                        ground_truth_states[k].fill_(0)
                    for k in state_estimates.keys():
                        state_estimates[k].fill_(0)

            self.envs.close()

if __name__ == "__main__":
    from config.default import get_config
    import random
    config = get_config("./tools/semantic_anti_train/config/chair_train_256.yaml")
    seed = int(time.time())
    print('change random seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = SemAntExpTrainer(config)
    trainer.train()