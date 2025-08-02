import os, random
from typing import Union

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs.common import save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.mujoco.envs.mujoco_env import MujocoEnv

import d4rl


@ENV_REGISTRY.register('offline_mujoco_lightzero')
class OfflineMujocoEnvLightZero(MujocoEnv):
    """
    Overview:
        The modified MuJoCo environment based on the offline dataset provided by D4RL for LightZero's algorithms.
    """

    config = dict(
        stop_value=int(1e6),
        action_clip=False,
        delay_reward_step=0,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) If True, save the replay as a gif file.
        save_replay_gif=False,
        # (str or None) The path to save the replay gif. If None, the replay gif will not be saved.
        replay_path_gif=None,
        action_bins_per_branch=None,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
    )

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the MuJoCo environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict. The dict should include keys like 'env_id', 'replay_path', etc.
        """
        super().__init__(cfg)

        self._cfg = cfg
        # We use env_id to indicate the env_id in LightZero.
        self._cfg.env_id = self._cfg.env_id
        self._action_clip = cfg.action_clip
        self._delay_reward_step = cfg.delay_reward_step
        self._init_flag = False
        self._replay_path = None
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif
        self._action_bins_per_branch = cfg.action_bins_per_branch

    def _preprocess_data(self, env):
        dataset = env.get_dataset() # a d4rl API
        N = dataset['rewards'].shape[0]
        obs_ls = []
        next_obs_ls = []
        action_ls = []
        reward_ls = []
        done_ls = []
        len_ls = []

        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True
        
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []
        episode_step = 0

        for i in range(N-1):
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i+1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])

            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1

            if done_bool or final_timestep:
                obs_ls.append(obs_.copy())
                next_obs_ls.append(next_obs_.copy())
                action_ls.append(action_.copy())
                reward_ls.append(reward_.copy())
                done_ls.append(done_.copy())
                len_ls.append(len(reward_.copy()))

                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []
                episode_step = 0
        
        return obs_ls, action_ls, reward_ls, next_obs_ls, done_ls, len_ls

    
    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): The initial observation after resetting, which also includes the stored intial actions.
        """
        if not self._init_flag:
            self._env = self._make_env()
            # change
            # self._dataset = d4rl.qlearning_dataset(self._env)
            self.obs_ls, self.action_ls, self.reward_ls, self.next_obs_ls, self.done_ls, self.len_ls = self._preprocess_data(self._env)
            self.traj_num = len(self.obs_ls)
            # print(self.traj_num, self.len_ls) # 1249

            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )

            self._env.observation_space.dtype = np.float32
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            random.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            random.seed(self._seed)

        # obs = self._env.reset()
        self.traj_id = random.randint(0, self.traj_num-1)
        obs = self.obs_ls[self.traj_id][0]
        action = self.action_ls[self.traj_id][0]
        self.time_step = 0

        obs = to_ndarray(obs).astype('float32')
        action = to_ndarray(action).astype('float32')

        self._eval_episode_return = 0.

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1, "action": action}

        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        # the input action is not used
        # if self._action_bins_per_branch: # False
        #     action = self.map_action(action)
        # action = to_ndarray(action)
        # if self._save_replay_gif:
        #     self._frames.append(self._env.render(mode='rgb_array')) # just keeping the original code
        # if self._action_clip:
        #     action = np.clip(action, -1, 1)

        obs = self.next_obs_ls[self.traj_id][self.time_step]
        rew = self.reward_ls[self.traj_id][self.time_step]
        done = self.done_ls[self.traj_id][self.time_step]
        action = self.action_ls[self.traj_id][self.time_step]
        info = {}
        
        self.time_step += 1
        # obs, rew, done, info = self._env.step(action)
        if self.time_step >= self.len_ls[self.traj_id]:
            done = True
        
        self._eval_episode_return += rew
        if done:
            # if self._save_replay_gif:
            #     path = os.path.join(
            #         self._replay_path_gif, '{}_episode_{}.gif'.format(self._cfg.env_id, self._save_replay_count)
            #     )
            #     save_frames_as_gif(self._frames, path)
            #     self._save_replay_count += 1
            info['eval_episode_return'] = self._eval_episode_return

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)
        action = to_ndarray(action).astype(np.float32)

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1, "action": action}

        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "Offline LightZero Mujoco Env({})".format(self._cfg.env_id)