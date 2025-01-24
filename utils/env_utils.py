import os
import torch
import numpy as np
from pathlib import Path


class MeanStdevFilter():
    def __init__(self, shape, clip=10.0):
        self.eps = 1e-12
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = 1

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
        self.stdev[self.stdev <= self.eps] = 1.0

    def reset(self):
        self.__init__(self.shape, self.clip)

    def update_torch(self, device):
        self.torch_mean = torch.FloatTensor(self.mean).to(device)
        self.torch_stdev = torch.FloatTensor(self.stdev).to(device)

    def filter(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def filter_torch(self, x: torch.Tensor, device):
        self.update_torch(device)
        return torch.clamp(((x - self.torch_mean) / self.torch_stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean

    def invert_torch(self, x: torch.Tensor):
        return (x * self.torch_stdev) + self.torch_mean

def reward_func(s1, s2, a, env_name, state_filter=None, is_done_func=None):
    if state_filter:
        s1_real = s1 * state_filter.stdev + state_filter.mean
        s2_real = s2 * state_filter.stdev + state_filter.mean
    else:
        s1_real = s1
        s2_real = s2
    if env_name == "HalfCheetah-v2":
        return np.squeeze(s2_real)[-1] - 0.1 * np.square(a).sum()
    if env_name == "Ant-v2":
        if is_done_func:
            if is_done_func(torch.Tensor(s2_real).reshape(1,-1)):
                return 0.0
        return np.squeeze(s2_real)[-1] - 0.5 * np.square(a).sum() + 1.0
    if env_name == "Swimmer-v2":
        return np.squeeze(s2_real)[-1] - 0.0001 * np.square(a).sum()
    if env_name == "Hopper-v2":
        if is_done_func:
            if is_done_func(torch.Tensor(s2_real).reshape(1,-1)):
                return 0.0
        return np.squeeze(s2_real)[-1] - 0.1 * np.square(a).sum() - 3.0 * np.square(s2_real[0] - 1.3) + 1.0


def check_or_make_folder(folder_path):
    """
    Helper function that (safely) checks if a dir exists; if not, it creates it
    """
    
    folder_path = Path(folder_path)

    try:
        folder_path.resolve(strict=True)
    except FileNotFoundError:
        print("{} dir not found, creating it".format(folder_path))
        os.mkdir(folder_path)


def halfcheetah_reward(nextstate, action):
    return (nextstate[:,-1] - 0.1 * torch.sum(torch.pow(action, 2), 1)).detach().cpu().numpy()

def ant_reward(nextstate, action, dones):
    reward = (nextstate[:,-1] - 0.5 * torch.sum(torch.pow(action, 2), 1) + 1.0).detach().cpu().numpy()
    reward[dones] = 0.0
    return reward

def swimmer_reward(nextstate, action):
    reward = (nextstate[:,-1] - 0.0001 * torch.sum(torch.pow(action, 2), 1)).detach().cpu().numpy()
    return reward

def hopper_reward(nextstate, action, dones):
    reward = (nextstate[:,-1] - 0.1 * torch.sum(torch.pow(action, 2), 1) - 3.0 * (nextstate[:,0] - 1.3).pow(2) + 1.0).detach().cpu().numpy()
    reward[dones] = 0.0
    return reward

def torch_reward(env_name, nextstate, action, dones=None):
    if env_name == "HalfCheetah-v2":
        return halfcheetah_reward(nextstate, action)
    elif env_name == "Ant-v2":
        return ant_reward(nextstate, action, dones)
    elif env_name == "Hopper-v2":
        return hopper_reward(nextstate, action, dones)
    elif env_name == "Swimmer-v2":
        return swimmer_reward(nextstate, action)
    else:
        raise Exception('Environment not supported')

