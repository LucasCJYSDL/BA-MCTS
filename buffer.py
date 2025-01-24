import torch
import numpy as np
from numpy.random import default_rng
from collections import namedtuple
from torch.utils.data import Dataset

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'real_done'))
SL_Transition = namedtuple('SL_Transition', ('state', 'action_list', 'action_num', 'action_dist', 'q'))

class EnsembleTransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter, n_models=1):
        state_action_filtered, delta_filtered = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter)
        data_count = state_action_filtered.shape[0]
        # idxs = np.arange(0,data_count)[None,:].repeat(n_models, axis=0)
        # [np.random.shuffle(row) for row in idxs]
        idxs = np.random.randint(data_count, size=[n_models, data_count])
        self._n_models = n_models
        self.data_X = torch.Tensor(state_action_filtered[idxs])
        self.data_y = torch.Tensor(delta_filtered[idxs])
        self.data_r = torch.Tensor(np.array(batch.reward)[idxs])

    def __len__(self):
        return self.data_X.shape[1]

    def __getitem__(self, index):
        return self.data_X[:, index], self.data_y[:, index], self.data_r[:, index]

def prepare_data(state, action, nextstate, state_filter, action_filter):
    state_filtered = state_filter.filter(state)
    action_filtered = action_filter.filter(action)
    state_action_filtered = np.concatenate((state_filtered, action_filtered), axis=1)
    delta = np.array(nextstate) - np.array(state)
    return state_action_filtered, delta

class TransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter):
        state_action_filtered, delta_filtered = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter)
        self.data_X = torch.Tensor(state_action_filtered)
        self.data_y = torch.Tensor(delta_filtered)
        self.data_r = torch.Tensor(np.array(batch.reward))

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index], self.data_r[index]
    
    
class SLFasterReplayPool:
    def __init__(self, action_dim, state_dim, num_sampled_actions, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._num_sampled_actions = num_sampled_actions
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()
    
    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action_list': np.zeros((self.capacity, self._num_sampled_actions, self._action_dim), dtype='float32'),
            'action_num': np.zeros((self.capacity, ), dtype='int'),
            'action_dist': np.zeros((self.capacity, self._num_sampled_actions), dtype='float32'),
            'q': np.zeros((self.capacity, self._num_sampled_actions), dtype='float32')
        } # numpy is not allowed since they don't have the same shape
    
    def push(self, transition: SL_Transition):
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        temp_dict = transition._asdict()

        for i in range(num_samples):
            i_idx = idx[i]
            i_action_num = len(temp_dict['action_list'][i])
            self._memory['action_num'][i_idx] = i_action_num
            self._memory['action_dist'][i_idx][:i_action_num] = temp_dict['action_dist'][i]
            self._memory['action_list'][i_idx][:i_action_num] = temp_dict['action_list'][i]
            self._memory['q'][i_idx][:i_action_num] = temp_dict['q'][i]
        
        self._memory['state'][idx] = temp_dict['state']

        self._memory['action_dist'][idx] = self._memory['action_dist'][idx] / self._memory['action_dist'][idx].sum(axis=1, keepdims=True)

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)
    
    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return SL_Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        # print(self._size, self.capacity) # 160456, 800000
        return self._return_from_idx(idx)

    

class FasterReplayPool:
    def __init__(self, action_dim, state_dim, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()

    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action': np.zeros((self.capacity, self._action_dim), dtype='float32'),
            'reward': np.zeros((self.capacity), dtype='float32'),
            'nextstate': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'real_done': np.zeros((self.capacity), dtype='bool')
        }

    def push(self, transition: Transition):
        # Handle 1-D Data
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        for key, value in transition._asdict().items():
            self._memory[key][idx] = value

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        # print(self._size, self.capacity) # 160456, 800000
        return self._return_from_idx(idx)

    def sample_all(self):
        return self._return_from_idx(np.arange(0, self._size))

    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def initialise(self, old_pool):
        # Not Tested
        old_memory = old_pool.sample_all()
        for key in self._memory:
            self._memory[key] = np.append(self._memory[key], old_memory[key], 0)