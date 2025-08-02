import argparse
import os
import random
import datetime

import gym
import numpy as np
import yaml
import torch

from ensemble import EnsembleGymEnv
from utils.done_funcs import hopper_is_done_func, walker2d_is_done_func, ant_is_done_func
from sac import SAC_Agent
from runner import Runner
from searcher import Searcher
from buffer import SLFasterReplayPool
from utils.search_utils import LinearParameter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def seed_everything(env, eval_env_vectors, params):
    eval_seed = params['seed'] + 1
    env.real_env.seed(params['seed'])
    env.eval_env.seed(eval_seed)
    env.real_env.action_space.seed(params['seed'])
    env.eval_env.action_space.seed(eval_seed)
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # optional
    # i = 0
    # for _env in eval_env_vectors:
    #     _env.seed(eval_seed + i)
    #     _env.action_space.seed(eval_seed + i)
    #     i += 100


def train_agent(params):
    # prepare the arguments
    params['zeros'] = False # IDK what this is for
    env = gym.make(params['env_name'])
    eval_env = gym.make(params['env_name'])
    eval_env_vectors = [gym.make(params['env_name']) for _ in range(params['n_eval_rollouts'])]

    env_name_lower = params['env_name'].lower()
    if 'hopper' in env_name_lower:
        params['is_done_func'] = hopper_is_done_func
    elif 'walker' in env_name_lower:
        params['is_done_func'] = walker2d_is_done_func
    elif 'ant' in env_name_lower:
        params['is_done_func'] = ant_is_done_func
    else:
        params['is_done_func'] = None
    
    state_dim = params['ob_dim'] = env.observation_space.shape[0]
    action_dim = params['ac_dim'] = env.action_space.shape[0]
    params['device'] = torch.device("cuda:{}".format(params['cuda_id']) if torch.cuda.is_available() else "cpu")
    params['time_stamp'] = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')

    # env
    env = EnsembleGymEnv(params, env, eval_env)
    seed_everything(env, eval_env_vectors, params)

    # buffer 
    if isinstance(params['steps_k'], list):
        init_steps_k = params['steps_k'][0]
    else:
        init_steps_k = params['steps_k']
    steps_per_epoch = params['epoch_steps'] if params['epoch_steps'] else env._max_timesteps

    # RL agent
    init_buffer_size = init_steps_k * params['num_rollouts_per_step'] * steps_per_epoch * params['model_retain_epochs'] #!!!
    print('Initial Buffer Size: {} using model_retain_epochs={}'.format(init_buffer_size, params['model_retain_epochs']))

    if params['train_mode'] != 0:
        entropy_coe_scheduler = LinearParameter(start=0.01, end=0.001, num_steps=params['offline_epochs'])
    else:
        entropy_coe_scheduler = None
 
    agent = SAC_Agent(params['seed'], state_dim, action_dim, gamma=params['gamma'], buffer_size=init_buffer_size,
                      target_entropy=params['target_entropy'], device=params['device'], scheduler=entropy_coe_scheduler)

    # algorithm runner: model/policy learning, collection, evaluation
    if params['use_search']:
        searcher = Searcher(params, agent, env)
        sl_buffer_size = int(init_steps_k * params['num_rollouts_per_step'] * params['model_train_freq'] * params['model_retain_epochs_sl'] * params['search_ratio'])
        print('Initial Buffer Size for Supervised Learning: {}'.format(sl_buffer_size))
        sl_buffer = SLFasterReplayPool(action_dim=action_dim, state_dim=state_dim, num_sampled_actions=searcher._cfg.num_actions, capacity=sl_buffer_size)
    else:
        searcher = None
        sl_buffer = None

    runner = Runner(params, env, agent, searcher, sl_buffer, eval_env_vectors)
    
    # main entry
    runner.train_offline(params['offline_epochs'], save_model=params['save_model'], 
                         save_policy=params['save_policy'], load_model_dir=params['load_model_dir']) # main entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')  
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--num_models', '-nm', type=int, default=7) ## size of the ensemble
    parser.add_argument('--steps_k', '-sk', type=int,  default=5)  ## maximum time we step through an env to make artificial rollouts
    parser.add_argument('--max_timesteps', '-maxt', type=int, default=6000)  ## total number of timesteps
    parser.add_argument('--model_epochs', '-me', type=int, default=1000)  ## max number of times we improve model
    parser.add_argument('--gamma', '-gm', type=float, default=0.99)
    parser.add_argument('--lam', '-la', type=float, default=0)
    parser.add_argument('--dir', '-d', type=str, default='data')
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--uuid', '-id', type=str, default=None)
    parser.add_argument('--reward_head', '-rh', type=int, default=1)  # 1 or 0
    parser.add_argument('--policy_update_steps', type=int, default=40)
    parser.add_argument('--num_rollouts_per_step', type=int, default=400)
    parser.add_argument('--n_eval_rollouts', type=int, default=10)
    parser.add_argument('--train_val_ratio', type=float, default=0.2)
    parser.add_argument('--model_train_freq', type=int, default=250)
    parser.add_argument('--oac', type=bool, default=False)
    parser.add_argument('--var_thresh', type=float, default=100)
    parser.add_argument('--epoch_steps', type=int, default=None)
    parser.add_argument('--target_entropy', type=float, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--d4rl', dest='d4rl', action='store_true')
    parser.add_argument('--train_memory', type=int, default=800000)
    parser.add_argument('--val_memory', type=int, default=200000)
    parser.add_argument('--mopo_uncertainty_target', type=float, default=1.5)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--offline_epochs', type=int, default=1000)
    parser.add_argument('--rl_pretrain_epochs', type=int, default=-1)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--deterministic_rollouts', type=bool, default=False)
    # Needed as some models seem to early terminate (this happens in author's code too, so not a PyTorch issue)
    parser.add_argument('--min_model_epochs', type=int, default=None)
    parser.add_argument('--save_policy', type=bool, default=False)
    parser.add_argument('--l2_reg_multiplier', type=float, default=1.)
    parser.add_argument('--model_lr', type=float, default=0.001)
    # important ones
    parser.add_argument('--real_sample_ratio', type=float, default=0.05)
    parser.add_argument('--mopo_lam', type=float, default=1) # mopo style: applying reward penalty
    parser.add_argument('--model_retain_epochs', type=int, default=100)
    parser.add_argument('--model_retain_epochs_sl', type=int, default=5)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--use_search', type=bool, default=False)
    parser.add_argument('--use_search_eval', type=bool, default=False)
    parser.add_argument('--train_mode', type=int, default=0) # 0: sac_only, 1: sl_only, 2: both, 3: sl_only and using root values as targets
    parser.add_argument('--rl_lr_decay', type=bool, default=False)
    parser.add_argument('--search_ratio', type=float, default=0.1)
    parser.add_argument('--search_alpha', type=float, default=0.8)
    parser.add_argument('--search_ucb_coe', type=float, default=2.5)
    parser.add_argument('--search_root_alpha', type=float, default=0.3)
    parser.add_argument('--search_n_actions', type=int, default=20)
    parser.add_argument('--search_n_states', type=int, default=1)
    parser.set_defaults(logvar_head=True)
 
    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]

    # sanity check
    assert isinstance(params['steps_k'], (int, list)), "must be either a single input or a collection"

    if isinstance(params['steps_k'], list):
        assert len(params['steps_k']) == 4, "if a list of inputs, must have 4 inputs (start steps, end steps, start epoch, end epoch)"

    if not params['use_search']:
        print("No searching.")
        params['train_mode'] = 0
    else:
        if params['train_mode'] == 1:
            print("Retain {} epochs for SL.".format(params['model_retain_epochs_sl']))
        print("Pretrain with RL for {} epochs.".format(params['rl_pretrain_epochs']))

    if params['rl_lr_decay']:
        print("Use learning rate decay in RL.")

    # create directories
    if not (os.path.exists(params['dir'])):
        os.makedirs(params['dir'])
    os.chdir(params['dir'])

    if params['uuid']:
        if not (os.path.exists(params['uuid'])):
            os.makedirs(params['uuid'])
        os.chdir(params['uuid'])

    train_agent(params)