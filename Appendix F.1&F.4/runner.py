import os, csv
import d4rl, gym, pickle
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from ensemble import EnsembleGymEnv
from sac import SAC_Agent
from buffer import Transition, SL_Transition
from utils.env_utils import check_or_make_folder

class Runner(object):

    def __init__(self, params, model: EnsembleGymEnv, agent: SAC_Agent, searcher, sl_buffer, eval_env_vectors):
        self.agent = agent
        self.model = model
        self.searcher = searcher # None for non-search case
        self.sl_buffer = sl_buffer # None for non-search case
        self.eval_env_vectors = eval_env_vectors

        self._max_model_epochs = params['model_epochs']
        self._num_rollouts = params['num_rollouts_per_step'] * params['model_train_freq'] # weird design by the initial implementation
        # self._num_rollouts = int(params['num_rollouts_per_step'] * params['model_train_freq'] * params['search_ratio'])
        self._model_retain_epochs = params['model_retain_epochs']

        self._device = params['device']
        self._reward_head = params['reward_head']
        self._policy_update_steps = params['policy_update_steps']
        self._steps_k = params['steps_k']
        if isinstance(self._steps_k, list):
            self._cur_steps_k = self._steps_k[0]
        else:
            self._cur_steps_k = self._steps_k

        self._n_eval_rollouts = params['n_eval_rollouts']
        self._real_sample_ratio = params['real_sample_ratio']
        self._model_train_freq = params['model_train_freq']
        self._oac = params['oac']

        self._n_epochs = 0
        self._is_done_func = params['is_done_func']
        self._var_thresh = params['var_thresh']
        self._keep_logvar = True if self._var_thresh is not None else False

        self._max_steps = params['epoch_steps'] if params['epoch_steps'] else self.model._max_timesteps
        self._deterministic = params['deterministic_rollouts']
        self._seed = params['seed']
        self._min_model_epochs = params['min_model_epochs']
        if self._min_model_epochs:
            assert self._min_model_epochs < self._max_model_epochs, "Can't have a min epochs that is less than the max"

        self._params = params
    
    def _train_model(self, save_model=False):
        print("\nTraining Model...")
        self.model.train_model(self._max_model_epochs, save_model=save_model,
                               min_model_epochs=self._min_model_epochs)

    def _get_action(self, state, all_prior, use_all=False, uniform_ensemble=False):
        actions, _, _, logits = self.agent.policy(state, get_logits=True)
        # using MCTS
        if not use_all:
            search_size = int(state.shape[0] * self._params['search_ratio'])
            # search_size = state.shape[0]
            # TODO: try different forms of randomness
            search_idx = np.random.choice(state.shape[0], size=search_size, replace=False)
        else:
            search_size = int(state.shape[0])
            search_idx = np.arange(search_size)
        state_input = state[search_idx].cpu().numpy()
        prior_input = all_prior[:, search_idx].cpu().numpy()
        logits[0] = logits[0][search_idx].cpu().numpy()
        logits[1] = logits[1][search_idx].cpu().numpy()

        tree_roots = self.searcher.set_roots(search_size)
        self.searcher.prepare(tree_roots, prior_input, state_input, logits)
        print("Start searching ...")
        self.searcher.search(tree_roots, self._is_done_func, uniform_ensemble=uniform_ensemble)
        print("Start sampling ...")
        searched_actions, action_dists, action_lists, q_list = self.searcher.sample(tree_roots)

        if self._params['train_mode'] != 0:
            self.sl_buffer.push(SL_Transition(state_input, action_lists, [], action_dists, q_list))

        actions[search_idx] = torch.FloatTensor(searched_actions).to(actions.device)
        return actions

    def _rollout_model(self, dataset, epoch_num):
        print("\nRolling out Policy in Model...")

        for model in self.model.model.models.values():
            model.to(self._device)

        self.model.convert_filter_to_torch()

        done_false = [False for _ in range(self._num_rollouts)]

        dst_size = dataset['rewards'].shape[0]
        rand_idx = np.random.randint(0, dst_size, self._num_rollouts)
        start_states = torch.FloatTensor(dataset['observations'][rand_idx]).to(self._device)
        state = start_states.clone() # torch.Size([50000, 11])

        all_prior = torch.FloatTensor(dataset['all_priors'][rand_idx]).to(self._device).T
        # print(start_states.shape, all_prior.shape)
        # torch.Size([50000, 11]) torch.Size([20, 50000])

        t = 0
        transition_count = 0
        # start the rollout
        while t < self._cur_steps_k: # 5
            t += 1
            with torch.no_grad():
                if self._params['use_search'] and epoch_num >= self._params['rl_pretrain_epochs']:
                # if self._params['use_search']:
                    action = self._get_action(state, all_prior)
                else:
                    action, _, _ = self.agent.policy(state) # torch.Size([50000, 11]) torch.Size([50000, 3]) #!!!
                nextstate, reward, penalties, ori_reward, all_probs = self.model.model.bayes_env_step(state,
                                                                                           action,
                                                                                           all_prior,
                                                                                           deterministic=self._deterministic)
                # TODO: using reward 
                # all_probs = self.model.get_bayes_priors({'observations': state.cpu().numpy(), 'actions': action.cpu().numpy(), \
                #                                          'next_observations': nextstate.cpu().numpy(), 'rewards': ori_reward.cpu().numpy()})
                all_probs = torch.FloatTensor(all_probs).to(all_prior.device)
                all_prod = all_probs * all_prior
                all_prior = all_prod / (all_prod.sum(dim=0, keepdim=True).repeat(all_probs.shape[0], 1) + 1e-6)
                # print(all_probs.shape, all_prod.shape, all_prior.shape, 
                #       all_prod.sum(dim=0, keepdim=True).repeat(all_prod.shape[0], 1).shape) 
                # torch.Size([20, 50000]) torch.Size([20, 50000]) torch.Size([20, 50000]) torch.Size([20, 50000])
            
                nextstate_copy = nextstate.clone()

            if self._is_done_func: # True
                done = self._is_done_func(nextstate).cpu().numpy().tolist()
            else:
                done = done_false[:nextstate.shape[0]]
            not_done = ~np.array(done)
            
            if self._reward_head:
                reward = reward.cpu().detach().numpy()
            else:
                # reward = torch_reward(self.model.name, nextstate, action, done)
                raise NotImplementedError

            state_np, action_np, nextstate_np = state.detach().cpu().numpy(), action.detach().cpu().numpy(), nextstate.detach().cpu().numpy()
            # actually, train_mode 3 does not require this 
            self.agent.replay_pool.push(Transition(state_np, action_np, reward, nextstate_np, np.array(done)))
            if not_done.sum() == 0:
                print("Finished rollouts early: all terminated after %s timesteps" % (t))
                break
            transition_count += len(nextstate)
            # Initialize state clean to be augmented next step.
            if len(nextstate_copy.shape) == 1:
                nextstate_copy.unsqueeze_(0)
            state = nextstate_copy[not_done]
            all_prior = all_prior[:, not_done]

            if len(state.shape) == 1:
                state.unsqueeze_(0)
            print("Remaining = {}".format(np.round(state.shape[0]) / start_states.shape[0], 2))

        print("Finished rollouts: all terminated after %s timesteps" % (t))
        print("Added {} transitions to agent replay pool".format(transition_count))
        print("Agent replay pool: {}/{}".format(len(self.agent.replay_pool), self.agent.replay_pool.capacity))

    def _train_agent(self, epoch_num):
        real_epoch_num = epoch_num 
        if not self._params['rl_lr_decay']:
            epoch_num = None

        if self._params['train_mode'] == 0:
            self.agent.optimize(n_updates=self._policy_update_steps, env_pool=self.model.model.memory,
                                env_ratio=self._real_sample_ratio, epoch_num=epoch_num) # both policy and critic
            
        elif self._params['train_mode'] == 2:
            self.agent.sl_optimize(n_updates=self._policy_update_steps, sl_buffer=self.sl_buffer) # policy only
            self.agent.optimize(n_updates=self._policy_update_steps, env_pool=self.model.model.memory,
                                env_ratio=self._real_sample_ratio, epoch_num=epoch_num) # both policy and critic
        # supervised learning only
        else: 
            if self._params['train_mode'] == 1:
                if real_epoch_num < self._params['rl_pretrain_epochs']:
                    self.agent.optimize(n_updates=self._policy_update_steps, env_pool=self.model.model.memory,
                                        env_ratio=self._real_sample_ratio, epoch_num=epoch_num) # both policy and critic
                else:
                    self.agent.sl_optimize(n_updates=self._policy_update_steps, sl_buffer=self.sl_buffer) # policy only
                    self.agent.optimize(n_updates=self._policy_update_steps, env_pool=self.model.model.memory,
                                        env_ratio=self._real_sample_ratio, value_only=True, epoch_num=epoch_num) # critic only
            else:
                # both policy and critic
                self.agent.sl_optimize(n_updates=self._policy_update_steps, sl_buffer=self.sl_buffer, policy_only=False)
    
    def test_agent(self, n_evals=None, use_search=True):
        n_evals = min(n_evals, self._n_eval_rollouts) if n_evals else self._n_eval_rollouts

        total_reward = [0.0 for _ in range(n_evals)]
        dones = [False for _ in range(n_evals)]
        
        states = []
        for i in range(n_evals):
            states.append(self.eval_env_vectors[i].reset())
        states = np.array(states)

        if use_search:
            all_priors = [np.array([1.0 for i in range(self._params['num_models'])])/float(self._params['num_models']) for j in range(n_evals)]
            all_priors = np.array(all_priors).T

        while True:
            # get the action
            with torch.no_grad():
                _, _, actions, logits = self.agent.policy(torch.FloatTensor(states).to(self._device), get_logits=True)
            actions = actions.cpu().numpy()
            if use_search:
                logits[0] = logits[0].cpu().numpy()
                logits[1] = logits[1].cpu().numpy()
                tree_roots = self.searcher.set_roots(states.shape[0])
                self.searcher.prepare(tree_roots, all_priors, states, logits)
                # print("Start searching ...")
                self.searcher.search(tree_roots, self._is_done_func, hide_tdqm=True)
                # print("Start sampling ...")
                actions, _, _, _ = self.searcher.sample(tree_roots, deterministic=True)
            
            # env step
            j = 0
            next_states = []
            idx_list = []
            real_next_states = []
            reward_list = []
            for i in range(n_evals):
                if not dones[i]:
                    state, reward, done, _ = self.eval_env_vectors[i].step(actions[j])
                    reward = 0.0 if reward is None else reward
                    reward_list.append(reward)
                    next_states.append(state)
                    total_reward[i] += reward
                    dones[i] = done
                    if not done:
                        real_next_states.append(state)
                        idx_list.append(j)
                    j += 1

            if use_search:
                if len(states) > len(real_next_states):
                    print("Number of remaining envs: {}".format(len(real_next_states)))
                # update the priors
                all_probs = self.model.get_bayes_priors({'observations': states, 'actions': actions, \
                                                        'next_observations': np.array(next_states), 'rewards': np.array(reward_list)})
                all_prod = all_probs * all_priors
                all_priors = all_prod / (all_prod.sum(axis=0, keepdims=True).repeat(all_probs.shape[0], axis=0) + 1e-6)

            # prepare for the next step
            if len(real_next_states) == 0:
                break
            states = np.array(real_next_states)
            if use_search:
                all_priors= all_priors[:, idx_list]

        return np.array(total_reward)
    
    def _test_rollout_model(self, dataset, uniform_prior, uniform_ensemble=False):
        print("\nRolling out Policy in Model...")

        for model in self.model.model.models.values():
            model.to(self._device)

        self.model.convert_filter_to_torch()

        done_false = [False for _ in range(self._num_rollouts)]

        rand_idx = np.arange(10000)
        start_states = torch.FloatTensor(dataset['observations'][rand_idx]).to(self._device)
        state = start_states.clone() # torch.Size([50000, 11])

        if uniform_prior is None:
            all_prior = torch.FloatTensor(dataset['all_priors'][rand_idx]).to(self._device).T
        else:
            all_prior = torch.FloatTensor(uniform_prior[rand_idx]).to(self._device).T
        print(start_states.shape, all_prior.shape)
        # torch.Size([50000, 11]) torch.Size([20, 50000])

        trajs = [{"state": [state[t_id].cpu().numpy()], "action": [], "reward": [], "ori_reward": [], "prior": [all_prior[:, t_id].cpu().numpy()]} for t_id in range(10000)]
        is_available = np.ones((10000, ), dtype=bool)

        t = 0
        transition_count = 0
        # start the rollout
        while t < self._cur_steps_k: # 5
            t += 1
            with torch.no_grad():
                if self._params['use_search']:
                    action = self._get_action(state, all_prior, use_all=True, uniform_ensemble=uniform_ensemble)
                else:
                    action, _, _ = self.agent.policy(state) # torch.Size([50000, 11]) torch.Size([50000, 3]) #!!!

                nextstate, reward, penalties, ori_reward, all_probs = self.model.model.bayes_env_step(state,
                                                                                           action,
                                                                                           all_prior,
                                                                                           deterministic=self._deterministic)
                # TODO: using reward 
                # all_probs = self.model.get_bayes_priors({'observations': state.cpu().numpy(), 'actions': action.cpu().numpy(), \
                #                                          'next_observations': nextstate.cpu().numpy(), 'rewards': ori_reward.cpu().numpy()})
                if not uniform_ensemble:
                    all_probs = torch.FloatTensor(all_probs).to(all_prior.device)
                    all_prod = all_probs * all_prior
                    all_prior = all_prod / (all_prod.sum(dim=0, keepdim=True).repeat(all_probs.shape[0], 1) + 1e-6)
                # print(all_probs.shape, all_prod.shape, all_prior.shape, 
                #       all_prod.sum(dim=0, keepdim=True).repeat(all_prod.shape[0], 1).shape) 
                # torch.Size([20, 50000]) torch.Size([20, 50000]) torch.Size([20, 50000]) torch.Size([20, 50000])
            
                nextstate_copy = nextstate.clone()

            if self._is_done_func: # True
                done = self._is_done_func(nextstate).cpu().numpy().tolist()
            else:
                done = done_false[:nextstate.shape[0]]
            not_done = ~np.array(done)
            
            if self._reward_head:
                reward = reward.cpu().detach().numpy()
                ori_reward = ori_reward.cpu().detach().numpy()
            else:
                # reward = torch_reward(self.model.name, nextstate, action, done)
                raise NotImplementedError

            state_np, action_np, nextstate_np = state.detach().cpu().numpy(), action.detach().cpu().numpy(), nextstate.detach().cpu().numpy()

            m_id = 0
            for t_id in range(10000):
                if not is_available[t_id]:
                    continue
                trajs[t_id]['state'].append(nextstate_np[m_id])
                trajs[t_id]['action'].append(action_np[m_id])
                trajs[t_id]['reward'].append(reward[m_id])
                trajs[t_id]['ori_reward'].append(ori_reward[m_id])
                trajs[t_id]['prior'].append(all_prior[:, m_id].cpu().numpy())

                if done[m_id]:
                    is_available[t_id] = False

                m_id += 1

            # actually, train_mode 3 does not require this 
            if not_done.sum() == 0:
                print("Finished rollouts early: all terminated after %s timesteps" % (t))
                break
            transition_count += len(nextstate)
            # Initialize state clean to be augmented next step.
            if len(nextstate_copy.shape) == 1:
                nextstate_copy.unsqueeze_(0)
            state = nextstate_copy[not_done]
            all_prior = all_prior[:, not_done]

            if len(state.shape) == 1:
                state.unsqueeze_(0)
            print("Remaining = {}".format(np.round(state.shape[0]) / start_states.shape[0], 2))
        
        # Save the traj list to a file using pickle
        current_directory = os.getcwd()
        file_name = current_directory + '/traj_data/' 
        os.makedirs(file_name, exist_ok=True)
        file_name += self._params['env_name'] 
        if uniform_prior is not None:
            file_name += '_uniform_prior'
            if uniform_ensemble:
                file_name += '_ensemble'
        file_name += '_seed_{}'.format(self._params['seed'])
        # file_name += '_prior'
        file_name += '.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(trajs, file)


    def test_belief_offline(self, load_model_dir=None):
        # d4rl stuff - load all the offline data and train
        env = self.model.real_env
        # dataset = d4rl.qlearning_dataset(env)
        dataset = env.get_dataset()
        
        # load the dyn model or train it based on the dataset
        self.model.model.load_model(load_model_dir)
        
        # get Bayes priors associated with the offline dataset
        all_probs = self.model.get_bayes_priors(dataset)
        temp_prior = 1.0 / self._params['num_models']
        uniform_all_prior = np.array([temp_prior for _ in range(self._params['num_models'])])
        all_priors = [uniform_all_prior]
        trans_num = all_probs.shape[1]
        for i in tqdm(range(trans_num - 1)):
            done = dataset['terminals'][i]
            if 'timeouts' in dataset:
                final_timestep = dataset['timeouts'][i]
                done = done or final_timestep
            if done:
                all_priors.append(uniform_all_prior)
            else:
                a_prior = all_priors[i]
                a_prob = all_probs[:, i]
                a_prod = a_prior * a_prob
                all_priors.append(a_prod / (a_prod.sum() + 1e-6))

        all_priors = np.array(all_priors)
        # print(all_priors.shape, all_priors[:100])
        dataset["all_priors"] = all_priors

        mean_pdf, mean_mean_pdf = self.model.model.predict_on_offline_dst(dataset, all_priors)

        all_uniform_priors = np.array([uniform_all_prior for _ in range(len(all_priors))])
        uniform_mean_pdf, uniform_mean_mean_pdf = self.model.model.predict_on_offline_dst(dataset, all_uniform_priors)
        # print(mean_pdf, uniform_mean_pdf)
        print("The likelihood ratio is: ", mean_pdf / uniform_mean_pdf)

        self._test_rollout_model(dataset, uniform_prior=None)
        self._test_rollout_model(dataset, uniform_prior=all_uniform_priors)
        self._test_rollout_model(dataset, uniform_prior=all_uniform_priors, uniform_ensemble=True)

        
    def train_offline(self, num_epochs, save_model=False, save_policy=False, load_model_dir=None):
        timesteps = 0
        val_size = 0
        train_size = 0

        # d4rl stuff - load all the offline data and train
        env = self.model.real_env
        # dataset = d4rl.qlearning_dataset(env)
        dataset = env.get_dataset()
        N = dataset['rewards'].shape[0] # number of transitions
        
        # load the dyn model or train it based on the dataset
        if load_model_dir:
            self.model.model.load_model(load_model_dir)
        else:
            self.model.update_state_filter(dataset['observations'][0]) # (11,)
            for i in range(N): # 200918
                state = dataset['observations'][i]
                action = dataset['actions'][i]
                nextstate = dataset['next_observations'][i]
                reward = dataset['rewards'][i]
                done = bool(dataset['terminals'][i])

                if 'timeouts' in dataset:
                    final_timestep = dataset['timeouts'][i]
                    done = done or final_timestep

                t = Transition(state, action, reward, nextstate, done)

                self.model.update_state_filter(nextstate)
                self.model.update_action_filter(action)

                # Do this probabilistically to avoid maintaining a huge array of indices
                if random.uniform(0, 1) < self.model.model.train_val_ratio:
                    self.model.model.add_data_validation(t)
                    val_size += 1
                else:
                    self.model.model.add_data(t)
                    train_size += 1
                timesteps += 1

            print("\nAdded {} samples for train, {} for valid".format(str(train_size), str(val_size)))
            # train the dyn models
            self._train_model(save_model=save_model) 
        
        # get Bayes priors associated with the offline dataset
        all_probs = self.model.get_bayes_priors(dataset)
        temp_prior = 1.0 / self._params['num_models']
        uniform_all_prior = np.array([temp_prior for _ in range(self._params['num_models'])])
        all_priors = [uniform_all_prior]
        trans_num = all_probs.shape[1]
        for i in tqdm(range(trans_num - 1)):
            done = dataset['terminals'][i]
            if 'timeouts' in dataset:
                final_timestep = dataset['timeouts'][i]
                done = done or final_timestep
            if done:
                all_priors.append(uniform_all_prior)
            else:
                a_prior = all_priors[i]
                a_prob = all_probs[:, i]
                a_prod = a_prior * a_prob
                all_priors.append(a_prod / (a_prod.sum() + 1e-6))

        all_priors = np.array(all_priors)
        # print(all_priors.shape, all_priors[:100])
        dataset["all_priors"] = all_priors

        # prepare the logger, including tensorboard and csv writer
        cwd = os.getcwd()
        exp_id = '/output/{}_lam{}_seed{}_search{}_'.format(self._params['env_name'], self._params['mopo_lam'], self._params['seed'], self._params['use_search'])
        if self._params['use_search']:
            exp_id += 'ratio{}_ns{}_nstate_{}_'.format(self._params['search_ratio'], self.searcher._cfg.num_search, self.searcher._cfg.num_states)
        log_dir = cwd + exp_id + self._params['time_stamp']
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)

        csv_path = os.path.join(log_dir, "output.csv")
        csv_file = open(csv_path, mode="a", newline="")
        csv_writer = csv.writer(csv_file)
        if os.stat(csv_path).st_size == 0:  # Check if the file is empty
            header = ["Iteration", "Return", "Return_Max", "Score", "Score_Max", "Score_10", "Score_10_Max",  "Score_50", "Score_50_Max",  "Score_100", "Score_100_Max"]
            csv_writer.writerow(header)
        # for calculating the sliding average
        scores, scores_max = [], []

        # rl training
        for i in range(num_epochs):
            self._rollout_model(dataset, epoch_num=i) # rollout with the dyn model
            self._train_agent(epoch_num=i) # train the agent with sac
            # reward_model = self.test_agent(use_model=True, n_evals=10) # test with the learned dynamic models
            reward_actual_stats = self.test_agent(use_search=False) # real metric

            cur_reward = reward_actual_stats.mean()
            cur_reward_max = reward_actual_stats.max()
            # log the evaluation results
            print("------------------------")
            stats_fmt = "{:<20}{:>30}"
            stats_str = ["Epoch",
                         "True Reward Mean",
                         "True Reward Max",
                         "True Reward Min",
                         "True Reward StdDev"]
            stats_num = [i,
                         cur_reward.round(2),
                         cur_reward_max.round(2),
                         reward_actual_stats.min().round(2),
                         reward_actual_stats.std().round(2)]
            
            if self._params['use_search_eval'] and (i % 10 == 0): # very time consuming
                reward_stats_with_search = self.test_agent()
                stats_str.extend(['True Reward Mean (Search)', 'True Reward Max (Search)',
                                  'True Reward Min (Search)', 'True Reward StdDev (Search)'])
                stats_num.extend([reward_stats_with_search.mean().round(2), reward_stats_with_search.max().round(2), 
                                  reward_stats_with_search.min().round(2), reward_stats_with_search.std().round(2)])
            
            for s, n in zip(stats_str, stats_num):
                print(stats_fmt.format(s, n))
                if s != "Epoch":
                    writer.add_scalar(s, n, i)
            print("------------------------")
            
            if self._params['d4rl']:
                cur_socre = self.model.eval_env.get_normalized_score(cur_reward)
                cur_score_max = self.model.eval_env.get_normalized_score(cur_reward_max)
            else:
                raise NotImplementedError
            
            scores.append(cur_socre)
            scores_max.append(cur_score_max)
            writer.add_scalar('score', cur_socre, i)
            writer.add_scalar('score_max', cur_score_max, i)

            csv_row = [i, cur_reward, cur_reward_max, cur_socre, cur_score_max, np.mean(scores[-10:]), np.mean(scores_max[-10:]),
                       np.mean(scores[-50:]), np.mean(scores_max[-50:]), np.mean(scores[-100:]), np.mean(scores_max[-100:])]
            csv_writer.writerow(csv_row)

            if save_policy and i % 50 == 0:
                save_path = './model_saved_weights_seed{}'.format(self._params['seed'])
                check_or_make_folder(save_path)
                print("Saving policy trained offline")
                self.agent.save_policy(
                    # "{}".format(self.model.model._model_id),
                    save_path,
                    num_epochs=i,
                    rew=int(reward_actual_stats.mean()))
        
        steps_k_used = self._cur_steps_k
        self._steps_k_update()
        csv_file.close()
    
    def _steps_k_update(self):
        if isinstance(self._steps_k, int):
            return
        else:
            steps_min, steps_max, start_epoch, end_epoch = self._steps_k
            m = (steps_max - steps_min) / (end_epoch - start_epoch)
            c = steps_min - m * start_epoch
        new_steps_k = m * self._n_epochs + c
        new_steps_k = int(min(steps_max, max(new_steps_k, steps_min)))
        if new_steps_k == self._cur_steps_k:
            return
        else:
            print("\nChanging model step size, going from %s to %s" % (self._cur_steps_k, new_steps_k))
            self._cur_steps_k = new_steps_k
            new_pool_size = int(
                self._cur_steps_k * self._num_rollouts * (
                        self._max_steps / self._model_train_freq) * self._model_retain_epochs)
            print("\nReallocating agent pool, going from %s to %s" % (self.agent.replay_pool.capacity, new_pool_size))
            self.agent.reallocate_replay_pool(new_pool_size)