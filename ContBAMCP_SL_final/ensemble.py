import gym
import numpy as np
import torch
import pickle, sys, time
from copy import deepcopy
from collections import deque, namedtuple

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from utils.env_utils import reward_func, MeanStdevFilter, check_or_make_folder
from buffer import FasterReplayPool, TransitionDataset, Transition, EnsembleTransitionDataset, prepare_data
from nn_model import Model

class Ensemble(object):
    def __init__(self, params):
        self.params = params
        self._device = params['device']

        self.models = {i: Model(input_dim=params['ob_dim'] + params['ac_dim'],
                                output_dim=params['ob_dim'] + params['reward_head'],
                                h=params['hidden_dim'],
                                is_probabilistic=params['logvar_head'],
                                is_done_func=params['is_done_func'],
                                reward_head=params['reward_head'],
                                seed=params['seed'] + i,
                                l2_reg_multiplier=params['l2_reg_multiplier'],
                                num=i,
                                device=self._device)
                       for i in range(params['num_models'])}

        self.num_models = params['num_models']
        self.output_dim = params['ob_dim'] + params['reward_head']
        self.ob_dim = params['ob_dim']
        self.memory = FasterReplayPool(action_dim=params['ac_dim'], state_dim=params['ob_dim'],
                                       capacity=params['train_memory'])
        self.memory_val = FasterReplayPool(action_dim=params['ac_dim'], state_dim=params['ob_dim'],
                                           capacity=params['val_memory'])
        
        self.train_val_ratio = params['train_val_ratio']
        self.is_done_func = params['is_done_func']
        self.is_probabilistic = params['logvar_head']
        self._model_lr = params['model_lr'] if 'model_lr' in params else 0.001

        # gather weights
        weights = [weight for model in self.models.values() for weight in model.weights]
        if self.is_probabilistic:
            self.max_logvar = torch.full((self.output_dim,), 0.5, requires_grad=True, device=self._device)
            self.min_logvar = torch.full((self.output_dim,), -10.0, requires_grad=True, device=self._device)
            weights.append({'params': [self.max_logvar]})
            weights.append({'params': [self.min_logvar]})
            self.set_model_logvar_limits()

        # set up the optimizer
        self.optimizer = torch.optim.Adam(weights, lr=self._model_lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.3, verbose=False)

        self._model_id = "Model_{}_seed{}_{}".format(params['env_name'], params['seed'], params['time_stamp'])
        self._env_name = params['env_name']
        self.state_filter = MeanStdevFilter(params['ob_dim'])
        self.action_filter = MeanStdevFilter(params['ac_dim'])

        self.mopo_lam = params['mopo_lam']
    
    def set_model_logvar_limits(self):
        if isinstance(self.max_logvar, dict):
            for i, model in enumerate(self.models.values()):
                model.update_logvar_limits(self.max_logvar[self._model_groups[i]], self.min_logvar[self._model_groups[i]])
        else:
            for model in self.models.values():
                model.update_logvar_limits(self.max_logvar, self.min_logvar)
    
    def load_model(self, model_dir):
        """
        Method to load model from checkpoint folder
        """
        # Check that the environment matches the dir name
        assert self._env_name.split('-')[0].lower() in model_dir.lower(), "Model loaded was not trained on this environment"

        print("Loading model from checkpoint...")

        torch_state_dict = torch.load(model_dir + '/torch_model_weights.pt', map_location=self._device)
        for i in range(self.num_models):
            self.models[i].load_state_dict(torch_state_dict['model_{}_state_dict'.format(i)])
        self.min_logvar = torch_state_dict['logvar_min']
        self.max_logvar = torch_state_dict['logvar_max']

        data_state_dict = pickle.load(open(model_dir + '/model_data.pkl', 'rb'))
        # Backwards Compatability
        self.memory, self.memory_val = data_state_dict['train_buffer'], data_state_dict['valid_buffer']
        self.state_filter, self.action_filter = data_state_dict['state_filter'], data_state_dict['action_filter']

        # Confirm that we retrieve the checkpointed validation performance
        all_valid = self.memory_val.sample_all()
        validate_dataset = TransitionDataset(all_valid, self.state_filter, self.action_filter)
        sampler = SequentialSampler(validate_dataset)
        validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=256,
            pin_memory=True
        )

        val_losses = self._validation_model(validation_loader)
        self.set_model_logvar_limits()

        model_id = model_dir.split('/')[-1]
        self._model_id = model_id

        return val_losses
    
    def _validation_model(self, validation_loader):
        val_losses, _ = self._get_validation_losses(validation_loader, get_weights=False)
        print('Sorting Models from most to least accurate...')
        models_val_rank = val_losses.argsort()
        val_losses.sort()
        print('\nModel validation losses: {}'.format(val_losses))
        self.models = {i: self.models[idx] for i, idx in enumerate(models_val_rank)}
        return val_losses
    
    def _get_validation_losses(self, validation_loader, get_weights=True):
        best_losses = []
        best_weights = []
        for model in self.models.values():
            best_losses.append(model.get_validation_loss(validation_loader))
            if get_weights:
                best_weights.append(deepcopy(model.state_dict()))
        best_losses = np.array(best_losses)
        return best_losses, best_weights
    
    def add_data(self, step):
        # for step in rollout:
        self.memory.push(step)

    def add_data_validation(self, step):
        # for step in rollout:
        self.memory_val.push(step)
    
    def check_validation_losses(self, validation_loader):
        improved_any = False
        current_losses, current_weights = self._get_validation_losses(validation_loader, get_weights=True)
        improvements = ((self.current_best_losses - current_losses) / self.current_best_losses) > 0.01
        # print(current_losses.shape, improvements.shape) # (7,) (7,)

        for i, improved in enumerate(improvements):
            if improved:
                self.current_best_losses[i] = current_losses[i]
                self.current_best_weights[i] = current_weights[i]
                improved_any = True
        return improved_any, current_losses

    def train_model(self, max_epochs: int = 100, n_samples: int = 200000, save_model=False, min_model_epochs=None):
        self.current_best_losses = np.zeros(self.params['num_models']) + sys.maxsize  
        # print(max_epochs, n_samples, d4rl_init, save_model, min_model_epochs)
        # 2000 200000 True True None

        self.current_best_weights = [None] * self.params['num_models']
        val_improve = deque(maxlen=6)
        lr_lower = False
        min_model_epochs = 0 if not min_model_epochs else min_model_epochs

        # Train on the full buffer until convergence, should be under 5k epochs
        n_samples = len(self.memory)
        n_samples_val = len(self.memory_val)
        samples_train = self.memory.sample(n_samples) # could be sample all
        samples_validate = self.memory_val.sample_all()
        batch_size = 256

        ########## MIX VALDIATION AND TRAINING ##########
        # totally unnecessary, but this is from the original implementation
        new_samples_train_dict = dict.fromkeys(samples_train._fields)
        new_samples_validate_dict = dict.fromkeys(samples_validate._fields)
        randperm = np.random.permutation(n_samples + n_samples_val)
        # print(samples_validate._fields) # ('state', 'action', 'reward', 'nextstate', 'real_done')
        train_idx, valid_idx = randperm[:n_samples], randperm[n_samples:]
        assert len(valid_idx) == n_samples_val

        for i, key in enumerate(samples_train._fields):
            train_vals = samples_train[i]
            valid_vals = samples_validate[i]
            all_vals = np.array(list(train_vals) + list(valid_vals))
            train_vals = all_vals[train_idx]
            valid_vals = all_vals[valid_idx]
            new_samples_train_dict[key] = tuple(train_vals)
            new_samples_validate_dict[key] = tuple(valid_vals)

        samples_train = Transition(**new_samples_train_dict)
        samples_validate = Transition(**new_samples_validate_dict)

        ########## MIX VALDIATION AND TRAINING ##########
        transition_loader = DataLoader(
            EnsembleTransitionDataset(samples_train, self.state_filter, self.action_filter, n_models=self.num_models),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
        )
        validate_dataset = TransitionDataset(samples_validate, self.state_filter, self.action_filter)
        sampler = SequentialSampler(validate_dataset)
        validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        )

        ### check validation before first training epoch
        improved_any, iter_best_loss = self.check_validation_losses(validation_loader)
        val_improve.append(improved_any)
        best_epoch = 0
        print('Epoch: %s, Total Loss: N/A' % (0))
        print('Validation Losses:')
        print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))

        for i in range(max_epochs):
            t0 = time.time()
            total_loss = 0
            loss = 0
            step = 0
            # value to shuffle dataloader rows by so each epoch each model sees different data
            perm = np.random.choice(self.num_models, size=self.num_models, replace=False)
            for x_batch, diff_batch, r_batch in transition_loader:
                x_batch = x_batch[:, perm]
                diff_batch = diff_batch[:, perm]
                r_batch = r_batch[:, perm]
                step += 1
                for idx in range(self.num_models):
                    loss += self.models[idx].train_model_forward(x_batch[:, idx], diff_batch[:, idx], r_batch[:, idx])
                total_loss = loss.item()
                if self.is_probabilistic:
                    loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = 0
            t1 = time.time()
            print("Epoch training took {} seconds".format(t1 - t0))

            improved_any, iter_best_loss = self.check_validation_losses(validation_loader)
            print('Epoch: {}, Total Loss: {}'.format(int(i + 1), float(total_loss)))
            print('Validation Losses:')
            print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
            print('Best Validation Losses So Far:')
            print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(self.current_best_losses)))
            val_improve.append(improved_any)
            if improved_any:
                best_epoch = (i + 1)
                print('Improvement detected this epoch.')
            else:
                epoch_diff = i + 1 - best_epoch
                plural = 's' if epoch_diff > 1 else ''
                print('No improvement detected this epoch: {} Epoch{} since last improvement.'.format(epoch_diff, plural))
            
            if len(val_improve) > 5:
                # early stop
                if not any(np.array(val_improve)[1:]):
                    # assert val_improve[0]
                    if (i >= min_model_epochs):
                        print('Validation loss stopped improving at %s epochs' % (best_epoch))
                        for model_index in self.models:
                            self.models[model_index].load_state_dict(self.current_best_weights[model_index])
                        self._validation_model(validation_loader)
                        if save_model:
                            self._save_model()
                        return
                    elif not lr_lower:
                        self._lr_scheduler.step()
                        lr_lower = True
                        val_improve = deque(maxlen=6)
                        val_improve.append(True)
                        print("Lowering Adam Learning for fine-tuning")
            t2 = time.time()
            print("Validation took {} seconds".format(t2 - t1))

        self._validation_model(validation_loader)
        if save_model:
            self._save_model()
    
    def _save_model(self):
        """
        Method to save model after training is completed
        """
        print("Saving model checkpoint...")
        check_or_make_folder("./checkpoints")
        check_or_make_folder("./checkpoints/model_saved_weights")
        save_dir = "./checkpoints/model_saved_weights/{}".format(self._model_id)
        check_or_make_folder(save_dir)
        # Create a dictionary with pytorch objects we need to save, starting with models
        torch_state_dict = {'model_{}_state_dict'.format(i): w for i, w in enumerate(self.current_best_weights)}
        # Then add logvariance limit terms
        torch_state_dict['logvar_min'] = self.min_logvar
        torch_state_dict['logvar_max'] = self.max_logvar
        # Save Torch files
        torch.save(torch_state_dict, save_dir + "/torch_model_weights.pt")
        # Create a dict containing training and validation datasets
        data_state_dict = {'train_buffer': self.memory, 'valid_buffer': self.memory_val,
                           'state_filter': self.state_filter, 'action_filter': self.action_filter}
        # Then add validation performance for checking purposes during loading (i.e., make sure we got the same performance)
        data_state_dict['validation_performance'] = self.current_best_losses
        # Pickle the data dict
        pickle.dump(data_state_dict, open(save_dir + '/model_data.pkl', 'wb'))
        print("Saved model snapshot trained on {} datapoints".format(len(self.memory)))
    
    def get_bayes_priors(self, dataset):
        state, action, next_state, reward = dataset['observations'], dataset['actions'], \
                                            dataset['next_observations'], dataset['rewards']
        mini_batchsize = 50000
        total_len = state.shape[0]
        
        prob_dict = {}
        for k in self.models:
            prob_dict[k] = []
        i = 0
        while i * mini_batchsize < total_len:
            s_id = i * mini_batchsize
            e_id = min((i+1) * mini_batchsize, total_len)
            b_state, b_action, b_next_satte, b_reward = state[s_id:e_id], action[s_id:e_id], \
                                                        next_state[s_id:e_id], reward[s_id:e_id]
            state_action_filtered, delta_filtered = prepare_data(b_state, b_action, b_next_satte, 
                                                                 self.state_filter, self.action_filter)
            
            # 200918 (50000,) (50000, 14) (50000, 11)
            # print(total_len, b_reward.shape, state_action_filtered.shape, delta_filtered.shape)
            b_input = torch.FloatTensor(state_action_filtered).to(self._device)
            b_delta = torch.FloatTensor(delta_filtered).to(self._device)
            # assert there is reward head
            b_reward = torch.FloatTensor(b_reward).to(self._device).unsqueeze(-1)
            b_output = torch.cat([b_delta, b_reward], dim=-1)

            for k, m in self.models.items():
                prob_dict[k].append(m.get_prob(b_input, b_output).cpu().detach().clone().numpy())

            i += 1
        
        all_prob_ls = []
        for k in self.models:
            prob_dict[k] = np.concatenate(prob_dict[k], axis=0)
            # print(prob_dict[k].shape)
            all_prob_ls.append(prob_dict[k])
        
        return np.array(all_prob_ls)
    
    def get_all_predictions(self, state, action):
        key_list = list(self.models.keys())
        
        all_predictions = [self.models[i].get_predictions(state, action, self.state_filter, self.action_filter) for i in key_list]

        mus, logvars, state_action_fs = [], [], []
        for pred in all_predictions:
            mus.append(pred[0])
            logvars.append(pred[1])
            state_action_fs.append(pred[2])
        
        return torch.stack(mus, dim=0), torch.stack(logvars, dim=0), torch.stack(state_action_fs, dim=0)
    
    def bayes_env_step(self, state, action, all_prior, deterministic=False):
        # print(get_var, deterministic) # True False
        mus, logvars, state_action_fs = self.get_all_predictions(state, action)
        prior_ls = all_prior.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
        # torch.Size([20, 150, 14]) torch.Size([20, 150]) torch.Size([20, 150, 12]) torch.Size([20, 150, 12]) torch.Size([20, 150, 12])
        # print(state_action_fs.shape, all_prior.shape, mus.shape, logvars.shape, prior_ls.shape) 
        # GMM
        mean_mus = (prior_ls * mus).sum(dim=0)
        ensemble_var = (prior_ls * (logvars.exp() + mus * mus)).sum(dim=0) - mean_mus * mean_mus

        # ensemble_std = ensemble_var.sqrt() + 1e-6
        ensemble_std = ensemble_var.sqrt()
        ensemble_std[ensemble_std<=0] = 1e-6
        ensemble_std[torch.isnan(ensemble_std)] = 1e-6

        if not deterministic:
            try:
                dist = torch.distributions.Normal(mean_mus, ensemble_std)
            except:
                print(ensemble_std[ensemble_std < 0.0])
                print(ensemble_std.min())
                raise NotImplementedError
            mean_mus = dist.sample()
        
        if self.params['reward_head']:
            mu_diff_f = mean_mus[:, :-1]
            mu_reward = mean_mus[:, -1]
        else:
            mu_diff_f = mean_mus
            mu_reward = torch.zeros_like(mean_mus[:, -1])

        nextstates = state + mu_diff_f # torch.Size([50000, 11]) torch.Size([50000])
        
        mopo_lam = self.mopo_lam
        # TODO: use other penalties
        mopo_pen = ensemble_std.mean(dim=1)

        rewards = mu_reward - mopo_lam * mopo_pen
        # if self.params['env_name'] == 'AntMOPOEnv':
        #     rewards += 1.0

        # calculate the transition prob
        trans_probs = []
        for k, m in self.models.items():
            # it's not necessary to collect and input state_action_fs[k]
            trans_probs.append(m.get_prob(state_action_fs[k], mean_mus.clone(), mus[k], logvars[k]).cpu().detach().clone().numpy())

        return nextstates, rewards, mopo_pen, mu_reward, np.array(trans_probs)
    
    def predict_state(self, state: np.array, action: np.array):
        model_index = int(np.random.uniform() * len(self.models.keys()))
        return self.models[model_index].predict_state(state, action, self.state_filter, self.action_filter)

class EnsembleGymEnv(gym.Env):
    """Wraps the Ensemble with a gym API, Outputs Normal states, and contains a copy of the true environment"""

    def __init__(self, params, env, eval_env):
        super(EnsembleGymEnv, self).__init__()
        self.name = params['env_name']
        self.real_env = env
        self.eval_env = eval_env
        self._device = params['device']

        self.observation_space = self.real_env.observation_space
        self.action_space = self.real_env.action_space

        self.model = Ensemble(params)
        
        self.current_state = self.reset()
        self.reward_head = params['reward_head']
        self.reward_func = reward_func
        self.action_bounds = self.get_action_bounds()
        self.spec = self.real_env.spec
        self._elapsed_steps = 0
        self._max_timesteps = self.spec.max_episode_steps
        torch.manual_seed(params['seed'])
    
    def reset(self):
        self.current_state = self.eval_env.reset()
        self._elapsed_steps = 0
        return self.current_state
    
    def get_action_bounds(self):
        Bounds = namedtuple('Bounds', ('lowerbound', 'upperbound'))
        lb = self.real_env.action_space.low
        ub = self.real_env.action_space.high
        return Bounds(lowerbound=lb, upperbound=ub)
    
    def update_state_filter(self, new_state):
        self.model.state_filter.update(new_state)
    
    def update_action_filter(self, new_action):
        self.model.action_filter.update(new_action)
    
    def train_model(self, max_epochs, n_samples: int = 200000, save_model=False, min_model_epochs=None):
        self.model.train_model(
            max_epochs=max_epochs,
            n_samples=n_samples,
            save_model=save_model,
            min_model_epochs=min_model_epochs)
    
    def get_bayes_priors(self, dataset):
        return self.model.get_bayes_priors(dataset)
    
    def convert_filter_to_torch(self):
        self.model.state_filter.update_torch(self._device)
        self.model.action_filter.update_torch(self._device)

    def reset(self):
        self.current_state = self.eval_env.reset()
        self._elapsed_steps = 0
        return self.current_state
    
    def step(self, action):
        action = np.clip(action, self.action_bounds.lowerbound, self.action_bounds.upperbound)
        next_state, reward = self.model.predict_state(
            self.current_state.reshape(1, -1),
            action.reshape(1, -1))
        # print(next_state.shape, reward) # (1, 11) 1.0149509906768799
        if not reward:
            reward = self.reward_func(
                self.current_state,
                next_state,
                action,
                self.name,
                is_done_func=self.model.is_done_func)
        if self._elapsed_steps > self._max_timesteps:
            done = True
        elif self.model.is_done_func: # True
            done = self.model.is_done_func(torch.Tensor(next_state).reshape(1, -1)).item()
        else:
            done = False
        self.current_state = next_state
        self._elapsed_steps += 1
        return next_state, reward, done, {}