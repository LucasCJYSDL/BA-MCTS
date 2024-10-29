import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, TransformedDistribution
import torch.nn.functional as F

from utils.nn_utils import reinitialize_fc_layer_, get_weight_bias_parameters_with_decays, GaussianMSELoss, TanhTransform

# policy and critic networks

class MLPNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)

class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def stable_network_forward(self, x):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        return mu, logstd

    def compute_action(self, mu, std, get_logprob=False):
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action, logprob, mean

    def forward(self, x, get_logprob=False, get_logits=False):
        mu, logstd = self.stable_network_forward(x)
        std = logstd.exp()
        # TODO
        std = torch.nan_to_num(std, nan=1e-6)

        if get_logits:
            action, logprob, mean = self.compute_action(mu, std, get_logprob)
            return action, logprob, mean, [mu, std]
        return self.compute_action(mu, std, get_logprob)
    
    # def forward_entropy(self, x):
    #     mu, logstd = self.stable_network_forward(x)
    #     std = logstd.exp()
    #     dist = Normal(mu, std)
    #     transforms = [TanhTransform(cache_size=1)]
    #     dist = TransformedDistribution(dist, transforms)
    #     action = dist.rsample()
    #     entropy = dist.entropy() # NotImplemented

    #     return action, entropy

class DoubleQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

# dynamic and reward models

class VanillaNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 1024,
                 is_done_func=None,
                 reward_head=True,
                 seed=0, device='cpu'):

        super().__init__()
        torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU()
        )
        self.delta = nn.Linear(h, output_dim)
        params = list(self.network.parameters()) + list(self.delta.parameters())
        self.weights = params
        self.to(device)
        self.loss = nn.MSELoss()
        self.is_done_func = is_done_func
        self.reward_head = reward_head
        self._device = device
    
    @property
    def is_probabilistic(self):
        return False

    def forward(self, x: torch.Tensor):
        hidden = self.network(x)
        delta = self.delta(hidden)
        return delta

    @staticmethod
    def filter_inputs(state, action, state_filter, action_filter, device):
        state_f = state_filter.filter_torch(state, device)
        action_f = action_filter.filter_torch(action, device)
        state_action_f = torch.cat((state_f, action_f), dim=1)
        return state_action_f

    def get_next_state_reward(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter,
                              keep_logvar=False):
        if keep_logvar:
            raise Exception("This is a deterministic network, there is no logvariance prediction")
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter, self._device)
        y = self.forward(state_action_f)
        if self.reward_head:
            diff_f = y[:, :-1]
            reward = y[:, -1].unsqueeze(1)
        else:
            diff_f = y
            reward = 0
        diff = diff_f
        nextstate = state + diff
        return nextstate, reward

    def get_state_action_uncertainty(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        raise Exception("This is a deterministic network, there is no logvariance prediction")
    
class ProbabilisticNeuralNetwork(VanillaNeuralNetwork): # IDK why succeeding from VanillaNeuralNetwork
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 200,
                 is_done_func=None,
                 reward_head=True,
                 l2_reg_multiplier=1.,
                 seed=0, device='cpu'):
        super().__init__(input_dim, output_dim, h, is_done_func, reward_head, seed)
        torch.manual_seed(seed)
        del self.network
        self._device = device

        self.fc1 = nn.Linear(input_dim, h)
        reinitialize_fc_layer_(self.fc1)
        self.fc2 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc2)
        self.fc3 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc3)
        self.fc4 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc4)
        self.use_blr = False
        self.delta = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.delta)
        self.logvar = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.logvar)
        self.loss = GaussianMSELoss()
        self.activation = nn.SiLU()
        self.lambda_prec = 1.0
        self.max_logvar = None
        self.min_logvar = None
        params = []
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.delta, self.logvar]
        self.decays = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]) * l2_reg_multiplier
        for layer, decay in zip(self.layers, self.decays):
            params.extend(get_weight_bias_parameters_with_decays(layer, decay))
        self.weights = params
        self.to(device)
    
    def update_logvar_limits(self, max_logvar, min_logvar):
        self.max_logvar, self.min_logvar = max_logvar, min_logvar
    
    @property
    def is_probabilistic(self):
        return True
    
    def get_l2_reg_loss(self):
        l2_loss = 0
        for layer, decay in zip(self.layers, self.decays):
            for name, parameter in layer.named_parameters():
                if 'weight' in name:
                    l2_loss += parameter.pow(2).sum() / 2 * decay
        return l2_loss

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        delta = self.delta(x)
        logvar = self.logvar(x)
        # Taken from the PETS code to stabilise training
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return torch.cat((delta, logvar), dim=1)
    
    def get_prob(self, input_batch, output_batch, mu=None, logvar=None):
        if mu is None:
            mu, logvar = self.forward(input_batch).chunk(2, dim=1)
        dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
        if not self.reward_head:
            output_batch = output_batch[:, :-1]
        l_prob = dist.log_prob(output_batch) # log of pdf
        # TODO: use sum
        prob = l_prob.mean(-1).exp() # not sum, to normalize with the action dim
        # print(l_prob[:10], mu[:10], logvar[:10], output_batch[:10], l_prob.mean(-1)[:10], prob[:10])
        return prob
    
    def get_predictions(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter, self._device)
        mu, logvar = self.forward(state_action_f).chunk(2, dim=1)

        return mu, logvar, state_action_f
    
    def get_next_state_reward(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter,
                              keep_logvar=False, deterministic=False, return_mean=False):
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter, self._device)
        mu, logvar = self.forward(state_action_f).chunk(2, dim=1)
        mu_orig = mu
        if not deterministic:
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            mu = dist.sample()

        if self.reward_head:
            mu_diff_f = mu[:, :-1]
            logvar_diff_f = logvar[:, :-1]
            mu_reward = mu[:, -1].unsqueeze(1)
            logvar_reward = logvar[:, -1].unsqueeze(1)
            mu_diff_f_orig = mu_orig[:, :-1]
            mu_reward_orig = mu_orig[:, -1].unsqueeze(1)
        else:
            mu_diff_f = mu
            logvar_diff_f = logvar
            mu_reward = torch.zeros_like(mu[:, -1].unsqueeze(1))
            logvar_reward = torch.zeros_like(logvar[:, -1].unsqueeze(1))
            mu_diff_f_orig = mu_orig
            mu_reward_orig = mu_reward

        mu_diff = mu_diff_f
        mu_nextstate = state + mu_diff
        logvar_nextstate = logvar_diff_f
        if return_mean:
            mu_nextstate = torch.cat((mu_nextstate, mu_diff_f_orig + state), dim=1)
            mu_reward = torch.cat((mu_reward, mu_reward_orig), dim=1)
        if keep_logvar:
            return (mu_nextstate, logvar_nextstate), (mu_reward, logvar_reward)
        else:
            return mu_nextstate, mu_reward

class Model(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 1024,
                 is_probabilistic=True,
                 is_done_func=None,
                 reward_head=1,
                 seed=0,
                 l2_reg_multiplier=1.,
                 num=0, device='cpu'):

        super(Model, self).__init__()
        torch.manual_seed(seed)

        if is_probabilistic:
            self.model = ProbabilisticNeuralNetwork(input_dim, output_dim, h, is_done_func, reward_head, l2_reg_multiplier,
                                                    seed, device=device)
        else:
            self.model = VanillaNeuralNetwork(input_dim, output_dim, h, is_done_func, reward_head, seed, device=device)
        self._device = device
        self.is_probabilistic = self.model.is_probabilistic
        self.weights = self.model.weights
        self.reward_head = reward_head

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def update_logvar_limits(self, max_logvar, min_logvar):
        assert self.is_probabilistic, self.is_probabilistic
        self.model.update_logvar_limits(max_logvar, min_logvar)
    
    def get_validation_loss(self, validation_loader):
        self.model.eval()
        preds, targets = self.get_predictions_from_loader(validation_loader, return_targets=True)
        if self.is_probabilistic:
            return self.model.loss(preds, targets, logvar_loss=False).item()
        else:
            return self.model.loss(preds, targets).item()
    
    def get_predictions_from_loader(self, data_loader, return_targets = False, return_sample=False):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_batch_val, delta_batch_val, r_batch_val in data_loader:
                x_batch_val, delta_batch_val, r_batch_val = x_batch_val.to(self._device, non_blocking=True), \
                                                            delta_batch_val.to(self._device, non_blocking=True), \
                                                            r_batch_val.to(self._device, non_blocking=True)
                y_pred_val = self.forward(x_batch_val)
                preds.append(y_pred_val)
                if return_targets:
                    y_batch_val = torch.cat([delta_batch_val, r_batch_val.unsqueeze(dim=1)], dim=1) if self.reward_head else delta_batch_val
                    targets.append(y_batch_val)
        
        preds = torch.vstack(preds)

        if return_sample:
            mu, logvar = preds.chunk(2, dim=1)
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            sample = dist.sample()
            # torch.Size([40462, 24]) torch.Size([40462, 12]) torch.Size([40462, 12]) torch.Size([40462, 12])
            # print(preds.shape, mu.shape, logvar.shape, sample.shape)
            preds = torch.cat((sample, preds), dim=1)

        if return_targets:
            targets = torch.vstack(targets)
            return preds, targets
        else:
            return preds
    
    def _train_model_forward(self, x_batch):
        self.model.train()
        self.model.zero_grad()
        x_batch = x_batch.to(self._device, non_blocking=True)
        y_pred = self.forward(x_batch)
        return y_pred

    def train_model_forward(self, x_batch, delta_batch, r_batch):
        delta_batch, r_batch = delta_batch.to(self._device, non_blocking=True), r_batch.to(self._device, non_blocking=True)
        y_pred = self._train_model_forward(x_batch)
        y_batch = torch.cat([delta_batch, r_batch.unsqueeze(dim=1)], dim=1) if self.reward_head else delta_batch
        loss = self.model.loss(y_pred, y_batch)
        return loss

    def get_prob(self, input_batch, output_batch, mus=None, logvars=None):
        assert self.is_probabilistic
        return self.model.get_prob(input_batch, output_batch, mus, logvars)
    
    def get_predictions(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        assert self.is_probabilistic
        return self.model.get_predictions(state, action, state_filter, action_filter)
    
    def get_next_state_reward(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter,
                              keep_logvar=False, deterministic=False, return_mean=False):
        # it does not make sense to set deterministic as True, since the env should be stochastic
        return self.model.get_next_state_reward(state, action, state_filter, action_filter, keep_logvar,
                                                deterministic, return_mean)
    
    def predict_state(self, state: np.array, action: np.array, state_filter, action_filter):
        state, action = torch.Tensor(state).to(self._device), torch.Tensor(action).to(self._device)
        nextstate, reward = self.get_next_state_reward(state, action, state_filter, action_filter)
        nextstate = nextstate.detach().cpu().numpy()
        if self.reward_head:
            reward = reward.detach().cpu().item()
        return nextstate, reward

