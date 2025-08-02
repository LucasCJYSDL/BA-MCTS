import os, math
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from buffer import FasterReplayPool, Transition, SLFasterReplayPool
from nn_model import Policy, DoubleQFunc

class SAC_Agent:

    def __init__(self, seed, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256,
                 update_interval=1, buffer_size=1e6, target_entropy=None, device='cpu', scheduler=None):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -action_dim / 2.0
        self.batchsize = batchsize
        self.update_interval = update_interval # it's weird to set the interval as 1, but it's from the original implementation
        self._device = device
        torch.manual_seed(seed)

        # critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(self._device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(self._device)

        # temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self.alpha = self.log_alpha.exp()

        # optimizer
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_pool = FasterReplayPool(action_dim=action_dim, state_dim=state_dim, capacity=buffer_size)
        self.scheduler = scheduler

        self._decay_step = 100 
        self._decay_factor = 0.5
    
    def get_action_and_value(self, nextstate_batch):
        with torch.no_grad():
            # TODO: using real entropy
            # nextaction_batch, entropy_batch = self.policy.forward_entropy(nextstate_batch)
            nextaction_batch, logprobs_batch, _, logits = self.policy(nextstate_batch, get_logprob=True, get_logits=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            q_target = torch.min(q_t1, q_t2)
        # TODO: using self.alpha is dangerous, which could be replaced with a constant, or self.scheduler.value
        return logits, q_target, - self.gamma * self.alpha * logprobs_batch # dereived based on the SAC objective

    def sl_update_q_functions(self, state_batch, action_list_batch, action_dist_batch, q_batch):
        # torch.Size([256, 11]) torch.Size([256, 20, 3]) torch.Size([256, 20]) torch.Size([256, 20])
        max_action_num = action_list_batch.shape[1]
        state_batch = state_batch.unsqueeze(1).repeat(1, max_action_num, 1).view(-1, state_batch.shape[-1])
        action_list_batch = action_list_batch.view(-1, action_list_batch.shape[-1])
        action_dist_batch = action_dist_batch.view(action_list_batch.shape[0], 1)
        # note that q_batch has involved log_prob (as in update_q_functions) via the tree search backtraverse
        q_batch = q_batch.view(action_list_batch.shape[0], 1) 

        q_1, q_2 = self.q_funcs(state_batch, action_list_batch)
        # print(state_batch.shape, q_1.shape, q_2.shape) # torch.Size([5120, 11]) torch.Size([5120, 1]) torch.Size([5120, 1])
        mask = (action_dist_batch > 0.0).to(dtype=torch.float32, device=q_1.device)

        loss_1 = (torch.square(q_1 - q_batch) * mask).sum() / mask.sum()
        loss_2 = (torch.square(q_2 - q_batch) * mask).sum() / mask.sum()
        
        return loss_1, loss_2
    
    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        # print(q_1.shape, q_2.shape, value_target.shape) # torch.Size([256, 1]) torch.Size([256, 1]) torch.Size([256, 1])
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2
    
    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.log_alpha.exp() * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss
    
    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def sl_update_policy_and_temp(self, state_batch, action_list_batch, action_dist_batch, action_num):
        max_action_num = int(np.max(action_num))

        # mu, logstd = self.policy.stable_network_forward(state_batch)
        # std = logstd.exp()
        _, logprobs_batch, _, logits = self.policy(state_batch, get_logprob=True, get_logits=True)
        mu, std = logits[0], logits[1]
        dist = Independent(Normal(mu, std), 1)

        # torch.Size([256, 20]) torch.Size([256, 20, 3])
        target_normalized_visit_count = action_dist_batch
        target_sampled_actions = action_list_batch

        # TODO: use logprobs_batch instead
        # policy_entropy_loss = -dist.entropy().mean()
        policy_entropy_loss = logprobs_batch.mean()

        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
        log_prob_sampled_actions = []
        num_sampled_actions = target_normalized_visit_count.shape[-1]
        batch_size = target_normalized_visit_count.shape[0]

        for k in range(max_action_num):
            # SAC-like
            y = 1 - target_sampled_actions[:, k, :].pow(2)
            # NOTE: for numerical stability.
            min_val = torch.tensor(-1 + 1e-6).to(target_sampled_actions.device)
            max_val = torch.tensor(1 - 1e-6).to(target_sampled_actions.device)
            target_sampled_actions_clamped = torch.clamp(target_sampled_actions[:, k, :], min_val, max_val)
            target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)

            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = dist.log_prob(target_sampled_actions_before_tanh).unsqueeze(-1)
            # TODO: remove this lome
            log_prob = log_prob - torch.log(y + 1e-6).sum(-1, keepdim=True)
            log_prob = log_prob.squeeze(-1)
                
            log_prob_sampled_actions.append(log_prob)

        log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1) # torch.Size([256, 20])
    
        if max_action_num < num_sampled_actions:
            supplement = torch.zeros((batch_size, num_sampled_actions-max_action_num), dtype=torch.float32, device=target_sampled_actions.device)
            log_prob_sampled_actions = torch.cat([log_prob_sampled_actions, supplement], dim=-1)
        
        log_prob_sampled_actions[target_normalized_visit_count==0.0] = np.log(1e-6)
        # print(log_prob_sampled_actions.shape, log_prob_sampled_actions[0]) # torch.Size([256, 20])
        
        # normalize the prob of sampled actions
        prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / (torch.exp(log_prob_sampled_actions).\
                                     sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach() + 1e-6)
        log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)
        # prepare the mask for optimization
        mask = (target_normalized_visit_count > 0.0).to(dtype=torch.float32, device=target_sampled_actions.device)

        # cross_entropy loss: - sum(p * log (q))
        policy_loss = - (torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions * mask).sum() / batch_size

        temp_loss = -self.log_alpha.exp() * (logprobs_batch.detach() + self.target_entropy).mean()

        return policy_loss, policy_entropy_loss, temp_loss


    def sl_optimize(self, n_updates, sl_buffer: SLFasterReplayPool, policy_only=True):
        q1_loss, q2_loss, pi_loss, pi_ent_loss, a_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        for i in tqdm(range(n_updates)):
            samples = sl_buffer.sample(self.batchsize)
            state_batch = torch.FloatTensor(np.array(samples.state)).to(self._device)
            action_list_batch = torch.FloatTensor(np.array(samples.action_list)).to(self._device)
            action_dist_batch = torch.FloatTensor(np.array(samples.action_dist)).to(self._device)

            pi_loss_step, pi_ent_loss_step, a_loss_step = self.sl_update_policy_and_temp(state_batch, action_list_batch, action_dist_batch, samples.action_num)

            self.policy_optimizer.zero_grad()
            # TODO: decouple self.alpha and pi_ent_loss_step, self.alpha.detach()， self.scheduler.value * pi_ent_loss_step
            (pi_loss_step + self.scheduler.value * pi_ent_loss_step).backward()
            self.policy_optimizer.step()

            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()

            pi_loss += pi_loss_step.detach().item()
            pi_ent_loss += pi_ent_loss_step.detach().item()
            a_loss += a_loss_step.detach().item()

            if not policy_only:
                q_batch = torch.FloatTensor(np.array(samples.q)).to(self._device)
                q1_loss_step, q2_loss_step = self.sl_update_q_functions(state_batch, action_list_batch, action_dist_batch, q_batch)
                q_loss_step = q1_loss_step + q2_loss_step
                self.q_optimizer.zero_grad()
                q_loss_step.backward()
                self.q_optimizer.step()

                q1_loss += q1_loss_step.detach().item()
                q2_loss += q2_loss_step.detach().item()
            
                if i % self.update_interval == 0:
                    self.update_target()

        self.alpha = self.log_alpha.exp()
        self.scheduler.decrease()

        return pi_loss, pi_ent_loss, a_loss, q1_loss, q2_loss

    def optimize(self, n_updates, state_filter=None, env_pool=None, env_ratio=0.05, value_only=False, epoch_num=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0
        hide_progress = True if n_updates < 50 else False

        # print(n_updates, env_ratio) # 1000 0.05
        for i in tqdm(range(n_updates), disable=hide_progress, ncols=100):
            if env_pool and env_ratio != 0:
                n_env_samples = int(env_ratio * self.batchsize)
                n_model_samples = self.batchsize - n_env_samples
                env_samples = env_pool.sample(n_env_samples)._asdict()
                model_samples = self.replay_pool.sample(n_model_samples)._asdict()

                samples = Transition(*[env_samples[key] + model_samples[key] for key in env_samples])
            else:
                samples = self.replay_pool.sample(self.batchsize)
            

            if state_filter:
                state_batch = torch.FloatTensor(state_filter(np.array(samples.state))).to(self._device)
                nextstate_batch = torch.FloatTensor(state_filter(np.array(samples.nextstate))).to(self._device)
            else:
                state_batch = torch.FloatTensor(np.array(samples.state)).to(self._device)
                nextstate_batch = torch.FloatTensor(np.array(samples.nextstate)).to(self._device)

            action_batch = torch.FloatTensor(np.array(samples.action)).to(self._device)
            reward_batch = torch.FloatTensor(np.array(samples.reward)).to(self._device).unsqueeze(1)
            done_batch = torch.FloatTensor(np.array(samples.real_done)).to(self._device).unsqueeze(1)

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch,
                                                                 nextstate_batch, done_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            if not value_only:
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step, a_loss_step = self.update_policy_and_temp(state_batch)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True

                pi_loss += pi_loss_step.detach().item()
                a_loss += a_loss_step.detach().item()

            self.alpha = self.log_alpha.exp()
            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()
            
            if i % self.update_interval == 0:
                self.update_target()
        
        if epoch_num is not None:
            if (epoch_num+1) % self._decay_step == 0:
                for pg in self.policy_optimizer.param_groups:
                    pg['lr'] *= self._decay_factor
                for pg in self.q_optimizer.param_groups:
                    pg['lr'] *= self._decay_factor
                for pg in self.temp_optimizer.param_groups:
                    pg['lr'] *= self._decay_factor

        return q1_loss, q2_loss, pi_loss, a_loss
    
    def get_action(self, state, state_filter=None, deterministic=False, oac=False):
        if state_filter:
            state = state_filter(state)
        state = torch.Tensor(state).view(1, -1).to(self._device)
        if oac:
            action, _, mean = self._get_optimistic_action(state)
        else:
            with torch.no_grad():
                action, _, mean = self.policy(state)
        if deterministic:
            return np.atleast_1d(mean.squeeze().cpu().numpy())
        return np.atleast_1d(action.squeeze().cpu().numpy())
    
    def _get_optimistic_action(self, state, get_logprob=False):

        beta_UB = 4.66  # Table 1: https://arxiv.org/pdf/1910.12807.pdf
        delta = 23.53  # Table 1: https://arxiv.org/pdf/1910.12807.pdf

        mu, logvar = self.policy.stable_network_forward(state)
        mu.requires_grad_()
        std = logvar.exp()

        action = torch.tanh(mu)
        q_1, q_2 = self.q_funcs(state, action)

        mu_Q = (q_1 + q_2) / 2.0

        sigma_Q = torch.abs(q_1 - q_2) / 2.0

        Q_UB = mu_Q + beta_UB * sigma_Q

        grad = torch.autograd.grad(Q_UB, mu)
        grad = grad[0]

        grad = grad.detach()
        mu = mu.detach()
        std = std.detach()

        Sigma_T = torch.pow(std.detach(), 2)
        denom = torch.sqrt(
            torch.sum(torch.mul(torch.pow(grad, 2), Sigma_T))) + 10e-6

        # Obtain the change in mu
        mu_C = math.sqrt(2.0 * delta) * torch.mul(Sigma_T, grad) / denom

        mu_E = mu + mu_C

        assert mu_E.shape == std.shape
        return self.policy.compute_action(mu_E, std, get_logprob=get_logprob)
    
    def save_policy(self, save_path, num_epochs, rew=None):
        q_funcs, target_q_funcs, policy, log_alpha = self.q_funcs, self.target_q_funcs, self.policy, self.log_alpha

        if rew is None:
            save_path = os.path.join(save_path, "torch_policy_weights_{}_epochs.pt".format(num_epochs))
        else:
            save_path = os.path.join(save_path, "torch_policy_weights_{}_epochs_{}.pt".format(num_epochs, rew))

        torch.save({
            'double_q_state_dict': q_funcs.state_dict(),
            'target_double_q_state_dict': target_q_funcs.state_dict(),
            'policy_state_dict': policy.state_dict(),
            'log_alpha_state_dict': log_alpha
        }, save_path)
