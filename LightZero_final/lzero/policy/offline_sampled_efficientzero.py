from .sampled_efficientzero import SampledEfficientZeroPolicy
from ding.utils import POLICY_REGISTRY

import numpy as np
import torch

from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSPtree

from lzero.policy import ez_network_output_unpack, select_action

@POLICY_REGISTRY.register('offline_sampled_efficientzero')
class OfflineSampledEfficientZeroPolicy(SampledEfficientZeroPolicy):
    """
    Overview:
        The policy class for offline Sampled EfficientZero
    """

    # The default_config for Sampled EfficientZero policy.
    config = dict(
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For 3-dimensional image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) the stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=True,
            # (int) The size of action space. For discrete action space, it is the number of actions.
            # For continuous action space, it is the dimension of action.
            action_space_size=6,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            categorical_distribution=True,
            # (int) the image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The scale of supports used in categorical distribution.
            # This variable is only effective when ``categorical_distribution=True``.
            support_scale=300,
            # (int) The hidden size in LSTM.
            lstm_hidden_size=512,
            # (str) The type of sigma. options={'conditioned', 'fixed'}
            sigma_type='conditioned',
            # (float) The fixed sigma value. Only effective when ``sigma_type='fixed'``.
            fixed_sigma_value=0.3,
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (str) The type of action encoding. Options are ['one_hot', 'not_one_hot']. Default to 'one_hot'.
            discrete_action_encoding_type='one_hot',
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'LN'.
            norm_type='BN',
        ),
        # ****** common ******
        # (bool) Whether to use multi-gpu training.
        multi_gpu=False,
        # (bool) ``sampled_algo=True`` means the policy is sampled-based algorithm (e.g. Sampled EfficientZero), which is used in ``collector``.
        sampled_algo=True,
        # (bool) Whether to enable the gumbel-based algorithm (e.g. Gumbel Muzero)
        gumbel_algo=False,
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda in policy.
        cuda=True,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (str) The type of environment. The options are ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of action space. Options are ['fixed_action_space', 'varied_action_space'].
        action_type='fixed_action_space',
        # (str) The type of battle mode. The options are ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment``.
        game_segment_length=200,
        # (bool): Indicates whether to perform an offline evaluation of the checkpoint (ckpt).
        # If set to True, the checkpoint will be evaluated after the training process is complete.
        # IMPORTANT: Setting eval_offline to True requires configuring the saving of checkpoints to align with the evaluation frequency.
        # This is done by setting the parameter learn.learner.hook.save_ckpt_after_iter to the same value as eval_freq in the train_muzero.py automatically.
        eval_offline=False,

        # ****** observation ******
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use gray scale image.
        gray_scale=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) The style of augmentation.
        augmentation=['shift', 'intensity'],

        # ****** learn ******
        # (bool) Whether to ignore the done flag in the training data. Typically, this value is set to False.
        # However, for some environments with a fixed episode length, to ensure the accuracy of Q-value calculations,
        # we should set it to True to avoid the influence of the done flag.
        ignore_done=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.replay_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        replay_ratio=0.25,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam', 'AdamW']
        optim_type='SGD',
        learning_rate=0.2,  # init lr for manually decay schedule
        # optim_type='Adam',
        # lr_piecewise_constant_decay=False,
        # learning_rate=0.003,  # lr for Adam optimizer
        # (float) Weight uniform initialization range in the last output layer
        init_w=3e-3,
        normalize_prob_of_sampled_actions=False,
        policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        # (int) Frequency of target network update.
        target_update_freq=100,
        weight_decay=1e-4,
        momentum=0.9,
        grad_clip_value=10,
        # You can use either "n_sample" or "n_episode" in collector.collect.
        # Get "n_episode" episodes per collect.
        n_episode=8,
        # (float) the number of simulations in MCTS.
        num_simulations=50,
        # (float) Discount factor (gamma) for returns.
        discount_factor=0.997,
        # (int) The number of step for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=5,
        # (int) reset the hidden states in LSTM every ``lstm_horizon_len`` horizon steps.
        lstm_horizon_len=5,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of policy entropy loss.
        policy_entropy_loss_weight=0,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=2,
        # (bool) Whether to use the cosine learning rate decay.
        cos_lr_scheduler=False,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (bool) Whether to use manually decayed temperature.
        # i.e. temperature: 1 -> 0.5 -> 0.25
        manual_temperature_decay=False,
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to use the true chance in MCTS in some environments with stochastic dynamics, such as 2048.
        use_ture_chance_label_in_chance_encoder=False,
        # (bool) Whether to add noise to roots during reanalyze process.
        reanalyze_noise=True,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=True,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,

        # ****** UCB ******
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        # (int) The number of episodes to collect data randomly before training.
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=False,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
    )

    def _forward_collect(
            self, data: torch.Tensor, action_mask: list = None, temperature: np.ndarray = 1, to_play=-1,
            epsilon: float = 0.25, offline_action_dict = {}, ready_env_id: np.array = None,
    ):
        """
        Overview:
            The forward function for collecting data in collect mode. Use model to execute MCTS search.
            Returning the action in the offline data.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - temperature (:obj:`float`): The temperature of the policy.
            - to_play (:obj:`int`): The player to play.
            - offline_action_dict (:obj:`dict`): the stored action dictionary.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """

        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}
        
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output
            )

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            reward_hidden_state_roots = (
                reward_hidden_state_roots[0].detach().cpu().numpy(),
                reward_hidden_state_roots[1].detach().cpu().numpy()
            )
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(active_collect_env_num)
                ]
            else:
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                ]

            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(
                    active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(
                    active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )

            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(self._cfg.model.num_of_sampled_actions)
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            # print(len(noises[0]), value_prefix_roots, self._cfg.root_noise_weight, len(policy_logits), len(policy_logits[0]))
            # 20 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.25 8 6

            roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
            self._mcts_collect.search(
                roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
            )

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}
            roots_sampled_actions = roots.get_sampled_actions()  # {list: 1}->{list:6}

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                if self._cfg.mcts_ctree:
                    # In ctree, the method roots.get_sampled_actions() returns a list object.
                    root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])
                else:
                    # In ptree, the same method roots.get_sampled_actions() returns an Action object.
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])

                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self._collect_mcts_temperature, deterministic=False
                )

                # if self._cfg.mcts_ctree:
                #     # In ctree, the method roots.get_sampled_actions() returns a list object.
                #     action = np.array(roots_sampled_actions[i][action])
                # else:
                #     # In ptree, the same method roots.get_sampled_actions() returns an Action object.
                #     action = roots_sampled_actions[i][action].value

                # key change
                action = offline_action_dict[env_id]
                # [-0.85976851  0.43879399 -0.43610477] offline: [0.9801126  0.9287043  0.99919546]

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }

        return output