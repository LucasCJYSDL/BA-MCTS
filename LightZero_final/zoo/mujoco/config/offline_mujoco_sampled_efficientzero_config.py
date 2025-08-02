from easydict import EasyDict
import d4rl

env_id = 'halfcheetah-medium-expert-v2'
# env_id = 'halfcheetah-medium-replay-v2'
# env_id = 'halfcheetah-medium-v2'
# env_id = 'halfcheetah-random-v2'

# env_id = 'hopper-medium-expert-v2'
# env_id = 'hopper-medium-replay-v2'
# env_id = 'hopper-medium-v2'
# env_id = 'hopper-random-v2'

# env_id = 'walker2d-medium-expert-v2'
# env_id = 'walker2d-medium-replay-v2'
# env_id = 'walker2d-medium-v2'
# env_id = 'walker2d-random-v2'

if 'cheetah' in env_id:
    action_space_size = 6
    observation_shape = 17
elif 'hopper' in env_id:
    action_space_size = 3
    observation_shape = 11
else:
    assert 'walker' in env_id
    action_space_size = 6
    observation_shape = 17

ignore_done = False # danger
if 'halfcheetah' in env_id:
    # for halfcheetah, we ignore done signal to predict the Q value of the last step correctly.
    ignore_done = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8 # 8
n_episode = 8 # 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 200
batch_size = 256

max_env_step = int(3e6)
if env_id in ['walker2d-random-v2', 'hopper-random-v2']:
    max_env_step = int(1e6)

reanalyze_ratio = 0.0 # too slow
policy_entropy_loss_weight = 0.005

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

offline_mujoco_sampled_efficientzero_config = dict(
    exp_name=f'data_sez/{env_id[:-3]}_offline_sampled_efficientzero_ns{num_simulations}_ne{n_episode}_upc{update_per_collect}_rer{reanalyze_ratio}_bs_{batch_size}_pelw{policy_entropy_loss_weight}_seed{seed}',
    env=dict(
        env_id=env_id,
        action_clip=True,
        continuous=True,
        offline=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
        ),
        cuda=True,
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        ignore_done=ignore_done,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        discount_factor=0.997,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        # replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        offline=True,
        use_root_value=False,
    ),
)

offline_mujoco_sampled_efficientzero_config = EasyDict(offline_mujoco_sampled_efficientzero_config)
main_config = offline_mujoco_sampled_efficientzero_config

offline_mujoco_sampled_efficientzero_create_config = dict(
    env=dict(
        type='offline_mujoco_lightzero',
        import_names=['zoo.mujoco.envs.offline_mujoco_lightzero_env'],
        eval_type='mujoco_lightzero',
        eval_import_names=['zoo.mujoco.envs.mujoco_lightzero_env']
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='offline_sampled_efficientzero',
        import_names=['lzero.policy.offline_sampled_efficientzero'],
    ),
)
offline_mujoco_sampled_efficientzero_create_config = EasyDict(offline_mujoco_sampled_efficientzero_create_config)
create_config = offline_mujoco_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    import warnings, os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Suppress a specific warning by category
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore")
    os.environ['DISPLAY'] = ':0'
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)