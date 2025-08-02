# Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization (CBOP)

## How to set up the virtual environment

Please refer to the original github repo: [CBOP](https://github.com/jihwan-jeong/CBOP).

## Reproducing the results

The following three steps are provided by the original repo.

1. Pretrain the dynamics ensemble
    ```
    # Set log_to_wandb to true to turn wandb on
    # Identify the d4rl environment with arguments "env" and "overrides.d4rl_config"
    python -m rlkit.examples.main algorithm=pretrain env=hopper overrides.d4rl_config=medium-v2 is_offline=true log_to_wandb=false overrides.policy_eval.train=false overrides.bc_prior.train=false overrides.dynamics.train=true save_dir=data/pretrain/hopper/medium overrides.dynamics.batch_size=1024 overrides.dynamics.ensemble_model.ensemble_size=30 seed=0
    ```

2. Pretrain the policy and the Q ensemble with behavior clone (BC) and policy evaluation (PE) respectively
    ```
    python -m rlkit.examples.main algorithm=pretrain env=hopper overrides.d4rl_config=medium-v2 is_offline=true log_to_wandb=false overrides.policy_eval.train=true overrides.bc_prior.train=true overrides.dynamics.train=false save_dir=data/pretrain/hopper/medium overrides.bc_prior.batch_size=1024 overrides.policy_eval.batch_size=1024 overrides.policy_eval.num_value_learning_repeat=5 algorithm.algorithm_cfg.num_total_epochs=100 seed=0
    ```

3. Reproduce CBOP results
    ```
    python -m rlkit.examples.main env=hopper overrides.d4rl_config=medium-v2 algorithm=mvepo is_offline=true log_to_wandb=false overrides.trainer_cfg.horizon=10 overrides.trainer_cfg.num_qfs=50 algorithm.algorithm_cfg.num_epochs=1000 overrides.trainer_cfg.lcb_coeff=3. overrides.trainer_cfg.lr=3e-4 overrides.dynamics.ensemble_model.ensemble_size=30 overrides.dynamics.num_elites=20 cache_dir=data/pretrain/hopper/medium overrides.trainer_cfg.sampling_method=min overrides.offline_cfg.checkpoint_type=behavior overrides.trainer_cfg.indep_sampling=false snapshot_mode=gap_and_last algorithm.algorithm_cfg.save_snapshot_gap=30 overrides.trainer_cfg.eta=1
    ```
However, these are only for reproducing the results on Hopper-medium. **We have provided a complete list of commands to run CBOP on all 12 D4RL MuJoCo tasks (following the 3 steps above) to reproduce the corresponding results in Table 1.** You need to change the seed number in each command to get results corresponding to each seed of [0, 1, 2]. Accordingly, you also need to change any `dir' in the command to the seed number that you have set.
