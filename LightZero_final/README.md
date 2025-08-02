# How to Reproduce the Evaluation Results on D4RL MuJoCo using Sampled EfficientZero

## Required environments:
- on Ubuntu 20.04
- Python 3.9.19
- mujoco_py 2.1.2
- d4rl 1.1
- mjrl 1.0
- [LightZero](https://github.com/opendilab/LightZero)

## How to run the experiments

- Modify the 'get_vec_env_setting' function in 'ding/envs/env/base_env.py' (i.e., a file in the installed DI-engine package) as:
    ```bash
    def get_vec_env_setting(cfg: dict, collect: bool = True, eval_: bool = True) -> Tuple[type, List[dict], List[dict]]:
        """
        Overview:
            Get vectorized env setting (env_fn, collector_env_cfg, evaluator_env_cfg).
        Arguments:
            - cfg (:obj:`dict`): Original input env config in user config, such as ``cfg.env``.
        Returns:
            - env_fn (:obj:`type`): Callable object, call it with proper arguments and then get a new env instance.
            - collector_env_cfg (:obj:`List[dict]`): A list contains the config of collecting data envs.
            - evaluator_env_cfg (:obj:`List[dict]`): A list contains the config of evaluation envs.

        .. note::
            Elements (env config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.

        """
        import_module(cfg.get('import_names', []))
        env_fn = ENV_REGISTRY.get(cfg.type)

        if cfg.get('offline', False):
            import_module(cfg.get('eval_import_names', []))
            eval_env_fn = ENV_REGISTRY.get(cfg.eval_type)

        collector_env_cfg = env_fn.create_collector_env_cfg(cfg) if collect else None
        if cfg.get('offline', False):
            evaluator_env_cfg = eval_env_fn.create_evaluator_env_cfg(cfg) if eval_ else None
        else:
            evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg) if eval_ else None

        if cfg.get('offline', False):
            return env_fn, eval_env_fn, collector_env_cfg, evaluator_env_cfg
        return env_fn, collector_env_cfg, evaluator_env_cfg
    ```

- Enter the corresponding folder:
    ```bash
    cd zoo/mujoco/config
    ```

- Specify the environment to run in 'offline_mujoco_sampled_efficientzero_config.py' (Line 4 -- Line 17);

- Run the main file:
    ```bash
    python offline_mujoco_sampled_efficientzero_config.py
    ```