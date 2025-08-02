# Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning

## Required environments:
- on Ubuntu 20.04
- Python 3.9.19
- mujoco_py 2.1.2
- [LightZero](https://github.com/opendilab/LightZero)
- The ones listed in 'requirements.txt'

## How to run the experiments

- Please download the dynamics and reward models used for [BA-MBRL](https://drive.google.com/drive/folders/1FlzAaJkOs7WKM73kJmpYkelFMVdShiYL?usp=sharing), [BA-MCTS](https://drive.google.com/drive/folders/14MxBjHX9eaiO5xCOjyUDSiqFMS0EKa2y?usp=sharing), [BA-MCTS-SL](https://drive.google.com/drive/folders/14MxBjHX9eaiO5xCOjyUDSiqFMS0EKa2y?usp=sharing), to the folder 'data'.

- An example command to run the program:

    ```bash
    python main.py --seed X --yaml_file args_yaml/Y/Z.yml --uuid E --load_model_dir F
    ```
    - X is the random seed for which we choose from [0, 1, 2];

    - Y is the folder to keep the yaml files, which can be one of [ba_marl, ba_mcts, ba_mcts_sl];

    - Z is the name of the specific yaml file, as listed in 'args_yaml/ba_marl';

    - E specifies the name of the subfolder to store the training results;

    - F is the path to the dynamics and reward model for model-based RL; if the 'load_model_dir' is not specified, the program will learn a new dynamics and reward model for the current environment.

- **You may need to compile the mcts component (written in C++) to get the program running:**
    ```bash
    cd ctree
    python setup.py build_ext --inplace
    ```

## About the experiments on tokamak control

- The execution of these experiments relies on operational data from DIII-D, which is protected. As a result, we are unable to release the corresponding code until we obtain the necessary approvals.