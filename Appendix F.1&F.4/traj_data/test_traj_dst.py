import os
import gym, d4rl
import numpy as np
import pickle

def main(file_name):
    current_directory = os.path.dirname(__file__)
    # Load the traj list from the file using pickle
    with open(current_directory+'/'+file_name, 'rb') as file:
        trajs = pickle.load(file)
    
    env_name = file_name.split('_')[0]
    if '.' in env_name:
        env_name = env_name.split('.')[0]
    env = gym.make(env_name)
    
    state_error, reward_error, ori_reward_error = 0.0, 0.0, 0.0
    N = 0
    traj_lengths = []
    traj_rets = []
    for traj in trajs:
        states = traj['state']
        actions = traj['action']
        rewards = traj['reward']
        ori_rewards = traj['ori_reward']
        traj_lengths.append(len(actions))
        env.reset()
        x_qpos = env.sim.data.qpos[:1]
        init_state = np.concatenate([x_qpos, states[0]], axis=0)
        env.set_state(qpos=init_state[:len(init_state)//2], qvel=init_state[len(init_state)//2:])
        ret = 0.0
        for i in range(len(actions)):
            action = actions[i]
            next_s, r, d, _ = env.step(action)
            state_error += np.mean(np.square(next_s - states[i+1]))
            reward_error += np.square(rewards[i] - r)
            ori_reward_error += np.square(ori_rewards[i] - r)
            N += 1
            ret += r
            if d:
                break
        traj_rets.append(ret)

    print(state_error/float(N), reward_error/float(N), ori_reward_error/float(N))
    print(np.mean(traj_lengths), np.mean(traj_rets))


if __name__ == "__main__":
    # main('halfcheetah-medium-replay-v2_seed_2.pkl')
    # main('halfcheetah-medium-replay-v2_uniform_prior_seed_2.pkl')
    # main("halfcheetah-medium-replay-v2_uniform_prior_ensemble_seed_2.pkl")

    # main('halfcheetah-medium-v2_seed_1.pkl')
    # main('halfcheetah-medium-v2_uniform_prior_seed_1.pkl')
    # main("halfcheetah-medium-v2_uniform_prior_ensemble_seed_1.pkl")

    # main('halfcheetah-random-v2_seed_2.pkl')
    # main('halfcheetah-random-v2_uniform_prior_seed_2.pkl')
    # main("halfcheetah-random-v2_uniform_prior_ensemble_seed_2.pkl")
    
    # main('halfcheetah-medium-expert-v2_seed_2.pkl')
    # main('halfcheetah-medium-expert-v2_uniform_prior_seed_2.pkl')
    # main('halfcheetah-medium-expert-v2_uniform_prior_ensemble_seed_2.pkl')

    # main('hopper-medium-expert-v2_seed_2.pkl')
    # main('hopper-medium-expert-v2_uniform_prior_seed_2.pkl')
    # main('hopper-medium-expert-v2_uniform_prior_ensemble_seed_2.pkl')

    main('hopper-medium-replay-v2_seed_2.pkl')
    main('hopper-medium-replay-v2_uniform_prior_seed_2.pkl')
    main('hopper-medium-replay-v2_uniform_prior_ensemble_seed_2.pkl')

    # main('hopper-medium-v2_seed_2.pkl')
    # main('hopper-medium-v2_uniform_prior_seed_2.pkl')
    # main('hopper-medium-v2_uniform_prior_ensemble_seed_2.pkl')

    # main('hopper-random-v2_seed_2.pkl')
    # main('hopper-random-v2_uniform_prior_seed_2.pkl')
    # main('hopper-random-v2_uniform_prior_ensemble_seed_2.pkl')

    # main('walker2d-medium-expert-v2_seed_2.pkl')
    # main('walker2d-medium-expert-v2_uniform_prior_seed_2.pkl')
    # main('walker2d-medium-expert-v2_uniform_prior_ensemble_seed_2.pkl')

    # main('walker2d-medium-v2_seed_2.pkl')
    # main('walker2d-medium-v2_uniform_prior_seed_2.pkl')
    # main('walker2d-medium-v2_uniform_prior_ensemble_seed_2.pkl')

    # main('walker2d-random-v2_seed_2.pkl')
    # main('walker2d-random-v2_uniform_prior_seed_2.pkl')
    # main('walker2d-random-v2_uniform_prior_ensemble_seed_2.pkl')