import os
import numpy as np
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main_1(file_name):
    sns.set_theme(style="dark", font_scale=1.8)

    current_directory = os.path.dirname(__file__)
    # Load the traj list from the file using pickle
    with open(current_directory+'/'+file_name, 'rb') as file:
        trajs = pickle.load(file)

    offline_belief = []
    for traj in trajs:
        offline_belief.append(list(traj['prior'][0]))

    offline_belief_ends = []
    for i in range(len(offline_belief)):
        if offline_belief[i][0] == offline_belief[i][1]:
            offline_belief_ends.append(i)

    # print(offline_belief_ends)
    # print(len(offline_belief))
    # print(offline_belief[offline_belief_ends[0]: offline_belief_ends[1]])

    data_1 = offline_belief[offline_belief_ends[0]: offline_belief_ends[0] + 100]
    df = pd.DataFrame(data_1, columns=['model {}'.format(i+1) for i in range(len(data_1[0]))])
    # print(df)
    # Melt the DataFrame to have a long-form structure suitable for Seaborn
    df_melted = df.reset_index().melt(id_vars='index', var_name='model', value_name='Value')
    # print(df_melted)
    # Draw the line plot
    plt.figure(figsize=(7.5, 5))
    plt.grid(visible=True)
    sns.lineplot(data=df_melted, x='index', y='Value', hue='model', legend=False, linewidth=3)
    plt.xlabel("Time Step")
    plt.ylabel("Belief")
    plt.tight_layout()
    plt.savefig("./offline_belief_change.png")

    t_id = 200 # 0, 22, 495
    rollout_index = offline_belief_ends[0] + t_id
    rollout_belief = trajs[rollout_index]['prior']
    data_2 = [list(b) for b in rollout_belief]

    df = pd.DataFrame(data_2, columns=['model {}'.format(i+1) for i in range(len(data_2[0]))])
    # print(df)
    # Melt the DataFrame to have a long-form structure suitable for Seaborn
    df_melted = df.reset_index().melt(id_vars='index', var_name='model', value_name='Value')
    # print(df_melted)
    # Draw the line plot
    plt.figure(figsize=(7.5, 5))
    plt.grid(visible=True)
    sns.lineplot(data=df_melted, x='index', y='Value', hue='model', legend=False, linewidth=3)
    plt.xlabel("Time Step")
    plt.ylabel("Belief")
    plt.tight_layout()
    plt.savefig("./online_belief_change_{}.png".format(t_id))


def main_2(file_name, t_id):
    sns.set_theme(style="dark", font_scale=1.8)

    current_directory = os.path.dirname(__file__)
    # Load the traj list from the file using pickle
    with open(current_directory+'/'+file_name, 'rb') as file:
        trajs = pickle.load(file)

    offline_belief = []
    for traj in trajs:
        offline_belief.append(list(traj['prior'][0]))

    offline_belief_ends = []
    for i in range(len(offline_belief)):
        if offline_belief[i][0] == offline_belief[i][1]:
            offline_belief_ends.append(i)


    rollout_index = offline_belief_ends[0] + t_id
    rollout_belief = trajs[rollout_index]['prior']
    data_2 = [list(b) for b in rollout_belief]

    df = pd.DataFrame(data_2, columns=['model {}'.format(i+1) for i in range(len(data_2[0]))])
    # print(df)
    # Melt the DataFrame to have a long-form structure suitable for Seaborn
    df_melted = df.reset_index().melt(id_vars='index', var_name='model', value_name='Value')
    # print(df_melted)
    # Draw the line plot
    plt.figure(figsize=(7.5, 5))
    plt.grid(visible=True)
    sns.lineplot(data=df_melted, x='index', y='Value', hue='model', legend=False, linewidth=3)
    plt.xlabel("Time Step")
    plt.ylabel("Belief")
    plt.tight_layout()
    plt.savefig("./online_belief_change_{}.png".format(t_id))


if __name__ == "__main__":
    main_1('hopper-medium-expert-v2_seed_0_prior.pkl')
    main_2('hopper-medium-expert-v2_seed_0_prior.pkl', t_id=0)