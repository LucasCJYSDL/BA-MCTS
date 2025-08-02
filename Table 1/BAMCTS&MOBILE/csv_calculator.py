import os
import pandas as pd
import numpy as np

# Path to the main folder
current_working_directory = os.getcwd()
main_folder_path = current_working_directory + '/log'

# Dictionary to store DataFrames for each CSV
dataframes = {}

# Iterate through each subfolder in the main folder
for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)
    
    for subsubfolder in os.listdir(subfolder_path):
        subsubfolder_path = subfolder_path + '/' + subsubfolder 
        for subsubsubfolder in os.listdir(subsubfolder_path):
            file_name = subsubfolder_path + '/' + subsubsubfolder + '/record/policy_training_progress.csv'
            # print(file_name)
            dataframes[subsubfolder_path + '/' + subsubsubfolder] = pd.read_csv(file_name)


# Display the DataFrames
tot_list = []
for k, df in dataframes.items():
    # print(f"DataFrame for subfolder {k}:")
    # print(df.head())  # Show the first few rows
    # print(df['eval/normalized_episode_reward'].to_numpy()[-10:])
    tot_list.append((k, np.mean(df['eval/normalized_episode_reward'].to_numpy()[-10:])))

sorted(tot_list, key = lambda x: x[0])
for i in range(len(tot_list)):
    print(tot_list[i][0])
    if (i+1)%3 == 0:
        
        score_list = [e[1] for e in tot_list[i-2:i+1]]
        print(score_list, np.mean(score_list), np.std(score_list))
        print("\n")
