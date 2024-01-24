import sys 

sys.path.append('/Users/sauravanchlia/Fair_ML/Projects/FairnessUnderLabelBias')

import os 

import numpy as np
import pandas as pd


from bin.utils import read_config, load_csv
from bin.dataset import Dataset
from bin.noise import Noise
from collections import defaultdict
# from calculate_fair_probability import fair_probability

config = read_config("configs/baseline_config.json")

# if 'dataset' not in config:
#     raise ValueError("Dataset not specified in config file.")
    
noise_types = ['bias', 'balanced', 'flip']
# noise_types = [ 'balanced']

noise_levels =[0.1,0.3]
datasets = {}
prob_path = "assets/probabilities/EmpiricalProbabilities"
base_path = 'assets/noisy_datasets'


for name, meta in config.items():
    ds = Dataset(meta)
    data = ds.data
    y = data[ds.label]
    sv = data[ds.sensitive_attribute]
    cols_to_drop = [ds.label]
    if ds.name in ['synthetic_20']:
        cols_to_drop.append('D')
    noisy_data = data.drop(cols_to_drop, axis = 1).copy()
    if ds.name == 'synthetic_20': data.drop(['D'], inplace=True, axis = 1)
    for noise in noise_types:
        for level in noise_levels:
            print(f"{name=}, {noise=}, {level=}")
            n_obj = Noise(noise, level, ds.majority_group, ds.positive_class, ds.label, ds.sensitive_attribute, data)
            y_noisy = n_obj.inject_noise(y, sv).reset_index(drop=True)
            noisy_data[ds.label] = y_noisy
            # calculate probabilities P(Y|X,S, Yobs)
            noisy_data_copy = noisy_data.copy()
            fair_df = n_obj.calculate_fair_probabilities(y, noisy_data_copy, noise)
            # y_cond = calculate_fair_probability(y,  noisy_data, ds.sensitive_attribute, ds.label)
            if name[:-3] == "synthetic":
                file_name = f"{name[-2:]}.csv"
            else:
                file_name = f"{name}_binerized.csv"
            # save the noisy_Data in a path 
            os.makedirs(os.path.join(base_path, name, noise, str(level)), exist_ok=True)
            noisy_data.to_csv(os.path.join(base_path, name, noise, str(level), file_name), index = False)
            fair_df.to_csv(os.path.join(base_path, name, noise, str(level), f"{name}_probs.csv"))
