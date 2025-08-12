# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 12:33:47 2025

@author: alexmacal
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd
import datetime
#import fed3bandit as f3b
import copy

file_loc = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025\data\temporary_data\Figure_3"

#%%

def neg_log_likelihood_single_alpha_bias(params, choices, rewards):
    alpha, beta, bias = params
    if not (0 <= alpha <= 1 and beta > 0):
        return np.inf

    Q = np.zeros(2)
    log_likelihood = 0

    for action, reward in zip(choices, rewards):
        logits = beta * Q + np.array([0.0, bias])  # apply bias to action 1 only
        probs = softmax(logits)
        log_likelihood += np.log(probs[action] + 1e-10)

        Q[action] += alpha * (reward - Q[action])

    return -log_likelihood

#
def fit_single_alpha_with_bias(choices, rewards, initial_guess=(0.5, 1.0, 0.0)):
    bounds = [(0, 1), (1e-3, 20), (-5, 5)]  # bias can be negative or positive
    result = minimize(
        neg_log_likelihood_single_alpha_bias,
        initial_guess,
        args=(choices, rewards),
        bounds=bounds
    )
    return result.x, result.fun

#
def neg_log_likelihood_two_alpha_bias(params, choices, rewards):
    alpha_pos, alpha_neg, beta, bias = params
    if not (0 <= alpha_pos <= 1 and 0 <= alpha_neg <= 1 and beta > 0):
        return np.inf

    Q = np.zeros(2)
    log_likelihood = 0

    for action, reward in zip(choices, rewards):
        logits = beta * Q + np.array([0.0, bias])  # bias added to action 1
        probs = softmax(logits)
        log_likelihood += np.log(probs[action] + 1e-10)

        alpha = alpha_pos if reward == 1 else alpha_neg
        Q[action] += alpha * (reward - Q[action])

    return -log_likelihood

#
def fit_two_alpha_with_bias(choices, rewards, initial_guess=(0.6, 0.2, 3.0, 0.0)):
    bounds = [(1e-3, 0.99), (1e-3, 0.99), (1e-3, 10), (-5, 5)]  # alpha_pos, alpha_neg, beta, bias
    result = minimize(
        neg_log_likelihood_two_alpha_bias,
        initial_guess,
        args=(choices, rewards),
        bounds=bounds
    )
    return result.x, result.fun

def prepare_data(data_choices):
    events = ["Right", "Left"]
    f_file = data_choices[data_choices["Event"].isin(events)]
    
    choices = []
    rewards = []
    for i in range(f_file.shape[0]-1):
        c_row = f_file.iloc[i,:]
        c_choice = c_row["Event"]
        if c_choice == "Left":
            choices.append(1)
        elif c_choice == "Right":
            choices.append(0)
        else:
            print("Error")
        
        c_pellet_count = c_row["Pellet_Count"]
        n_row = f_file.iloc[i+1,:]
        n_pellet_count = n_row["Pellet_Count"]
        if (n_pellet_count - c_pellet_count) == 1:
            rewards.append(1)
        elif (n_pellet_count - c_pellet_count) == 0:
            rewards.append(0)
        else:
            print("Error pellet number")
            
    return (choices, rewards)


#%%

saline_data = {
    "C116F3": pd.read_csv(file_loc + "\\C116F3_intrastriatal_saline_4ug_053023.csv"),
    "C48M3": pd.read_csv(file_loc + "\\C48M3_intrastriatal_saline_4ug_060123.csv"),
    "C50F5": pd.read_csv(file_loc + "\\C50F5_intrastriatal_saline_4ug_060523.csv"),
    "C116F1": pd.read_csv(file_loc + "\\C116F1_intrastriatal_saline_4ug_060523.csv"),
    "C86F1": pd.read_csv(file_loc + "\\C86F1_intrastriatal_saline_4ug_092523.csv"),
    "C86F2": pd.read_csv(file_loc + "\\C86F2_intrastriatal_saline_4ug_092523.csv"),
    "C129F1": pd.read_csv(file_loc + "\\C129F1_intrastriatal_saline_4ug_100423.csv"),
    "C84M2": pd.read_csv(file_loc + "\\C84M2_intrastriatal_saline_4ug_100423.csv"),
    "FCK4M1": pd.read_csv(file_loc + r"\FCK4M1_intrastriatal_saline_4ug_111224.CSV"),
    "FCK4M3": pd.read_csv(file_loc + r"\FCK4M3_intrastriatal_saline_4ug_111224.CSV"),
    "FCK3F1": pd.read_csv(file_loc + r"\FCK3F1_intrastriatal_saline_4ug_111324.CSV"),
    "FCK4M2": pd.read_csv(file_loc + r"\FCK4M2_intrastriatal_saline_4ug_111924.CSV"),
    "FCK3F3": pd.read_csv(file_loc + r"\FCK3F3_intrastriatal_saline_4ug_111924.CSV"),
    "FCK5M3": pd.read_csv(file_loc + r"\FCK5M3_intrastriatal_saline_4ug_010925.CSV"),
    "FCK5M4": pd.read_csv(file_loc + r"\FCK5M4_intrastriatal_saline_4ug_010925.CSV"),
}

mk801_data = {
    "C50F5": pd.read_csv(file_loc + "\\C50F5_intrastriatal_mk801_4ug_053023.csv"),
    "C116F2": pd.read_csv(file_loc + "\\C116F2_intrastriatal_mk801_4ug_053023.csv"),
    "C116F4": pd.read_csv(file_loc + "\\C116F4_intrastriatal_mk801_4ug_053023.csv"),
    "C75M1": pd.read_csv(file_loc + "\\C75M1_intrastriatal_mk801_4ug_060123.csv"),
    "C75M2": pd.read_csv(file_loc + "\\C75M2_intrastriatal_mk801_4ug_060123.csv"),
    "C75M3": pd.read_csv(file_loc + "\\C75M3_intrastriatal_mk801_4ug_060123.csv"),
    "C129F1": pd.read_csv(file_loc + "\\C129F1_intrastriatal_mk801_4ug_092523.csv"),
    "C84M2": pd.read_csv(file_loc + "\\C84M2_intrastriatal_mk801_4ug_092523.csv"),
    "C86F1": pd.read_csv(file_loc + "\\C86F1_intrastriatal_mk801_4ug_100423.csv"),
    "C86F2": pd.read_csv(file_loc + "\\C86F2_intrastriatal_mk801_4ug_100423.csv"),
    "FCK4M1": pd.read_csv(file_loc + r"\FCK4M1_intrastriatal_mk801_4ug_112524.CSV"),
    "FCK4M3": pd.read_csv(file_loc + r"\FCK4M3_intrastriatal_mk801_4ug_112524.CSV"),
    "FCK3F1": pd.read_csv(file_loc + r"\FCK3F1_intrastriatal_mk801_4ug_111824.CSV"),
    "FCK4M2": pd.read_csv(file_loc + r"\FCK4M2_intrastriatal_mk801_4ug_112524.CSV"),
    "FCK3F3": pd.read_csv(file_loc + r"\FCK3F3_intrastriatal_mk801_4ug_112524.CSV"),
    "FCK5M3": pd.read_csv(file_loc + r"\FCK5M3_intrastriatal_mk801_4ug_011725.csv"),
    "FCK5M4": pd.read_csv(file_loc + r"\FCK5M4_intrastriatal_mk801_4ug_011625.csv"),
}

#%%

saline_params = []
for mouse in saline_data:
    t_file = saline_data[mouse]
    t_file[t_file.columns[0]] = pd.to_datetime(t_file[t_file.columns[0]])
    c_file = t_file[t_file.iloc[:,0] < (t_file.iloc[0,0]+datetime.timedelta(hours=8))]
    
    f_file = prepare_data(c_file)
    
    params, nll = fit_single_alpha_with_bias(f_file[0], f_file[1])
    
    # ---------- Print Results ----------
    alpha_pos, beta, bias = params
    saline_params.append([mouse, alpha_pos, beta, bias])
    
saline_params_df = pd.DataFrame(saline_params, columns=["Mouse", "alpha", "beta", "bias"])


#%%

mk801_params = []
for mouse in mk801_data:
    t_file = mk801_data[mouse]
    t_file[t_file.columns[0]] = pd.to_datetime(t_file[t_file.columns[0]])
    c_file = t_file[t_file.iloc[:,0] < (t_file.iloc[0,0]+datetime.timedelta(hours=8))]
    
    f_file = prepare_data(c_file)
    
    params, nll = fit_single_alpha_with_bias(f_file[0], f_file[1])
    
    # ---------- Print Results ----------
    alpha_pos, beta, bias = params
    mk801_params.append([mouse, alpha_pos, beta, bias])
    
mk801_params_df = pd.DataFrame(mk801_params, columns=["Mouse", "alpha", "beta", "bias"])

#%%

fig, ax = plt.subplots()
ax.scatter(np.ones_like(saline_params_df["alpha"].to_numpy()), saline_params_df["alpha"].to_numpy())
ax.scatter(np.ones_like(mk801_params_df["alpha"].to_numpy())+1, mk801_params_df["alpha"].to_numpy())
ax.set_xlim(0,4)
    

    