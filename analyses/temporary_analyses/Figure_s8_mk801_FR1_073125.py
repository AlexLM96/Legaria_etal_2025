# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:27:06 2023

@author: Alex
"""

from pandas.api.types import CategoricalDtype
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import fed3bandit as f3b
import statsmodels.api as sm
import copy
import math
from scipy.stats import ttest_rel, ttest_ind

plt.rcParams.update({'font.size': 28, 'figure.autolayout': True})

cat_size_order = CategoricalDtype(["Sal", "MK801"], ordered=True)
figure_dir = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\icv_4ug_MK801_FR1\Figure_panels"

#%%

def count_pellets(data_choices):
    """Counts the number of pellets in fed3 data file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file

    Returns
    --------
    c_pellets : int
        total number of pellets
    """

    f_data_choices = f3b.filter_data(data_choices)
    pellet_count = f_data_choices["Pellet_Count"]
      
    c_diff = np.diff(pellet_count)
    c_diff2 = np.where(c_diff < 0, 1, c_diff)
    c_pellets = int(c_diff2.sum())
    
    return c_pellets

def win_stay(data_choices):
    """Calculates the win-stay probaility
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file

    Returns
    --------
    win_stay_p : int
        win-stay probability
    """
    f_data_choices = f3b.filter_data(data_choices)
    pellet_count = f_data_choices["Pellet_Count"]
    events = f_data_choices["Event"]
        
    win_stay = 0
    win_shift = 0
    for i in range(f_data_choices.shape[0]-1):
        c_choice = events.iloc[i]
        next_choice = events.iloc[i+1]
        c_count = pellet_count.iloc[i]
        next_count = pellet_count.iloc[i+1]
        if np.logical_or(next_count-c_count == 1, next_count-c_count < 0):
            c_outcome = 1
        else:
            c_outcome = 0
            
        if c_outcome == 1:
            if ((c_choice == "Left") and (next_choice == "Left")):
                win_stay += 1
            elif ((c_choice == "Right") and (next_choice == "Right")):
                win_stay += 1
            elif((c_choice == "Left") and (next_choice == "Right")):
                win_shift += 1
            elif((c_choice == "Right") and (next_choice == "Left")):
                win_shift += 1
                
    if (win_stay+win_shift) == 0:
        win_stay_p = np.nan
    else:
        win_stay_p = win_stay / (win_stay + win_shift)
    
    return win_stay_p

def lose_shift(data_choices):
    """Calculates the lose-shift probaility
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file

    Returns
    --------
    lose_shift_p : int
        lose-shift probability
    """
    f_data_choices = f3b.filter_data(data_choices)
    block_pellet_count = f_data_choices["Pellet_Count"]
    events = f_data_choices["Event"]
    
    lose_stay = 0
    lose_shift = 0
    for i in range(f_data_choices.shape[0]-1):
        c_choice = events.iloc[i]
        next_choice = events.iloc[i+1]
        c_count = block_pellet_count.iloc[i]
        next_count = block_pellet_count.iloc[i+1]
        if np.logical_or(next_count-c_count == 1, next_count-c_count == -19):
            c_outcome = 1
        else:
            c_outcome = 0
            
        if c_outcome == 0:
            if ((c_choice == "Left") and (next_choice == "Left")):
                lose_stay += 1
            elif ((c_choice == "Right") and (next_choice == "Right")):
                lose_stay += 1
            elif((c_choice == "Left") and (next_choice == "Right")):
                lose_shift += 1
            elif((c_choice == "Right") and (next_choice == "Left")):
                lose_shift += 1
                 
    if (lose_shift+lose_stay) == 0:
        lose_shift_p = np.nan
    else:
        lose_shift_p = lose_shift / (lose_shift + lose_stay)
    
    return lose_shift_p


#%%

#file_loc = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\icv_4ug_MK801_FR1\Data"
#file_loc = r"C:\Users\Alex\Downloads\Data_4ug_FR1\Data"
file_loc = r"C:\Users\Alex\Desktop\Legaria_etal_2025\data\temporary_data\Figure_s8"


saline_icv_data = {
    "saline_1": {
        "C109M1": pd.read_csv(file_loc + "\\saline_icv_FR1_C109M1_120423.csv"),
        "C109M2": pd.read_csv(file_loc + "\\saline_icv_FR1_C109M2_120423.csv")
        },
    "saline_2": {
        "C109M3": pd.read_csv(file_loc + "\\saline_icv_FR1_C109M3_121123.csv"),
        "C109M4": pd.read_csv(file_loc + "\\saline_icv_FR1_C109M4_121123.csv"),
        "C92M3": pd.read_csv(file_loc + "\\saline_icv_FR1_C92M3_121123.csv")
        }
    }

saline_icv_injectiontimes = {
    "saline_1": datetime.datetime(2023, 12, 4, 18, 50),
    "saline_2": datetime.datetime(2023, 12, 11, 20, 20)
    }

p_saline_icv_data = {}
for session in saline_icv_data:
    print(f"Processing {session}")
    c_session = saline_icv_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_saline_icv_data[session] = p_session

#Fixing wrong time in C84M1
p_saline_icv_data["saline_2"]["C109M3"].iloc[:,0] = p_saline_icv_data["saline_2"]["C109M3"].iloc[:,0] - datetime.timedelta(hours=1, minutes=10)

mk801_icv_data = {
    "mk801_1": {
        "C109M3": pd.read_csv(file_loc + "\\mk801_icv_FR1_C109M3_120423.csv"),
        "C109M4": pd.read_csv(file_loc + "\\\mk801_icv_FR1_C109M4_120423.csv"),
        "C92M3": pd.read_csv(file_loc + "\\\mk801_icv_FR1_C92M3_120423.csv")
        },
    "mk801_2": {
        "C109M1": pd.read_csv(file_loc + "\\mk801_icv_FR1_C109M1_121123.csv"),
        "C109M2": pd.read_csv(file_loc + "\\mk801_icv_FR1_C109M2_121123.csv")
        }
    }

mk801_icv_injectiontimes = {
    "mk801_1": datetime.datetime(2023, 12, 4, 23, 15),
    "mk801_2": datetime.datetime(2023, 12, 11, 20, 20)
    }

p_mk801_icv_data = {}
for session in mk801_icv_data:
    print(f"Processing {session}")
    c_session = mk801_icv_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_mk801_icv_data[session] = p_session

#Fixing wrong time in C84M1
p_mk801_icv_data["mk801_1"]["C109M3"].iloc[:,0] = p_mk801_icv_data["mk801_1"]["C109M3"].iloc[:,0] - datetime.timedelta(hours=1, minutes=10)


#Now loading the systemic data
saline_syst_data = {
    "saline_1": {
        "C109M1": pd.read_csv(file_loc + "\\C109M1_MK801_FR1_111723.csv"),
        "C109M2": pd.read_csv(file_loc + "\\C109M2_MK801_FR1_111723.csv"),
        },
    "saline_2": {
        "C109M3": pd.read_csv(file_loc + "\\C109M3_MK801_FR1_112523.csv"),
        "C109M4": pd.read_csv(file_loc + "\\C109M4_MK801_FR1_112523.csv"),
        "C92M3": pd.read_csv(file_loc + "\\C92M3_MK801_FR1_112523.csv")
        }
    }

saline_syst_injectiontimes = {
    "saline_1": datetime.datetime(2023, 11, 17, 22, 45),
    "saline_2": datetime.datetime(2023, 11, 25, 21, 30)
    }

p_saline_syst_data = {}
for session in saline_syst_data:
    print(f"Processing {session}")
    c_session = saline_syst_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_saline_syst_data[session] = p_session

#Fixing wrong time in C84M1
#p_saline_data["saline_2"]["C84M1"].iloc[:,0] = p_saline_data["saline_2"]["C84M1"].iloc[:,0] - datetime.timedelta(hours=1)

mk801_syst_data = {
    "mk801_1": {
        "C109M3": pd.read_csv(file_loc + "\\C109M3_MK801_FR1_111523.csv"),
        "C109M4": pd.read_csv(file_loc + "\\C109M4_MK801_FR1_111523.csv"),
        "C92M3": pd.read_csv(file_loc + "\\C92M3_MK801_FR1_111723.csv"),
        },
    "mk801_2": {
        "C109M1": pd.read_csv(file_loc + "\\C109M1_MK801_FR1_112523.csv"),
        "C109M2": pd.read_csv(file_loc + "\\C109M2_MK801_FR1_112523.csv")
        }
    }

mk801_syst_injectiontimes = {
    "mk801_1": datetime.datetime(2023, 11, 17, 22, 45),
    "mk801_2": datetime.datetime(2023, 11, 25, 21, 30)
    }

p_mk801_syst_data = {}
for session in mk801_syst_data:
    print(f"Processing {session}")
    c_session = mk801_syst_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_mk801_syst_data[session] = p_session


#%%

"""Here we get the slice of the data that goes from 0 to 6 hours after the injection"""

post_hours = 8

# Saline groups
saline_icv_slices = {}
for session in p_saline_icv_data:
    c_session = p_saline_icv_data[session]
    c_injection = saline_icv_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    saline_icv_slices[session] = s_session
                   
# MK801 groups                                     
mk801_icv_slices = {}
for session in p_mk801_icv_data:
    c_session = p_mk801_icv_data[session]
    c_injection = mk801_icv_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    mk801_icv_slices[session] = s_session


#BELOW IS SYSTEMIC DATASET

# Saline groups
saline_syst_slices = {}
for session in p_saline_syst_data:
    c_session = p_saline_syst_data[session]
    c_injection = saline_syst_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    saline_syst_slices[session] = s_session
                   
# MK801 groups                                     
mk801_syst_slices = {}
for session in p_mk801_syst_data:
    c_session = p_mk801_syst_data[session]
    c_injection = mk801_syst_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    mk801_syst_slices[session] = s_session


#%%

"""Here we get example traces"""

saline_icv_sample = saline_icv_slices["saline_1"]["C109M1"].iloc[:100,:]
saline_icv_bactions = f3b.binned_paction(saline_icv_sample)
saline_icv_true = np.ones(len(saline_icv_bactions))

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(saline_icv_true, c="red", linewidth=5)
ax.plot(saline_icv_bactions, c="darkcyan", linewidth=3)
ax.set_ylim(0,1)
plt.axis("off")
plt.savefig(figure_dir + "\\sample_saline.eps")

mk801_icv_sample = mk801_icv_slices["mk801_2"]["C109M1"].iloc[:100,:]
mk801_icv_bactions = f3b.binned_paction(mk801_icv_sample)
mk801_icv_true = np.ones(mk801_icv_sample.shape[0])

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(mk801_icv_true, c="red", linewidth=5)
ax.plot(mk801_icv_bactions, c="olive", linewidth=3)
ax.set_ylim(0,1)
plt.axis("off")
plt.savefig(figure_dir + "\\sample_icv_mk801.eps")


#%%
"""Here we get the number of pellets"""

# Count of pellets after saline injections
saline_icv_pellets = {}
for session in saline_icv_slices:
    c_session = saline_icv_slices[session]
    c_session_icv_pellets = {mouse: [count_pellets(c_session[mouse])] for mouse in c_session}
    saline_icv_pellets = saline_icv_pellets | c_session_icv_pellets
    
saline_icv_pellets = pd.DataFrame(saline_icv_pellets).T
    
    
# Count of pellets after mk801 injections
mk801_icv_pellets = {}
for session in mk801_icv_slices:
    c_session = mk801_icv_slices[session]
    c_session_pellets = {mouse: [count_pellets(c_session[mouse])] for mouse in c_session}
    mk801_icv_pellets = mk801_icv_pellets | c_session_icv_pellets
    
mk801_icv_pellets = pd.DataFrame(mk801_icv_pellets).T

# We pool all the numbers
all_icv_pellets = pd.concat([saline_icv_pellets, mk801_icv_pellets], axis=1).reset_index()
all_icv_pellets.columns = ["Mouse", "Sal", "MK801"]
m_all_icv_pellets = pd.melt(all_icv_pellets, id_vars="Mouse")
m_all_icv_pellets["variable"] = m_all_icv_pellets["variable"].astype(cat_size_order)
m_all_icv_pellets = m_all_icv_pellets.sort_values(by="variable")

#Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_icv_pellets, palette=[
            "darkcyan", "olive", "salmon"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_icv_pellets, palette=["silver", "silver", "silver"], s=10)
ax.set_ylabel("Pellets")
ax.set_xlabel("")
sns.despine()
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.set_ylim(0,180)
ax.set_yticks(np.arange(0,180,40))
#plt.savefig(figure_dir +  "\\pellets_4ug_icv_FR1_mk801.eps", bbox_inches="tight")

mk801_pellets_ttest = ttest_rel(m_all_icv_pellets["value"][m_all_icv_pellets["variable"]== "Sal"], 
                                m_all_icv_pellets["value"][m_all_icv_pellets["variable"] == "MK801"])

#%%

# Count of pellets after saline injections
saline_icv_pokes = {}
for session in saline_icv_slices:
    c_session = saline_icv_slices[session]
    c_session_icv_pokes = {mouse: [f3b.count_pokes(c_session[mouse])] for mouse in c_session}
    saline_icv_pokes = saline_icv_pokes | c_session_icv_pokes
    
saline_icv_pokes = pd.DataFrame(saline_icv_pokes).T

# Count of pokes after mk801 injections
mk801_icv_pokes = {}
for session in mk801_icv_slices:
    c_session = mk801_icv_slices[session]
    c_session_pokes = {mouse: [f3b.count_pokes(c_session[mouse])] for mouse in c_session}
    mk801_icv_pokes = mk801_icv_pokes | c_session_pokes
    
mk801_icv_pokes = pd.DataFrame(mk801_icv_pokes).T

# We pool all the numbers
all_icv_pokes = pd.concat([saline_icv_pokes, mk801_icv_pokes], axis=1).reset_index()
all_icv_pokes.columns = ["Mouse", "Sal", "MK801"]
m_all_icv_pokes = pd.melt(all_icv_pokes, id_vars="Mouse")
m_all_icv_pokes["variable"] = m_all_icv_pokes["variable"].astype(cat_size_order)
m_all_icv_pokes = m_all_icv_pokes.sort_values(by="variable")

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_icv_pokes, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_icv_pokes, palette=["silver", "silver"], s=10)
ax.set_ylabel("pokes")
ax.set_xlabel("")
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.set_ylim(0,185)
#ax.set_yticks(np.arange(0,201,40))
sns.despine()
plt.savefig(figure_dir +  "\\pokes_4ug_icv_FR1_mk801.eps", bbox_inches="tight")

mk801_pokes_ttest = ttest_rel(m_all_icv_pokes["value"][m_all_icv_pokes["variable"]== "Sal"], 
                                m_all_icv_pokes["value"][m_all_icv_pokes["variable"] == "MK801"])

#%%

"""Here we calculate the percentage of times they poke in the 'Active' poke"""

m_all_icv_ppp = copy.deepcopy(m_all_icv_pokes)
m_all_icv_ppp["value"] = m_all_icv_pokes["value"] / m_all_icv_pellets["value"]

left_icv_prcnt = copy.deepcopy(m_all_icv_ppp)
left_icv_prcnt["value"] = 1/left_icv_prcnt["value"] * 100

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=left_icv_prcnt, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=left_icv_prcnt, palette=["silver", "silver"], s=10)
ax.set_ylabel("%Left")
ax.set_xlabel("")
ax.set_yticks(np.arange(50,100,20))
sns.despine()
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.set_ylim(45,100)
plt.savefig(figure_dir +  "\\leftpokes_4ug_icv_FR1_mk801.eps", bbox_inches="tight")

left_percent_icv_ttest = ttest_rel(left_icv_prcnt["value"][left_icv_prcnt["variable"]== "Sal"], left_icv_prcnt["value"][left_icv_prcnt["variable"] == "MK801"])

#%%

"""Here we look for the win-stay"""

saline_icv_ws = {}
for session in saline_icv_slices:
    c_session = saline_icv_slices[session]
    c_session_ws = {mouse: [win_stay(c_session[mouse])] for mouse in c_session}
    saline_icv_ws = saline_icv_ws | c_session_ws
    
saline_icv_ws = pd.DataFrame(saline_icv_ws).T

# Count of ws after mk801 injections
mk801_icv_ws = {}
for session in mk801_icv_slices:
    c_session = mk801_icv_slices[session]
    c_session_ws = {mouse: [win_stay(c_session[mouse])] for mouse in c_session}
    mk801_icv_ws = mk801_icv_ws | c_session_ws
    
mk801_icv_ws = pd.DataFrame(mk801_icv_ws).T

# We pool all the numbers
all_icv_ws = pd.concat([saline_icv_ws, mk801_icv_ws], axis=1).reset_index()
all_icv_ws.columns = ["Mouse", "Sal", "MK801"]
m_all_icv_ws = pd.melt(all_icv_ws, id_vars="Mouse")
m_all_icv_ws["variable"] = m_all_icv_ws["variable"].astype(cat_size_order)
m_all_icv_ws = m_all_icv_ws.sort_values(by="variable")

fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_icv_ws, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_icv_ws, palette=["silver", "silver"], s=10, jitter=True)
ax.set_ylabel("Win-stay")
ax.set_xlabel("")
ax.set_ylim(0,1)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
plt.savefig(figure_dir +  "\\winstay_4ug_icv_FR1_mk801.eps", bbox_inches="tight")

# Do paired ttest
mk801_ws_ttest = ttest_rel(m_all_icv_ws["value"][m_all_icv_ws["variable"]== "Sal"], m_all_icv_ws["value"][m_all_icv_ws["variable"] == "MK801"], alternative="greater")

#%%

"""Hereon we analyze the systemic dataset"""

saline_syst_sample = saline_syst_slices["saline_1"]["C109M2"].iloc[:,:]
saline_syst_bactions = f3b.binned_paction(saline_syst_sample)
saline_syst_true = np.ones(len(saline_syst_bactions))

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(saline_syst_true, c="red", linewidth=5)
ax.plot(saline_syst_bactions, c="darkcyan", linewidth=3)
ax.set_ylim(0,1)
plt.axis("off")
plt.savefig(figure_dir + "\\Figure_s8_systemic_sample_saline.eps")

mk801_syst_sample = mk801_syst_slices["mk801_2"]["C109M2"].iloc[:,:]
mk801_syst_bactions = f3b.binned_paction(mk801_syst_sample)
mk801_syst_true = np.ones(mk801_syst_sample.shape[0])

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(mk801_syst_true, c="red", linewidth=5)
ax.plot(mk801_syst_bactions, c="olive", linewidth=3)
ax.set_ylim(0,1)
plt.axis("off")
plt.savefig(figure_dir + "\\Figure_s8_systemic_sample_mk801.eps")

#%%

"""Here we get the number of pellets"""

# Count of pellets after saline injections
saline_syst_pellets = {}
for session in saline_syst_slices:
    c_session = saline_syst_slices[session]
    c_session_pellets = {mouse: [count_pellets(c_session[mouse])] for mouse in c_session}
    saline_syst_pellets = saline_syst_pellets | c_session_pellets
    
saline_syst_pellets = pd.DataFrame(saline_syst_pellets).T
    
    
# Count of pellets after mk801 injections
mk801_syst_pellets = {}
for session in mk801_syst_slices:
    c_session = mk801_syst_slices[session]
    c_session_pellets = {mouse: [count_pellets(c_session[mouse])] for mouse in c_session}
    mk801_syst_pellets = mk801_syst_pellets | c_session_pellets
    
mk801_syst_pellets = pd.DataFrame(mk801_syst_pellets).T

# We pool all the numbers
all_syst_pellets = pd.concat([saline_syst_pellets, mk801_syst_pellets], axis=1).reset_index()
all_syst_pellets.columns = ["Mouse", "Sal", "MK801"]
m_all_syst_pellets = pd.melt(all_syst_pellets, id_vars="Mouse")
m_all_syst_pellets["variable"] = m_all_syst_pellets["variable"].astype(cat_size_order)
m_all_syst_pellets = m_all_syst_pellets.sort_values(by="variable")

#Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_syst_pellets, palette=[
            "darkcyan", "olive", "salmon"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_syst_pellets, palette=["silver", "silver", "silver"], s=10)
ax.set_ylabel("Pellets")
ax.set_xlabel("")
sns.despine()
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
#ax.set_ylim(0,180)
ax.set_yticks(np.arange(0,180,40))
plt.savefig(figure_dir +  "\\pellets_ip_FR1_mk801.eps", bbox_inches="tight")

mk801_pellets_ttest = ttest_rel(m_all_syst_pellets["value"][m_all_syst_pellets["variable"]== "Sal"], 
                                m_all_syst_pellets["value"][m_all_syst_pellets["variable"] == "MK801"])

#%%

# Count of pellets after saline injections
saline_syst_pokes = {}
for session in saline_syst_slices:
    c_session = saline_syst_slices[session]
    c_session_pokes = {mouse: [f3b.count_pokes(c_session[mouse])] for mouse in c_session}
    saline_syst_pokes = saline_syst_pokes | c_session_pokes
    
saline_syst_pokes = pd.DataFrame(saline_syst_pokes).T

# Count of pokes after mk801 injections
mk801_syst_pokes = {}
for session in mk801_syst_slices:
    c_session = mk801_syst_slices[session]
    c_session_pokes = {mouse: [f3b.count_pokes(c_session[mouse])] for mouse in c_session}
    mk801_syst_pokes = mk801_syst_pokes | c_session_pokes
    
mk801_syst_pokes = pd.DataFrame(mk801_syst_pokes).T

# We pool all the numbers
all_syst_pokes = pd.concat([saline_syst_pokes, mk801_syst_pokes], axis=1).reset_index()
all_syst_pokes.columns = ["Mouse", "Sal", "MK801"]
m_all_syst_pokes = pd.melt(all_syst_pokes, id_vars="Mouse")
m_all_syst_pokes["variable"] = m_all_syst_pokes["variable"].astype(cat_size_order)
m_all_syst_pokes = m_all_syst_pokes.sort_values(by="variable")

mk801_pokes_ttest = ttest_rel(m_all_syst_pokes["value"][m_all_syst_pokes["variable"]== "Sal"], 
                              m_all_syst_pokes["value"][m_all_syst_pokes["variable"] == "MK801"])

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_syst_pokes, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_syst_pokes, palette=["silver", "silver"], s=10)
ax.set_ylabel("pokes")
ax.set_xlabel("")
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
#ax.set_ylim(0,210)
#ax.set_yticks(np.arange(0,201,40))
sns.despine()
plt.savefig(figure_dir +  "\\pokes_ip_FR1_mk801.eps", bbox_inches="tight")

#%%

m_all_syst_ppp = copy.deepcopy(m_all_syst_pokes)
m_all_syst_ppp["value"] = m_all_syst_pokes["value"] / m_all_syst_pellets["value"]

left_syst_prcnt = copy.deepcopy(m_all_syst_ppp)
left_syst_prcnt["value"] = 1/left_syst_prcnt["value"] * 100

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=left_syst_prcnt, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=left_syst_prcnt, palette=["silver", "silver"], s=10)
ax.set_ylabel("%Left")
ax.set_xlabel("")
#ax.set_yticks(np.arange(0,100,20))
sns.despine()
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.set_ylim(0,100)
plt.savefig(figure_dir +  "\\leftprcnt_ip_FR1_mk801.eps", bbox_inches="tight")

left_percent_syst_ttest = ttest_rel(left_syst_prcnt["value"][left_syst_prcnt["variable"]== "Sal"], left_syst_prcnt["value"][left_syst_prcnt["variable"] == "MK801"])

#%%

"""Here we look for the win-stay"""

saline_syst_ws = {}
for session in saline_syst_slices:
    c_session = saline_syst_slices[session]
    c_session_ws = {mouse: [win_stay(c_session[mouse])] for mouse in c_session}
    saline_syst_ws = saline_syst_ws | c_session_ws
    
saline_syst_ws = pd.DataFrame(saline_syst_ws).T

# Count of ws after mk801 injections
mk801_syst_ws = {}
for session in mk801_syst_slices:
    c_session = mk801_syst_slices[session]
    c_session_ws = {mouse: [win_stay(c_session[mouse])] for mouse in c_session}
    mk801_syst_ws = mk801_syst_ws | c_session_ws
    
mk801_syst_ws = pd.DataFrame(mk801_syst_ws).T

# We pool all the numbers
all_syst_ws = pd.concat([saline_syst_ws, mk801_syst_ws], axis=1).reset_index()
all_syst_ws.columns = ["Mouse", "Sal", "MK801"]
m_all_syst_ws = pd.melt(all_syst_ws, id_vars="Mouse")
m_all_syst_ws["variable"] = m_all_syst_ws["variable"].astype(cat_size_order)
m_all_syst_ws = m_all_syst_ws.sort_values(by="variable")

fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_syst_ws, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_syst_ws, palette=["silver", "silver"], s=10, jitter=True)
ax.set_ylabel("Win-stay")
ax.set_xlabel("")
ax.set_ylim(0,1)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
plt.savefig(figure_dir +  "\\ws_ip_FR1_mk801.eps", bbox_inches="tight")

# Do paired ttest
mk801_ws_syst_ttest = ttest_rel(m_all_syst_ws["value"][m_all_syst_ws["variable"]== "Sal"], m_all_syst_ws["value"][m_all_syst_ws["variable"] == "MK801"])
