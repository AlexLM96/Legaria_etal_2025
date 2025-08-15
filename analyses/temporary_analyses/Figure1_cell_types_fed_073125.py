# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:52:10 2025

@author: Alex
"""


import pickle
import invian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings
from scipy.signal import find_peaks, periodogram, spectrogram, filtfilt, butter, find_peaks_cwt, correlate, welch, peak_prominences, resample
from scipy.stats import ttest_rel, ttest_ind

figure_dir = r"C:\Users\Alex\Desktop\Legaria_etal_2025\Figures\Figure_panels"
fileloc = r"C:\Users\Alex\Desktop\Legaria_etal_2025\data\temporary_data\Figure_1\FED"

plt.rcParams.update({'font.size': 32, 'figure.autolayout': True, 'lines.linewidth': 2})

events_fixed = False

#%%

#
def pulse_split(event_ts, max_gap=0.3):
    counter = 1
    peak_dict = {}
    time_diff = np.diff(event_ts)
    i=0
    while True:
        
        c_diff = time_diff[i]
        if c_diff > max_gap:  
            if counter not in peak_dict.keys():
                peak_dict[counter] = []
            
            else:
                pass
            
            peak_dict[counter].append(event_ts[i])
            
            if i < (len(time_diff)-1):
                counter = 1
                i += 1
                
            else:
                break
        
        else:
            while c_diff < max_gap:
                counter += 1
                if i < (len(time_diff)-1):
                    i += 1
                    c_diff = time_diff[i]
                    
                else:
                    
                    break
                
        if not i < (len(time_diff)-1):   
            break

    return peak_dict

#Temp functions
def butter_filter(signal, filt_type, freqs, sr, order=3):
    b,a = butter(order, freqs, btype=filt_type, fs=sr)
    y=filtfilt(b, a, signal, padtype="even")
    
    return y

#Peri-event histogram for continuous values.
def contvar_peh(var_ts, var_vals, ref_ts, min_max, bin_width = False):
    r"""
    Function to perform a peri-event histogram of spiking activity.
    
    Parameters
    ----------
    var_ts : array-like
        Photometry signal timestamps
    var_vals : array-like
        Photometry signal values
    ref_ts : array-like
        Reference events that spiking will be aligned to
    min_max : tuple
        Time window in seconds around ref_ts to be analyzed in seconds. E.g. (-4,8)
    bin_width : float
        Bin width of histogram in seconds

    Returns
    ---------
    all_trials : 2d-array
        Continuous variable values around each timestamp in ref_ts in bin_width wide bins
    """
    if not isinstance(var_ts,np.ndarray):
        try:
            var_ts = np.array(var_ts)          
        except:
            raise TypeError(f"Expected spike_ts to be of type: array-like but got {type(var_ts)} instead")
        
    if not isinstance(var_vals,np.ndarray):
        try:
            var_vals = np.array(var_vals)          
        except:
            raise TypeError(f"Expected var_vals to be of type: array-like but got {type(var_vals)} instead")
    
    if not isinstance(ref_ts,np.ndarray):
        try:
            ref_ts = np.array(ref_ts)     
        except:
            raise TypeError(f"Expected spike_ts to be of type: array-like but got {type(ref_ts)} instead")
    
    if bin_width:
        ds_ts = np.linspace(var_ts.min(), var_ts.max(), int((var_ts.max()-var_ts.min())/bin_width))
        ds_vals = resample(var_vals, ds_ts.size)
            
        #ds_vals = np.interp(ds_ts, var_ts, var_vals)
        rate = bin_width
    
    else:
        rate = np.diff(var_ts).mean()
        ds_ts, ds_vals = (np.array(var_ts), np.array(var_vals))       
        
    left_idx = int(min_max[0]/rate)
    right_idx = int(min_max[1]/rate)

    event_ts = ref_ts[np.logical_and((ref_ts + min_max[0]) > ds_ts[0], (ref_ts + min_max[1]) < ds_ts[-1])]
    if len(event_ts) < len(ref_ts):
        event_diff = len(ref_ts) - len(event_ts)
        warnings.warn(f"The trial time range of {event_diff} events is outside the timestamps range. Events were ommitted.")
        
    
    all_idx = np.searchsorted(ds_ts,event_ts, "right")   
    all_trials = np.vstack([ds_vals[idx+left_idx:idx+right_idx] for idx in all_idx])
    
    return all_trials

#%%


"""
Loading all the data
"""

raw_data = {
    "D1-Cre": {
        "Saline": {
            "F488": pd.read_csv(fileloc + r"\saline_D1_F488_bandit_031825.csv"),
            "M690": pd.read_csv(fileloc + r"\saline_D1_M690_bandit_040325.csv"),
            "M700": pd.read_csv(fileloc + r"\saline_D1_M700_bandit_040325.csv"),
            "M780": pd.read_csv(fileloc + r"\saline_D1_M780_bandit_051325.csv"),
            "F797": pd.read_csv(fileloc + r"\saline_D1_F797_bandit_051325.csv"),
            },
        "MK801": {
            "F488": pd.read_csv(fileloc + r"\mk801_D1_F488_bandit_040425.csv"),
            "M690": pd.read_csv(fileloc + r"\mk801_D1_M690_bandit_042925.csv"),
            "M700": pd.read_csv(fileloc + r"\mk801_D1_M700_bandit_042525.csv"),
            "M780": pd.read_csv(fileloc + r"\mk801_D1_M780_bandit_060925.csv"),
            "F797": pd.read_csv(fileloc + r"\mk801_D1_F797_bandit_060925.csv")
            }
        },
    "A2a-Cre": {
        "Saline": {
            "M521": pd.read_csv(fileloc + r"\saline_A2a_M521_bandit_011025.csv"),
            "C139M1": pd.read_csv(fileloc + r"\saline_A2a_C139M1_bandit_041425.csv"),
            "C139M2": pd.read_csv(fileloc + r"\saline_A2a_C139M2_bandit_041425.csv"),
            "C139M4": pd.read_csv(fileloc + r"\saline_A2a_C139M4_bandit_040725.csv")
            },
        "MK801": {
            "M521": pd.read_csv(fileloc + r"\mk801_A2a_M521_bandit_030625.csv"),
            "C139M1": pd.read_csv(fileloc + r"\mk801_A2a_C139M1_bandit_050225.csv"),
            "C139M4": pd.read_csv(fileloc + r"\mk801_A2a_C139M4_bandit_042425.csv"),
            "C139M2": pd.read_csv(fileloc + r"\mk801_A2a_C139M2_bandit_042525.csv")
            }
        }
    }

events = {
    "D1-Cre": {
        "Saline": {
            "F488": pd.read_csv(fileloc + r"\events_saline_D1_F488_bandit_031825.csv", index_col=0),
            "M690": pd.read_csv(fileloc + r"\events_saline_D1_M690_bandit_040325.csv", index_col=0),
            "M700": pd.read_csv(fileloc + r"\events_saline_D1_M700_bandit_040325.csv", index_col=0),
            "M780": pd.read_csv(fileloc + r"\events_saline_D1_M780_bandit_051325.csv", index_col=0),
            "F797": pd.read_csv(fileloc + r"\events_saline_D1_F797_bandit_051325.csv", index_col=0)
            },
        "MK801": {
            "F488": pd.read_csv(fileloc + r"\events_mk801_D1_F488_bandit_040425.csv", index_col=0),
            "M690": pd.read_csv(fileloc + r"\events_mk801_D1_M690_bandit_042925.csv", index_col=0),
            "M700": pd.read_csv(fileloc + r"\events_mk801_D1_M700_bandit_042525.csv", index_col=0),
            "M780": pd.read_csv(fileloc + r"\events_mk801_D1_M780_bandit_060925.csv", index_col=0),
            "F797": pd.read_csv(fileloc + r"\events_mk801_D1_F797_bandit_060925.csv", index_col=0)
            }
        },
    "A2a-Cre": {
        "Saline": {
            "M521": pd.read_csv(fileloc + r"\events_saline_A2a_M521_bandit_011025.csv", index_col=0),
            "C139M1": pd.read_csv(fileloc + r"\events_saline_A2a_C139M1_bandit_041425.csv", index_col=0),
            "C139M2": pd.read_csv(fileloc + r"\events_saline_A2a_C139M2_bandit_041425.csv", index_col=0),
            "C139M4": pd.read_csv(fileloc + r"\events_saline_A2a_C139M4_bandit_040725.csv", index_col=0)
            },
        "MK801": {
            "M521": pd.read_csv(fileloc + r"\events_mk801_A2a_M521_bandit_030625.csv", index_col=0),
            "C139M4": pd.read_csv(fileloc + r"\events_mk801_A2a_C139M4_bandit_042425.csv", index_col=0),
            "C139M1": pd.read_csv(fileloc + r"\events_mk801_A2a_C139M1_bandit_050225.csv", index_col=0),
            "C139M2": pd.read_csv(fileloc + r"\events_mk801_A2a_C139M2_bandit_042525.csv", index_col=0)
            }
        }
    }

metadata = pd.read_csv(r"C:\Users\Alex\Desktop\Legaria_etal_2025\data\temporary_data\Figure_1\FED\all_metadata_081425.csv")

#%%

#Check the all the raw data looks okay
strain = "D1-Cre"
treatment = "Saline"
mouse = "M700"

c_t_data = raw_data[strain][treatment][mouse].copy(deep=True)
fig, ax = plt.subplots()
ax.plot(c_t_data["Time"], c_t_data["Fluorescence"])

fig, ax = plt.subplots()
ax.plot(c_t_data["Time"], c_t_data["Isosbestic"])

#%%

"""
Pre-processing
"""

#We first find the timestamps of each behavioral event.
rekey = {1: "Left_poke", 2: "Right_poke", 3: "Rewarded", 4: "Unrewarded", 5: "Pellet_drop", 6: "Pellet_retrieval"}
p_events = {}
for strain in events:
    p_events[strain] = {}
    for treatment in events[strain]:
        p_events[strain][treatment] = {}
        for mouse in events[strain][treatment]:
            print(treatment, mouse)
            c_events = events[strain][treatment][mouse].to_numpy().squeeze()
            c_split = pulse_split(c_events)
            c_n_split = dict((rekey[key], value) for key,value in c_split.items())
            p_events[strain][treatment][mouse] = c_n_split
            
        for mouse in events[strain][treatment]:
            c_left_pokes = p_events[strain][treatment][mouse]["Left_poke"]
            c_right_pokes = p_events[strain][treatment][mouse]["Right_poke"]
            p_events[strain][treatment][mouse]["Pokes"] = c_left_pokes + c_right_pokes
            
#%%

"""
Fix some recordings where pellet drop and pellet retrieval both have 6 pulses
"""

<<<<<<< HEAD
if not events_fixed:
    p_events["D1-Cre"]["Saline"]["M690"]["Pellet_drop"] = p_events["D1-Cre"]["Saline"]["M690"]["Pellet_retrieval"][::2]
    p_events["D1-Cre"]["Saline"]["M690"]["Pellet_retrieval"] = p_events["D1-Cre"]["Saline"]["M690"]["Pellet_retrieval"][1::2]
    
    p_events["D1-Cre"]["Saline"]["M700"]["Pellet_drop"] = p_events["D1-Cre"]["Saline"]["M700"]["Pellet_retrieval"][::2]
    p_events["D1-Cre"]["Saline"]["M700"]["Pellet_retrieval"] = p_events["D1-Cre"]["Saline"]["M700"]["Pellet_retrieval"][1::2]
    
    p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_drop"] = p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_retrieval"][::2]
    p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_retrieval"] = p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_retrieval"][1::2]
    
    p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_drop"] = p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_retrieval"][::2]
    p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_retrieval"] = p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_retrieval"][1::2]
    
    p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_drop"] = p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_retrieval"][::2]
    p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_retrieval"] = p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_retrieval"][1::2]
    
    p_events["D1-Cre"]["MK801"]["M700"]["Pellet_drop"] = p_events["D1-Cre"]["MK801"]["M700"]["Pellet_retrieval"][::2]
    p_events["D1-Cre"]["MK801"]["M700"]["Pellet_retrieval"] = p_events["D1-Cre"]["MK801"]["M700"]["Pellet_retrieval"][1::2]
    
    p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_drop"] = p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_retrieval"][::2]
    p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_retrieval"] = p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_retrieval"][1::2]
    
    events_fixed = True
    
else:
    print("This cell has already been run. Processing was not applied. Applying it again will disrupt the data")
=======
p_events["D1-Cre"]["Saline"]["M690"]["Pellet_drop"] = p_events["D1-Cre"]["Saline"]["M690"]["Pellet_retrieval"][::2]
p_events["D1-Cre"]["Saline"]["M690"]["Pellet_retrieval"] = p_events["D1-Cre"]["Saline"]["M690"]["Pellet_retrieval"][1::2]

p_events["D1-Cre"]["Saline"]["M700"]["Pellet_drop"] = p_events["D1-Cre"]["Saline"]["M700"]["Pellet_retrieval"][::2]
p_events["D1-Cre"]["Saline"]["M700"]["Pellet_retrieval"] = p_events["D1-Cre"]["Saline"]["M700"]["Pellet_retrieval"][1::2]

p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_drop"] = p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_retrieval"][::2]
p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_retrieval"] = p_events["A2a-Cre"]["Saline"]["C139M1"]["Pellet_retrieval"][1::2]

p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_drop"] = p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_retrieval"][::2]
p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_retrieval"] = p_events["A2a-Cre"]["Saline"]["C139M2"]["Pellet_retrieval"][1::2]

p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_drop"] = p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_retrieval"][::2]
p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_retrieval"] = p_events["A2a-Cre"]["Saline"]["C139M4"]["Pellet_retrieval"][1::2]

p_events["D1-Cre"]["MK801"]["M700"]["Pellet_drop"] = p_events["D1-Cre"]["MK801"]["M700"]["Pellet_retrieval"][::2]
p_events["D1-Cre"]["MK801"]["M700"]["Pellet_retrieval"] = p_events["D1-Cre"]["MK801"]["M700"]["Pellet_retrieval"][1::2]

p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_drop"] = p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_retrieval"][::2]
p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_retrieval"] = p_events["A2a-Cre"]["MK801"]["C139M1"]["Pellet_retrieval"][1::2]

>>>>>>> 094af522984cd562da55f3c05b34fb99dbfcda90

#%%
            
#Here we do a sanity check that the pre-processing looks like its supposed to
strain = "D1-Cre"
treatment = "Saline"
mouse = "C139M2"
event_name = "Pellet_retrieval"

c_uf_events = events[strain][treatment][mouse].to_numpy().squeeze()
c_p_events = p_events[strain][treatment][mouse][event_name]

fig, ax = plt.subplots()
ax.eventplot(c_uf_events)
ax.eventplot(c_p_events, lineoffsets=2, linelengths=0.5, color="r")

#%%

all_events = []
session_duration = []
for strain in p_events:
    for treatment in p_events[strain]:
        for mouse in p_events[strain][treatment]:
            print(strain,treatment,mouse)
            #Getting the number of events
            c_events = p_events[strain][treatment][mouse]
            c_event_quant = pd.DataFrame({event_name: [len(c_events[event_name])] for event_name in 
                                          c_events})

            
            #Getting the duration of the session
            c_data = raw_data[strain][treatment][mouse]
            c_meta = metadata[np.logical_and(metadata["Mouse"] == mouse, metadata["Treatment"] == treatment)]
            c_post_start_idx = np.searchsorted(c_data["Time"], c_meta["Post_start"].values[0])
            
            if not np.isnan(c_meta["Post_end"]).values:
                c_post_end_idx = np.searchsorted(c_data["Time"], c_meta["Post_end"].values[0])
                
                c_post = c_data.iloc[c_post_start_idx: c_post_end_idx]
                
            else:
                c_post = c_data.iloc[c_post_start_idx:]
                
            c_duration = c_post["Time"].iloc[-1] - c_post["Time"].iloc[0]
            
            
            c_event_quant = c_event_quant.assign(Strain=strain, Treatment=treatment, Mouse=mouse, Duration=c_duration)
            all_events.append(c_event_quant)
            
concat_events = pd.concat(all_events)
concat_events = concat_events.assign(Pellet_rate=concat_events["Pellet_retrieval"]/(concat_events["Duration"]/3600),
                                     Poke_rate=concat_events["Pokes"]/(concat_events["Duration"]/3600))

fig, ax = plt.subplots(figsize=(5,8))
sns.boxplot(x="Treatment", y="Pellet_rate", data=concat_events, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Pellet_rate", data=concat_events, s=10, palette=["silver"])
ax.set_xlabel("")
ax.set_ylabel ("Pellet Rate")
ax.set_xticklabels(["Sal", "MK801"])
sns.despine()
#plt.savefig(figure_dir + r"\Figure1_pellet_quant.eps", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,8))
sns.boxplot(x="Treatment", y="Poke_rate", data=concat_events, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Poke_rate", data=concat_events, s=10, palette=["silver"])
ax.set_xlabel("")
ax.set_ylabel ("Poke Rate")
ax.set_xticklabels(["Sal", "MK801"])
sns.despine()
#plt.savefig(figure_dir + r"\Figure1_poke_quant.eps", bbox_inches="tight")

pellets_ttest = ttest_ind(concat_events["Pellet_rate"][concat_events["Treatment"] == "Saline"], concat_events["Pellet_rate"][concat_events["Treatment"] == "MK801"])
pokes_ttest = ttest_ind(concat_events["Poke_rate"][concat_events["Treatment"] == "Saline"], concat_events["Poke_rate"][concat_events["Treatment"] == "MK801"])
            
#%%

#Now we pre-process the photometry data
hp_filter = True

p_base_gcamp = {}
p_post_gcamp = {}
for strain in raw_data:
    p_base_gcamp[strain] = {}
    p_post_gcamp[strain] = {}
    for treatment in raw_data[strain]:
        p_base_gcamp[strain][treatment] = {}
        p_post_gcamp[strain][treatment] = {}
        for mouse in raw_data[strain][treatment]:
            print(strain, treatment, mouse)
            c_meta = metadata[np.logical_and(metadata["Mouse"] == mouse, metadata["Treatment"] == treatment)]
            c_data = raw_data[strain][treatment][mouse]
            
            #If there is a baseline period, get it, otherwirse skip
            if not np.isnan(c_meta["Baseline_start"]).values:
                try:
                    c_base_start_idx = np.searchsorted(c_data["Time"], c_meta["Baseline_start"].values)[0]
                    c_base_end_idx = np.searchsorted(c_data["Time"], c_meta["Baseline_end"].values)[0]
                
                    c_baseline = c_data.iloc[c_base_start_idx:c_base_end_idx,:]
                    
                except:    
                    print(f"Error getting the baseline: {strain}, {treatment}, {mouse}")
                    
                
                c_base_ts = c_baseline["Time"].to_numpy()
                c_base_gcamp = c_baseline["Fluorescence"].to_numpy()
                c_base_isos = c_baseline["Isosbestic"].to_numpy()
                
                c_regr = LinearRegression()
                c_regr.fit(c_base_isos.reshape(-1,1), c_base_gcamp.reshape(-1,1))
                c_base_hat = c_regr.predict(c_base_isos.reshape(-1,1))[:,0]
                
                #c_gcamp_norm = c_base_gcamp - c_base_isos
                c_base_gcamp_norm = (c_base_gcamp - c_base_hat) / c_base_hat[0]
                
                c_sr = 1 / np.diff(c_base_ts).mean()
                
                if hp_filter:
                    c_f_gcamp = butter_filter(c_base_gcamp_norm, "bandpass", (0.005, 6), c_sr)
                   # c_f_gcamp = invian.bp_filter(c_gcamp_norm, 0.005, 6, c_sr)
                else:
                    c_f_gcamp = c_base_gcamp_norm    
                    
                p_base_gcamp[strain][treatment][mouse] = (c_base_ts, c_f_gcamp)
                
            try:
                c_post_start_idx = np.searchsorted(c_data["Time"], c_meta["Post_start"])[0]
                
                if not np.isnan(c_meta["Post_end"]).values:
                    c_post_end_idx = np.searchsorted(c_data["Time"], c_meta["Post_end"])[0]
                    c_post = c_data.iloc[c_post_start_idx: c_post_end_idx]
                    
                else:
                    c_post = c_data.iloc[c_post_start_idx:]

                
                    
            except:
                print(f"Error getting post data: {strain}, {treatment}, {mouse}")
                
            c_post_ts = c_post["Time"].to_numpy()
            c_post_gcamp = c_post["Fluorescence"].to_numpy()
            c_post_isos = c_post["Isosbestic"].to_numpy()
            
            c_post_hat = c_regr.predict(c_post_isos.reshape(-1,1))[:,0]
            
            c_post_gcamp_norm = (c_post_gcamp - c_post_hat)/c_post_hat
            
            
            if hp_filter:
                c_sr = 1 / np.diff(c_post_ts).mean()
                #c_f_gcamp = invian.hp_filter(c_gcamp_norm, 0.001, c_sr)
                c_f_gcamp = butter_filter(c_post_gcamp_norm, "high", 0.001, c_sr)
            else:
                c_f_gcamp = c_post_gcamp_norm
                
            p_post_gcamp[strain][treatment][mouse] = (c_post_ts, c_f_gcamp)


#%%

# Test whether the baseline and post traces make sense

strain = "A2a-Cre"
treatment = "Saline"
mouse = "M521"

if mouse in p_base_gcamp[strain][treatment]:
    t_baseline = p_base_gcamp[strain][treatment][mouse]
    
    fig, ax = plt.subplots()
    ax.plot(t_baseline[0], t_baseline[1])
    
else:
    print(f"{strain} {treatment} {mouse} has no baseline")
            

if mouse in p_post_gcamp[strain][treatment]:
    t_post = p_post_gcamp[strain][treatment][mouse]
    c_uf_events= p_events[strain][treatment][mouse]["Pellet_retrieval"]
    
    fig, ax = plt.subplots()
    ax.plot(t_post[0], t_post[1])
    ax.eventplot(c_uf_events, lineoffsets=np.max(t_post[1])+0.5, linelengths=1)
    
else:
    print(f"{strain} {treatment} {mouse} has no baseline")

#%%

"""
Here we calculate the peri-event histogram for all the events.
"""

event_windows = {
    "Left_poke": (-2,2),
    "Right_poke": (-2,2),
    "Pokes": (-2,2),
    "Rewarded": (-4,4),
    "Unrewarded": (-4,4),
    "Pellet_drop": (-10,15),
    "Pellet_retrieval": (-20,20)
    }
bin_width = 0.1
norm_bins = 50
event_pehs = {}
for strain in p_post_gcamp:
    event_pehs[strain] = {}
    
    for treatment in p_post_gcamp[strain]:
        event_pehs[strain][treatment] = {}
        
        for mouse in p_post_gcamp[strain][treatment]:
            print(f"Calculating PEHs of {strain}, {treatment}, {mouse}")
            event_pehs[strain][treatment][mouse] = {}
            
            c_ts = p_post_gcamp[strain][treatment][mouse][0]
            c_gcamp = p_post_gcamp[strain][treatment][mouse][1]
            c_all_events = p_events[strain][treatment][mouse]
            
            for event_name in event_windows:
                c_event = c_all_events[event_name]
                c_window = event_windows[event_name]
                
                #c_peh = invian.contvar_peh(c_ts, c_gcamp, c_event, min_max=c_window, bin_width=bin_width)
                c_peh = contvar_peh(c_ts, c_gcamp, c_event, min_max=c_window, bin_width=bin_width)
                c_norm_peh = c_peh - (c_peh[:,:norm_bins].mean(axis=1)[:,np.newaxis])
                
                event_pehs[strain][treatment][mouse][event_name] = c_norm_peh

#%%

"""
Here we visualize the peri-event histograms
"""

event_name = "Pellet_retrieval"
strain = "D1-Cre"
treatment = "MK801"
mouse = "F797"

c_window = event_windows[event_name]
if mouse == "all":
    c_data = event_pehs[strain][treatment]
    
    if (event_name == "Pellet_drop") or (event_name == "Pellet_retrieval"):
        y_lim = (-2,1)
    else:
        y_lim = (-0.5,1)
    
    for c_mouse in c_data:
        c_peh = event_pehs[strain][treatment][c_mouse][event_name]
        
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        sns.heatmap(c_peh, ax=ax[0], cbar=False, xticklabels=False)
        ax[1].plot(np.linspace(c_window[0], c_window[1],c_peh.shape[1]), c_peh[:].mean(axis=0))
        #ax[1].set_xlim(c_window[0], c_window[1])
        ax[0].set_title(f"{event_name}, {strain}, {treatment}, {c_mouse}")
        #ax[1].set_ylim(y_lim[0], y_lim[1])
        ax[1].set_xlabel(f"Time from {event_name}")
        ax[1].set_ylabel("Z-Score")
        sns.despine(ax=ax[1])
        
else:
    c_peh = event_pehs[strain][treatment][mouse][event_name]
    
    if (event_name == "Pellet_drop") or (event_name == "Pellet_retrieval"):
        y_lim = (-2,1)
    else:
        y_lim = (-0.5,1)
    
    fig, ax = plt.subplots(2,1, figsize=(8,8))
    sns.heatmap(c_peh, ax=ax[0], cbar=False, xticklabels=False)
    ax[1].plot(np.linspace(c_window[0], c_window[1],c_peh.shape[1]), c_peh.mean(axis=0))
    ax[1].set_xlim(c_window[0], c_window[1])
    ax[0].set_title(f"{event_name}, {strain}, {treatment}, {mouse}")
    #ax[1].set_ylim(y_lim[0], y_lim[1])
    ax[1].set_xlabel(f"Time from {event_name}")
    ax[1].set_ylabel("Z-Score")
    sns.despine(ax=ax[1])

#%%

"""Here we find the averages of all D1-Cre and A2a-Cre responses after saline or MK-801"""

pr_bins = np.arange(event_windows["Pellet_retrieval"][0],event_windows["Pellet_retrieval"][1], 0.1)
all_pr_avgs = []
for strain in event_pehs:
    for treatment in event_pehs[strain]:
        for mouse in event_pehs[strain][treatment]:
            c_pr_peh = event_pehs[strain][treatment][mouse]["Pellet_retrieval"]
            c_pr_peh_avg = c_pr_peh.mean(axis=0)
            
            pr_peh_avg_df = pd.DataFrame({"Value": c_pr_peh_avg, "Time": pr_bins})
            pr_peh_avg_df = pr_peh_avg_df.assign(Strain=strain, Treatment=treatment, Mouse=mouse)
            all_pr_avgs.append(pr_peh_avg_df)
            
concat_pr_avgs = pd.concat(all_pr_avgs)
sal_pr_avgs = concat_pr_avgs[concat_pr_avgs["Treatment"] == "Saline"]
mk801_pr_avgs = concat_pr_avgs[concat_pr_avgs["Treatment"] == "MK801"]

fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Time", y="Value", hue="Treatment", data=concat_pr_avgs, errorbar=None, palette=["darkcyan", "olive"], linewidth=3, legend=False)
sns.lineplot(x="Time", y="Value", hue="Mouse", data=sal_pr_avgs, errorbar=None, palette=["darkcyan"], alpha=0.2, linewidth=3, legend=False)
sns.lineplot(x="Time", y="Value", hue="Mouse", data=mk801_pr_avgs, errorbar=None, palette=["olive"], alpha=0.2, linewidth=3, legend=False)
sns.despine()
ax.set_ylabel(r"$\Delta F/F_0$")
ax.set_xlabel("Time (s)")
ax.axvline(0, c="red", linestyle="--", linewidth=3)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)

#%%

max_responses = []
for strain in event_pehs:
    for treatment in event_pehs[strain]:
        for mouse in event_pehs[strain][treatment]:
            print(strain, treatment, mouse)
            for event in event_pehs[strain][treatment][mouse]:
                c_peh = event_pehs[strain][treatment][mouse][event]
                
                if event == "Pellet_retrieval":
                    c_max_pre = c_peh[:,70:100].max(axis=1)
                    c_max_pre_avg = c_max_pre.mean()
                    
                    c_inhibition = c_peh[:,100:].min(axis=1)
                    c_inhibition_avg = c_inhibition.mean()
                    
                    c_max_resp = pd.DataFrame({
                        "Mouse": mouse,
                        "Treatment": treatment,
                        "Strain": strain,
                        "Pre_max": [c_max_pre_avg],
                        "Post_min": [c_inhibition_avg]
                        })
                    
                    max_responses.append(c_max_resp)
                
concat_responses = pd.concat(max_responses)

fig, ax = plt.subplots(figsize=(5.5,8))
sns.boxplot(x="Treatment", y="Pre_max", data=concat_responses, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Pre_max", data=concat_responses, palette=["silver"], s=10, dodge=False)
ax.set_xlabel("")
ax.set_ylabel(r"Pre-retrieval Max. $\Delta F/F_0$")
ax.set_xticklabels(["Sal", "MK801"])
sns.despine()
plt.savefig(figure_dir + r"\Figure1_pre_retrieval_quant.eps", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5.5,8))
sns.boxplot(x="Treatment", y="Post_min", data=concat_responses, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Post_min", data=concat_responses, palette=["silver"], s=10, dodge=False)
ax.set_xlabel("")
ax.set_ylabel(r"Post-retrieval Min. $\Delta F/F_0$")
ax.set_xticklabels(["Sal", "MK801"])
ax.set_yticks(np.arange(-0.03,0.01, 0.01))
ax.set_ylim(-0.035,0.005)
sns.despine()
plt.savefig(figure_dir + r"\Figure1_post_retrieval_quant.eps", bbox_inches="tight")


pre_resp_ttest = ttest_rel(concat_responses["Pre_max"][concat_responses["Treatment"] == "Saline"], 
                           concat_responses["Pre_max"][concat_responses["Treatment"] == "MK801"])
post_resp_ttest = ttest_rel(concat_responses["Post_min"][concat_responses["Treatment"] == "Saline"], 
                            concat_responses["Max"][concat_responses["Treatment"] == "MK801"])