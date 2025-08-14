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

#figure_dir = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\gcamp_bandit_corrs\Figure_panels"
figure_dir = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025\Figure_panels"
#fileloc = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\gcamp_bandit_corrs\photometry recordings"
fileloc = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025_data\gcamp_bandit_corrs\photometry recordings"
fileloc2 = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025_data\Mason Data"


plt.rcParams.update({'font.size': 24, 'figure.autolayout': True, 'lines.linewidth': 2})

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
Notes F783 MK801 has a very strong inhibition. Check data
You need to normalize to baseline.
"""

#%%


"""
Loading all the data
"""

raw_data = {
    "D1-Cre": {
        "Saline": {
            "F488": pd.read_csv(fileloc + r"\Good_data\Raw_files\saline_D1_F488_bandit_031825.csv"),
            "M690": pd.read_csv(fileloc + r"\Good_data\Raw_files\saline_D1_M690_bandit_040325.csv"),
            "M700": pd.read_csv(fileloc + r"\Good_data\Raw_files\saline_D1_M700_bandit_040325.csv"),
            "M780": pd.read_csv(fileloc + r"\saline_D1_M780_bandit_051325.csv"),
            "F797": pd.read_csv(fileloc + r"\saline_D1_F797_bandit_051325.csv"),
            },
        "MK801": {
            "F488": pd.read_csv(fileloc + r"\Good_data\Raw_files\mk801_D1_F488_bandit_040425.csv"),
            "M690": pd.read_csv(fileloc + r"\Good_data\Raw_files\mk801_D1_M690_bandit_042925.csv"),
            "M700": pd.read_csv(fileloc + r"\Good_data\Raw_files\mk801_D1_M700_bandit_042525.csv"),
            "M780": pd.read_csv(fileloc2 + r"\mk801_D1_M780_bandit_060925.csv"),
            "F797": pd.read_csv(fileloc2 + r"\mk801_D1_F797_bandit_060925.csv")
            }
        },
    "A2a-Cre": {
        "Saline": {
            "M521": pd.read_csv(fileloc + r"\Good_data\Raw_files\saline_A2a_M521_bandit_011025.csv"),
            "C139M1": pd.read_csv(fileloc + r"\Good_data\Raw_files\saline_A2a_C139M1_bandit_041425.csv"),
            "C139M2": pd.read_csv(fileloc + r"\saline_C139M2_A2a_bandit_041425.csv"),
            "C139M4": pd.read_csv(fileloc + r"\Good_data\Raw_files\saline_A2a_C139M4_bandit_040725.csv")
            },
        "MK801": {
            "M521": pd.read_csv(fileloc + r"\Good_data\Raw_files\mk801_A2a_M521_bandit_030625.csv"),
            "C139M1": pd.read_csv(fileloc + r"\Good_data\Raw_files\mk801_A2a_C139M1_bandit_050225.csv"),
            "C139M4": pd.read_csv(fileloc + r"\Good_data\Raw_files\mk801_A2a_C139M4_bandit_042425.csv"),
            "C139M2": pd.read_csv(fileloc + r"\mk801_A2a_C139M2_bandit_042525.csv")
            }
        }
    }

events = {
    "D1-Cre": {
        "Saline": {
            "F488": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_D1_F488_bandit_031825.csv", index_col=0),
            "M690": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_D1_M690_bandit_040325.csv", index_col=0),
            "M700": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_D1_M700_bandit_040325.csv", index_col=0),
            "M780": pd.read_csv(fileloc + r"\events_saline_D1_M780_bandit_051325.csv", index_col=0),
            "F797": pd.read_csv(fileloc + r"\events_saline_D1_F797_bandit_051325.csv", index_col=0)
            },
        "MK801": {
            "F488": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_mk801_D1_F488_bandit_040425.csv", index_col=0),
            "M690": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_mk801_D1_M690_bandit_042925.csv", index_col=0),
            "M700": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_mk801_D1_M700_bandit_042525.csv", index_col=0),
            "M780": pd.read_csv(fileloc2 + r"\events_mk801_D1_M780_bandit_060925.csv", index_col=0),
            "F797": pd.read_csv(fileloc2 + r"\events_mk801_D1_F797_bandit_060925.csv", index_col=0)
            }
        },
    "A2a-Cre": {
        "Saline": {
            "M521": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_A2a_M521_bandit_011025.csv", index_col=0),
            "C139M1": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_A2a_C139M1_bandit_041425.csv", index_col=0),
            "C139M2": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_A2a_C139M2_bandit_041425.csv", index_col=0),
            "C139M4": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_saline_A2a_C139M4_bandit_040725.csv", index_col=0)
            },
        "MK801": {
            "M521": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_mk801_A2a_M521_bandit_030625.csv", index_col=0),
            "C139M4": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_mk801_A2a_C139M4_bandit_042425.csv", index_col=0),
            "C139M1": pd.read_csv(fileloc + r"\Good_data\Event_timestamps\events_mk801_A2a_C139M1_bandit_050225.csv", index_col=0),
            "C139M2": pd.read_csv(fileloc + r"\events_mk801_A2a_C139M2_bandit_042525.csv", index_col=0)
            }
        }
    }

metadata = {
    "D1-Cre": {
        "Saline": {
            "F488": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_D1_F488_bandit_031825.csv", index_col=0),
            "M690": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_D1_M690_bandit_040325.csv", index_col=0),
            "M700": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_D1_M700_bandit_040325.csv", index_col=0),
            "M780": pd.read_csv(fileloc + r"\meta_saline_D1_M780_bandit_051325.csv", index_col=0),
            "F797": pd.read_csv(fileloc + r"\meta_saline_D1_F797_bandit_051325.csv", index_col=0)
            },
        "MK801": {
            "F488": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_mk801_D1_F488_bandit_040425.csv", index_col=0),
            "M690": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_mk801_D1_M690_bandit_042925.csv", index_col=0),
            "M700": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_mk801_D1_M700_bandit_042525.csv", index_col=0),
            "M780": pd.read_csv(fileloc2 + r"\meta_mk801_D1_M780_bandit_060925.csv", index_col=0),
            "F797": pd.read_csv(fileloc2 + r"\meta_mk801_D1_F797_bandit_060925.csv", index_col=0)
            }
        },
    "A2a-Cre": {
        "Saline": {
            "M521": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_A2a_M521_bandit_011025.csv", index_col=0),
            "C139M1": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_A2a_C139M1_bandit_041425.csv", index_col=0),
            "C139M2": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_A2a_C139M2_bandit_041425.csv", index_col=0),
            "C139M4": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_saline_A2a_C139M4_bandit_040725.csv", index_col=0)
            },
        "MK801": {
            "M521": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_mk801_A2a_M521_bandit_030625.csv", index_col=0),
            "C139M4": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_mk801_A2a_C139M4_bandit_042425.csv", index_col=0),
            "C139M1": pd.read_csv(fileloc + r"\Good_data\Metadata\meta_mk801_A2a_C139M1_bandit_050225.csv", index_col=0),
            "C139M2": pd.read_csv(fileloc + r"\meta_mk801_A2a_C139M2_bandit_042525.csv", index_col=0)
            }
        }
    }

#%%

#Check the all the raw data looks okay
strain = "D1-Cre"
treatment = "MK801"
mouse = "F797"

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


#%%
            
#Here we do a sanity check that the pre-processing looks like its supposed to
strain = "D1-Cre"
treatment = "MK801"
mouse = "F797"
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
            c_meta = metadata[strain][treatment][mouse]
            c_post_start_idx = np.searchsorted(c_data["Time"], c_meta["Post_start"][0])
            
            if not np.isnan(c_meta["Post_end"])[0]:
                c_post_end_idx = np.searchsorted(c_data["Time"], c_meta["Post_end"][0])
                
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
ax.set_ylabel ("Pellet/hour")
sns.despine()
#plt.savefig(figure_dir + r"\Figure1_pellet_quant.eps", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,8))
sns.boxplot(x="Treatment", y="Poke_rate", data=concat_events, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Poke_rate", data=concat_events, s=10, palette=["silver"])
ax.set_xlabel("")
ax.set_ylabel ("Pokes/hour")
sns.despine()
#plt.savefig(figure_dir + r"\Figure1_poke_quant.eps", bbox_inches="tight")

pellets_ttest = ttest_ind(concat_events["Pellet_rate"][concat_events["Treatment"] == "Saline"], concat_events["Pellet_rate"][concat_events["Treatment"] == "MK801"])
pokes_ttest = ttest_ind(concat_events["Poke_rate"][concat_events["Treatment"] == "Saline"], concat_events["Poke_rate"][concat_events["Treatment"] == "MK801"])

#%%

#Now we pre-process the photometry data

z_score = False
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
            c_meta = metadata[strain][treatment][mouse]
            c_data = raw_data[strain][treatment][mouse]
            
            #If there is a baseline period, get it, otherwirse skip
            if not np.isnan(c_meta["Baseline_start"])[0]:
                
                try:
                    c_base_start_idx = np.searchsorted(c_data["Time"], c_meta["Baseline_start"][0])
                    c_base_end_idx = np.searchsorted(c_data["Time"], c_meta["Baseline_end"][0])
                
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
                
                if z_score:
                    c_z_gcamp = (c_f_gcamp - c_f_gcamp.mean()) / c_f_gcamp.std()
                    
                else:
                    c_z_gcamp = c_f_gcamp
                    
                p_base_gcamp[strain][treatment][mouse] = (c_base_ts, c_z_gcamp)
                
            try:
                c_post_start_idx = np.searchsorted(c_data["Time"], c_meta["Post_start"][0])
                
                if not np.isnan(c_meta["Post_end"])[0]:
                    c_post_end_idx = np.searchsorted(c_data["Time"], c_meta["Post_end"][0])
                    
                    c_post = c_data.iloc[c_post_start_idx: c_post_end_idx]
                    
                else:
                    c_post = c_data.iloc[c_post_start_idx:]
                
                    
            except:
                print(f"Error getting post data: {strain}, {treatment}, {mouse}")
                
            c_post_ts = c_post["Time"].to_numpy()
            c_post_gcamp = c_post["Fluorescence"].to_numpy()
            c_post_isos = c_post["Isosbestic"].to_numpy()
            
            c_post_hat = c_regr.predict(c_post_isos.reshape(-1,1))[:,0]
            
            c_post_gcamp_norm = (c_post_gcamp - c_post_hat)/c_post_hat[0]
            
            
            if hp_filter:
                c_sr = 1 / np.diff(c_post_ts).mean()
                #c_f_gcamp = invian.hp_filter(c_gcamp_norm, 0.001, c_sr)
                c_f_gcamp = butter_filter(c_post_gcamp_norm, "high", 0.001, c_sr)
            else:
                c_f_gcamp = c_post_gcamp_norm
            
            if z_score:
                c_z_gcamp = (c_f_gcamp - c_f_gcamp.mean()) / c_f_gcamp.std()
                
            else:
                c_z_gcamp = c_f_gcamp
                
            p_post_gcamp[strain][treatment][mouse] = (c_post_ts, c_z_gcamp)


#%%

# Test whether the baseline and post traces make sense

strain = "D1-Cre"
treatment = "MK801"
mouse = "F797"

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

v_lim = (-2,4)


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
        sns.heatmap(c_peh, ax=ax[0], cbar=False, xticklabels=False, vmin=v_lim[0], vmax=v_lim[1])
        ax[1].plot(np.linspace(c_window[0], c_window[1],c_peh.shape[1]), c_peh[:].mean(axis=0))
        ax[1].set_xlim(c_window[0], c_window[1])
        ax[0].set_title(f"{event_name}, {strain}, {treatment}, {c_mouse}")
        ax[1].set_ylim(y_lim[0], y_lim[1])
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
    sns.heatmap(c_peh, ax=ax[0], cbar=False, xticklabels=False, vmin=v_lim[0], vmax=v_lim[1])
    ax[1].plot(np.linspace(c_window[0], c_window[1],c_peh.shape[1]), c_peh.mean(axis=0))
    ax[1].set_xlim(c_window[0], c_window[1])
    ax[0].set_title(f"{event_name}, {strain}, {treatment}, {mouse}")
    ax[1].set_ylim(y_lim[0], y_lim[1])
    ax[1].set_xlabel(f"Time from {event_name}")
    ax[1].set_ylabel("Z-Score")
    sns.despine(ax=ax[1])

#%%

"""Here we find the averages of all D1-Cre and A2a-Cre responses after saline or MK-801"""


d1_saline_pr = {mouse: event_pehs["D1-Cre"]["Saline"][mouse]["Pellet_retrieval"] for mouse in 
                event_pehs["D1-Cre"]["Saline"]}
            
d1_saline_pr_avg = pd.DataFrame({mouse: d1_saline_pr[mouse].mean(axis=0) for mouse in d1_saline_pr})
d1_saline_pr_avg = d1_saline_pr_avg.assign(Strain="D1-Cre", Treatment="Saline", 
                                           Time=np.arange(event_windows["Pellet_retrieval"][0],event_windows["Pellet_retrieval"][1], 0.1))
m_d1_saline_pr_avg = pd.melt(d1_saline_pr_avg, id_vars=["Strain", "Treatment", "Time"])


#Now the D2s
a2a_saline_pr = {mouse: event_pehs["A2a-Cre"]["Saline"][mouse]["Pellet_retrieval"] for mouse in 
                event_pehs["A2a-Cre"]["Saline"]}
            
a2a_saline_pr_avg = pd.DataFrame({mouse: a2a_saline_pr[mouse].mean(axis=0) for mouse in a2a_saline_pr})
a2a_saline_pr_avg = a2a_saline_pr_avg.assign(Strain="a2a-Cre", Treatment="Saline", 
                                           Time=np.arange(event_windows["Pellet_retrieval"][0],event_windows["Pellet_retrieval"][1], 0.1))
m_a2a_saline_pr_avg = pd.melt(a2a_saline_pr_avg, id_vars=["Strain", "Treatment", "Time"])


m_both_saline_pr = pd.concat([m_d1_saline_pr_avg, m_a2a_saline_pr_avg])


#MK801
d1_mk801_pr = {mouse: event_pehs["D1-Cre"]["MK801"][mouse]["Pellet_retrieval"] for mouse in 
                event_pehs["D1-Cre"]["MK801"]}
            
d1_mk801_pr_avg = pd.DataFrame({mouse: d1_mk801_pr[mouse].mean(axis=0) for mouse in d1_mk801_pr})
d1_mk801_pr_avg = d1_mk801_pr_avg.assign(Strain="D1-Cre", Treatment="MK801", 
                                           Time=np.arange(event_windows["Pellet_retrieval"][0],event_windows["Pellet_retrieval"][1], 0.1))
m_d1_mk801_pr_avg = pd.melt(d1_mk801_pr_avg, id_vars=["Strain", "Treatment", "Time"])

#Both 
m_d1_both_pr_avg = pd.concat([m_both_saline_pr, m_d1_mk801_pr_avg])

#Now the D2s
a2a_mk801_pr = {mouse: event_pehs["A2a-Cre"]["MK801"][mouse]["Pellet_retrieval"] for mouse in 
                event_pehs["A2a-Cre"]["MK801"]}
            
a2a_mk801_pr_avg = pd.DataFrame({mouse: a2a_mk801_pr[mouse].mean(axis=0) for mouse in a2a_mk801_pr})
a2a_mk801_pr_avg = a2a_mk801_pr_avg.assign(Strain="a2a-Cre", Treatment="MK801", 
                                           Time=np.arange(event_windows["Pellet_retrieval"][0],event_windows["Pellet_retrieval"][1], 0.1))
m_a2a_mk801_pr_avg = pd.melt(a2a_mk801_pr_avg, id_vars=["Strain", "Treatment", "Time"])

#Both A2a
m_a2a_both_pr_avg = pd.concat([m_a2a_saline_pr_avg, m_a2a_mk801_pr_avg])


# Pool both MK801 groups
m_both_mk801_pr = pd.concat([m_d1_mk801_pr_avg, m_a2a_mk801_pr_avg])


#Pool saline and MK801 for plotting
m_all_pr = pd.concat([m_both_saline_pr, m_both_mk801_pr])

m_sal_pr = m_all_pr[m_all_pr["Treatment"] == "Saline"]
m_mk801_pr = m_all_pr[m_all_pr["Treatment"] == "MK801"]

#
fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Time", y="value", hue="Treatment", data=m_all_pr, errorbar=None, palette=["darkcyan", "olive"], linewidth=3, legend=False)
sns.lineplot(x="Time", y="value", hue="variable", data=m_sal_pr, errorbar=None, palette=["darkcyan"], alpha=0.2, linewidth=3, legend=False)
sns.lineplot(x="Time", y="value", hue="variable", data=m_mk801_pr, errorbar=None, palette=["olive"], alpha=0.2, linewidth=3, legend=False)
sns.despine()
ax.set_ylabel(r"$\Delta F/F_0$")
ax.set_xlabel("Time from pellet retrieval (s)")
#ax.set_ylim(-1.5,1.5)
ax.axvline(0, c="red", linestyle="--", linewidth=3)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
#plt.savefig(figure_dir + r"\Figure1_pellet_retrieval_sal_mk801.eps", bbox_inches="tight")


#
fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Time", y="value", hue="Treatment", data=m_d1_both_pr_avg, errorbar="se", linewidth=3, legend=False, palette=["darkcyan", "olive"])
sns.despine()
ax.set_ylabel("Z-Score")
ax.set_xlabel("Time from pellet retrieval (s)")
#ax.set_ylim(-2.5,1.7)
ax.axvline(0, c="red", linestyle="--", linewidth=3)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)


#
fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Time", y="value", hue="Treatment", data=m_a2a_both_pr_avg, errorbar="se", linewidth=3, legend=False, palette=["darkcyan", "olive"])
sns.despine()
ax.set_ylabel("Z-Score")
ax.set_xlabel("Time from pellet retrieval (s)")
#ax.set_ylim(-2.5,1.7)
ax.axvline(0, c="red", linestyle="--", linewidth=3)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)

#%%

"""
Here we quantify the maximum response for each mouse
"""

max_responses = {
    "Left_poke": [],
    "Right_poke": [],
    "Pokes": [],
    "Rewarded": [],
    "Unrewarded": [],
    "Pellet_drop": [],
    "Pellet_retrieval_pre": [],
    "Pellet_retrieval_post": []
    }

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
                    
                    max_responses["Pellet_retrieval_pre"].append((mouse, treatment, strain, c_max_pre_avg, "Pellet_retrieval_max"))
                    max_responses["Pellet_retrieval_post"].append((mouse, treatment, strain, c_inhibition_avg, "Pellet_retrieval_inh"))
                    
                elif event == "Pellet_drop":
                    c_max = c_peh[:,:10].max(axis=1)
                    c_max_avg = c_max.mean()
                    
                    max_responses[event].append((mouse, treatment, strain, c_max_avg,event))
                    
                elif ((event == "Left_poke") or (event == "Right_poke") or (event == "Pokes")):
                    c_max = c_peh[:,10:30].max(axis=1)
                    c_max_avg = c_max.mean()
                    max_responses[event].append((mouse, treatment, strain, c_max_avg, event))
                    
                elif ((event == "Rewarded") or (event == "Unrewarded")):
                    c_max = c_peh[:, 40:60].max(axis=1)
                    c_max_avg = c_max.mean()
                    max_responses[event].append((mouse, treatment, strain, c_max_avg, event))
                
                
max_responses_df = {event: pd.DataFrame(max_responses[event], columns=["Mouse", "Treatment", "Strain" , "Max", "Event"]) for event in max_responses}

all_responses_df = pd.concat(max_responses_df.values())

# d1_responses_df = all_responses_df[all_responses_df["Strain"] == "D1-Cre"]
# a2a_responses_df = all_responses_df[all_responses_df["Strain"] == "A2a-Cre"]

pre_resp = all_responses_df[all_responses_df["Event"] == "Pellet_retrieval_max"]
post_resp = all_responses_df[all_responses_df["Event"] == "Pellet_retrieval_inh"]

fig, ax = plt.subplots(figsize=(5,8))
sns.boxplot(x="Treatment", y="Max", data=pre_resp, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Max", data=pre_resp, palette=["silver"], s=10, dodge=False)
ax.set_xlabel("")
ax.set_ylabel(r"Pre-retrieval Max. $\Delta F/F_0$")
ax.set_xticklabels(["Sal", "MK801"])
ax.set_yticks([0,0.01, 0.02])
ax.set_ylim(-0.002, 0.022)
sns.despine()
#plt.savefig(figure_dir + r"\Figure1_pre_retrieval_quant.eps")

fig, ax = plt.subplots(figsize=(5,8))
sns.boxplot(x="Treatment", y="Max", data=post_resp, palette=["darkcyan", "olive"])
sns.swarmplot(x="Treatment", y="Max", data=post_resp, palette=["silver"], s=10, dodge=False)
ax.set_xlabel("")
ax.set_ylabel(r"Post-retrieval Min. $\Delta F/F_0$")
ax.set_xticklabels(["Sal", "MK801"])
ax.set_yticks(np.arange(-0.04,0.01, 0.01))
ax.set_ylim(-0.045,0.005)
sns.despine()
#plt.savefig(figure_dir + r"\Figure1_post_retrieval_quant.eps")


pre_resp_ttest = ttest_rel(pre_resp["Max"][pre_resp["Treatment"] == "Saline"], pre_resp["Max"][pre_resp["Treatment"] == "MK801"])
post_resp_ttest = ttest_rel(pre_resp["Max"][post_resp["Treatment"] == "Saline"], post_resp["Max"][post_resp["Treatment"] == "MK801"])
