# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:59:33 2024

@author: Alex
"""

import pickle
import invian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import copy
from scipy.signal import find_peaks, periodogram, spectrogram, filtfilt, butter, find_peaks_cwt, correlate, peak_prominences, welch


figure_dir = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025\Figure_panels"
file_loc = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025_data\fiber_cannula_pharmacology\Data"
file_loc = r"C:\Users\Alex\Desktop\Legaria_etal_2025\data\temporary_data\Figure_2"
plt.rcParams.update({'font.size': 32, 'figure.autolayout': True})

#%%

def butter_highpass(low, sr):
    b,a = butter(3, low, btype='high', fs=sr)
    return b, a

def butter_lowpass(high, sr):
    b,a = butter(3, high, btype = 'low', fs=sr)
    return b, a

def hp_filter(signal, low, sr):
    b, a = butter_highpass(low, sr)
    y = filtfilt(b, a, signal, padtype = "even")
    return y
def lp_filter(signal, high, sr):
    b,a = butter_lowpass(high, sr)
    y = filtfilt(b, a, signal, padtype = 'even')
    return y

def butter_bandpass(low, high, sr):
    b,a = butter(3, (low,high), btype="bandpass", fs=sr)
    return b,a

def bp_filter(signal, low, high, sr):
    b,a = butter_bandpass(low, high, sr)
    y = filtfilt(b, a, signal, padtype="even")
    return y

def find_adjust_peaks(signal, widths, min_snr, sr, threshold=0, prominence=0):
    peaks = find_peaks_cwt(signal, widths, min_snr=min_snr)
    
    c_threshold = np.std(signal) * threshold
    
    adjusted_peaks = []
    for peak in peaks:
        start_index = max(0, peak - sr)
        end_index = min(len(signal) - 1, peak + sr)
        local_max_index = np.argmax(signal[start_index:end_index + 1]) + start_index
        
        
        if signal[local_max_index] > c_threshold:
            adjusted_peaks.append(local_max_index)
    
    peak_proms = peak_prominences(signal, adjusted_peaks)[0]
    
    f_peaks = [adjusted_peaks[i] for i in range(len(peak_proms)) if (peak_proms[i] > prominence)]
    
    f_peaks = np.array(list(set(f_peaks)))  # Remove duplicates
    
    return f_peaks

#Temp functions
def butter_filter(signal, filt_type, freqs, sr, order=3):
    b,a = butter(order, freqs, btype=filt_type, fs=sr)
    y=filtfilt(b, a, signal, padtype="even")
    
    return y


#%%

raw_data_mk801 = {
    "saline": {
        "C115a": pd.read_csv(file_loc + r"\c115a_icv_saline_050124.csv"),
        "C115c": pd.read_csv(file_loc + r"\c115c_icv_saline_061724.csv"),
        "FC1M1": pd.read_csv(file_loc + r"\fc1m1_icv_saline_070324.csv"),
        "FC1M2": pd.read_csv(file_loc + r"\fc1m2_icv_saline_070924.csv"),
        "FC2F1": pd.read_csv(file_loc + r"\fc2f1_icv_saline_081324.csv"),
        "GAC1M1": pd.read_csv(file_loc + r"\GAC1M1_icv_saline_081324.csv"),
        "GAC1M2": pd.read_csv(file_loc + r"\GAC1M2_icv_saline_081424.csv"),
        "FC3M2": pd.read_csv(file_loc + r"\fc3m2_icv_saline_082124.csv"),
        "FC3M3": pd.read_csv(file_loc + r"\fc3m3_icv_saline_082924.csv"),
        "FC3M5": pd.read_csv(file_loc + r"\fc3m5_icv_saline_091724.csv")
        },
    "0.1mgml": {
        "FC3M5": pd.read_csv(file_loc + r"\fc3m5_icv_mk801_0.1mgml_081924.csv"),
        "FC1M3": pd.read_csv(file_loc + r"\fc1m3_icv_mk801_0.1mgml_081924.csv"),
        "FC1M2": pd.read_csv(file_loc + r"\fc1m2_icv_mk801_0.1mgml_081924.csv"),
        "FC1M1": pd.read_csv(file_loc + r"\fc1m1_icv_mk801_0.1mgml_090224.csv"),
        "FC3M2": pd.read_csv(file_loc + r"\fc3m2_icv_mk801_0.1mgml_090224.csv"),
        "FC3M3": pd.read_csv(file_loc + r"\fc3m3_icv_mk801_0.1mgml_091724.csv")
        },
    "1mgml": {
        "C115c": pd.read_csv(file_loc + r"\c115c_icv_mk801_1mgml_070224.csv"),
        "C115d": pd.read_csv(file_loc + r"\c115d_icv_MK801_1mgkg_072624.csv"),
        "C115a": pd.read_csv(file_loc + r"\c115a_icv_mk801_1mgkg_072924.csv"),
        "FC1M1": pd.read_csv(file_loc + r"\fc1m1_icv_mk801_1mgkg_0730242.csv"),
        "FC1M3": pd.read_csv(file_loc + r"\fc1m3_icv_mk801_1mgkg_073024.csv"),
        "FC1M2": pd.read_csv(file_loc + r"\fc1m2_icv_mk801_1mgml_090324.csv")
        },
    "2mgml": {
        "C115a": pd.read_csv(file_loc + r"\c115a_icv_mk801_2mgml_061824.csv"),
        "C115d": pd.read_csv(file_loc + r"\c115d_icv_mk801_2mgml_070224.csv"),
        "FC1M1": pd.read_csv(file_loc + r"\fc1m1_icv_mk801_2mgml_071724.csv"),
        "FC1M2": pd.read_csv(file_loc + r"\fc1m2_icv_mk801_2mgml_072424.csv"),
        "GAC1M1": pd.read_csv(file_loc + r"\GAC1M1_icv_mk801_2mgml_082024.csv"),
        "FC1M3": pd.read_csv(file_loc + r"\fc1m3_icv_mk801_2mgml_082624.csv")
        },
    "4mgml": {
        "C115c": pd.read_csv(file_loc + r"\c115c_icv_mk801_042324.csv"),
        "FC1M1": pd.read_csv(file_loc + r"\fc1m1_icv_mk801_4mgml_072524.csv"),
        "C115a": pd.read_csv(file_loc + r"\c115a_icv_mk801_4mgml_080524.csv"),
        "FC1M2": pd.read_csv(file_loc + r"\FC1M2_icv_mk801_4mgml_081224.csv"),
        "FC1M3": pd.read_csv(file_loc + r"\FC1M3_icv_mk801_4mgml_0813242.csv"),
        "FC3M2": pd.read_csv(file_loc + r"\fc3m2_icv_mk801_4mgml_0918242.csv")
        }
    }

event_ts_mk801 = {
    "saline": {
        "C115a": [1107, 2077],
        "C115c": [2300, 9115],
        "FC1M1": [1950, 6050],
        "FC1M2": [1580, 6400],
        "FC2F1": [1850, 5000],
        "GAC1M1": [2450, 5950],
        "GAC1M2": [2100, 5750],
        "FC3M2": [1700, 5800],
        "FC3M3": [1670, 4300],
        "FC3M5": [1900, 8200]
        },
    "0.1mgml": {
        "FC3M5": [2000, 5600],
        "FC1M2": [2200, 5500],
        "FC1M3": [2500, 5800],
        "FC3M2": [2200, 6500],
        "FC1M1": [2100, 6500],
        "FC3M3": [3400, 7000]
        },
    "1mgml": {
        "C115c": [2350, 7400],
        "C115d": [2250, 6750],
        "C115a": [2000, 6200],
        "FC1M1": [2450, 4800],
        "FC1M3": [1950, 10500],
        "FC1M2": [2250, 5600]
        },
    "2mgml": {
        "C115a": [2200, 6710],
        "C115d": [1950, 5250],
        "FC1M1": [2000, 4500],
        "FC1M2": [1750, 6300],
        "GAC1M1": [2100, 5400],
        "FC1M3": [2200, 11200]
        },
    "4mgml": {
        "C115c": [1057, 2080],
        "FC1M1": [4000, 8800],
        "C115a": [2250, 6900],
        "FC1M2": [2450, 5100],
        "FC1M3": [2100, 5100],
        "FC3M2": [2900, 7800]
        }
    }

with open(file_loc + r'\icv_photometry_peaks_mk801_091824.p', 'rb') as fp:
    peaks_mk801_2 = pickle.load(fp)

#%%

"""
Plotting the raw data
"""

group = "4mgml"
mouse = "FC3M2"

t_ts = (raw_data_mk801[group][mouse]["Time"] - raw_data_mk801[group][mouse]["Time"].iloc[0])

fig, ax = plt.subplots(2,1)
ax[0].plot(t_ts, raw_data_mk801[group][mouse]["Fluorescence"])
ax[1].plot(t_ts, raw_data_mk801[group][mouse]["Isosbestic"])
        
#%%

"""
Here we are using a isosbestic-dependent df/f ((signal - isosbesic_pred) / isosbestic_pred)).
We are also normalizing all to baseline.
"""

p_data = {}
ts = {}
for condition in raw_data_mk801:
    p_data[condition] = {}
    ts[condition] = {}
    
    c_condition = raw_data_mk801[condition]
    
    for mouse in c_condition:
        print(condition, mouse)
        p_data[condition][mouse] = {}
        ts[condition][mouse] = {}
        
        c_data = c_condition[mouse]
        c_data["norm_timestamp"] = c_data["Time"] - c_data["Time"].iloc[0]
        
        c_ts = c_data["norm_timestamp"].iloc[150:].to_numpy()
        c_isos = c_data["Isosbestic"].iloc[150:].to_numpy()
        c_gcamp = c_data["Fluorescence"].iloc[150:].to_numpy()
        
        c_events = event_ts_mk801[condition][mouse]
        c_baseline_end = np.searchsorted(c_ts, c_events[0])
        c_post_start = np.searchsorted(c_ts, c_events[1]+600)
        c_post_end = np.searchsorted(c_ts, c_events[1]+4200)
        
       
        #We slice and normalize baseline
        c_baseline_ts = c_ts[:c_baseline_end]
        c_baseline_isos = c_isos[:c_baseline_end]
        c_baseline_gcamp = c_gcamp[:c_baseline_end]
        
        regr = LinearRegression()
        regr.fit(c_baseline_isos.reshape(-1,1), c_baseline_gcamp.reshape(-1,1))
        c_baseline_hat = regr.predict(c_baseline_isos.reshape(-1,1))
        
        c_norm_baseline = (c_baseline_gcamp - c_baseline_hat[:,0])/c_baseline_hat[:,0]
        c_norm_baseline_2 = butter_filter(c_norm_baseline, "bandpass", (0.005,10), 40)
        #c_norm_baseline_2 = bp_filter(c_norm_baseline, 0.005, 10, 40)
        c_norm_baseline_3 = c_norm_baseline_2
        #c_norm_baseline_3 = (c_norm_baseline_2 - c_norm_baseline_2.mean()) / c_norm_baseline_2.std()
        
        #
        c_post_ts = c_ts[c_post_start:c_post_end]
        c_post_isos = c_isos[c_post_start:c_post_end]
        c_post_gcamp = c_gcamp[c_post_start:c_post_end]
        
        c_post_hat = regr.predict(c_post_isos.reshape(-1,1))
        
        c_norm_post = (c_post_gcamp - c_post_hat[:,0]) / c_post_hat[:,0]
        c_norm_post_2 = butter_filter(c_norm_post, "bandpass", (0.005,10), 40)
        #c_norm_post_2 = bp_filter(c_norm_post, 0.005, 10, 40)
        c_norm_post_3 = c_norm_post_2
        #c_norm_post_3 = (c_norm_post_2 - c_norm_post_2.mean()) / c_norm_post_2.std()
        
        p_data[condition][mouse]["Baseline"] = c_norm_baseline_3
        p_data[condition][mouse]["Post"] = c_norm_post_3
        
        ts[condition][mouse]["Baseline"] = c_baseline_ts
        ts[condition][mouse]["Post"] = c_post_ts

#%%

"""
Plotting processed data
"""

group = "2mgml"
mouse = "FC1M3"

fig, ax = plt.subplots(2,1)
ax[0].plot(ts[group][mouse]["Baseline"], p_data[group][mouse]["Baseline"])
ax[1].plot(ts[group][mouse]["Post"], p_data[group][mouse]["Post"])
ax[0].set_ylim(-0.01,0.02)
ax[1].set_ylim(-0.01,0.02)
            
#%%

"""
This finds the peaks with the new data pre-processing
"""

widths = np.arange(1,120)
min_snr = 6
prominence = 0
thresh = 1

peaks_mk801 = {}
peak_ts = {}
for condition in p_data:
    peaks_mk801[condition] = {}
    peak_ts[condition] = {}
    c_condition = p_data[condition]
    
    for mouse in c_condition:
        print(condition, mouse)
        peaks_mk801[condition][mouse] = {}
        peak_ts[condition][mouse] = {}
        c_baseline_ts = ts[condition][mouse]["Baseline"]
        c_baseline_vals = p_data[condition][mouse]["Baseline"]
        
        c_post_ts = ts[condition][mouse]["Post"]
        c_post_vals = p_data[condition][mouse]["Post"]

        c_baseline_peaks_mk801 = find_adjust_peaks(c_baseline_vals, widths, min_snr, 40, threshold=thresh)
        c_post_peaks_mk801 = find_adjust_peaks(c_post_vals, widths, min_snr, 40, prominence=prominence)
        
        peaks_mk801[condition][mouse]["Baseline"] = c_baseline_peaks_mk801
        peaks_mk801[condition][mouse]["Post"] = c_post_peaks_mk801
        
        
        c_baseline_ts = ts[condition][mouse]["Baseline"]
        peak_ts[condition][mouse]["Baseline"] = c_baseline_ts[c_baseline_peaks_mk801]
        
        c_post_ts = ts[condition][mouse]["Post"]
        if len(c_post_peaks_mk801) > 0:
            peak_ts[condition][mouse]["Post"] = c_post_ts[c_post_peaks_mk801]
        else:
            peak_ts[condition][mouse]["Post"] = []

#%%

#Find peaks of new recordings and update the pickle file

#Add new recording
condition = "4mgml"
mouse = "FC3M2"

if mouse not in peaks_mk801[condition].keys():
    print(f"Processing mouse {mouse}")
    peaks_mk801[condition][mouse] = {}
    #peaks_mk801_ts[condition][mouse] = {}`
    
    c_baseline_ts = ts[condition][mouse]["Baseline"]
    c_baseline_vals = p_data[condition][mouse]["Baseline"]
    
    c_post_ts = ts[condition][mouse]["Post"]
    c_post_vals = p_data[condition][mouse]["Post"]

    c_baseline_peaks = find_adjust_peaks(c_baseline_vals, widths, min_snr, 40, prominence=prominence)
    c_post_peaks = find_adjust_peaks(c_post_vals, widths, min_snr, 40, prominence=prominence)
    
    peaks_mk801[condition][mouse]["Baseline"] = c_baseline_peaks
    peaks_mk801[condition][mouse]["Post"] = c_post_peaks
    
    c_baseline_ts = ts[condition][mouse]["Baseline"]
    #peaks_mk801_ts[condition][mouse]["Baseline"] = c_baseline_ts[c_baseline_peaks]
    
    c_post_ts = ts[condition][mouse]["Post"]
    #peaks_mk801_ts[condition][mouse]["Post"] = c_post_ts[c_post_peaks]

else:
    print("Recording already analyzed")

#%%

#Save additional peaks
with open(file_loc + r'\icv_photometry_peaks_mk801_081525.p', 'wb') as pp:
    response = input("Are you sure you want to overwrite peaks (Y/N)? It may be better to use a new name.")
    
    if response == "Y":
        pickle.dump(peaks_mk801, pp, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        pass
    
#%%

"""Check that the peaks found make sense"""

c_cond = "0.1mgml"
c_mouse = "FC3M5"

#'
fig, ax = plt.subplots(2,1, figsize=(12,6))
ax[0].plot(ts[c_cond][c_mouse]["Baseline"], p_data[c_cond][c_mouse]["Baseline"])
ax[0].scatter(ts[c_cond][c_mouse]["Baseline"][peaks_mk801[c_cond][c_mouse]["Baseline"]], p_data[c_cond][c_mouse]["Baseline"][peaks_mk801[c_cond][c_mouse]["Baseline"]], c="red")
ax[0].set_ylim(-0.01,0.035)
ax[0].set_title(f"{c_cond}, {c_mouse}")
ax[1].plot(ts[c_cond][c_mouse]["Post"], p_data[c_cond][c_mouse]["Post"])
ax[1].scatter(ts[c_cond][c_mouse]["Post"][peaks_mk801[c_cond][c_mouse]["Post"]], p_data[c_cond][c_mouse]["Post"][peaks_mk801[c_cond][c_mouse]["Post"]], c="red")
ax[1].set_ylim(-0.01,0.035)
plt.axis("off")
#plt.savefig(figure_dir + r"\example_01_mk801.eps", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(12,2))
ax.plot(ts[c_cond][c_mouse]["Post"], p_data[c_cond][c_mouse]["Post"])
#ax[1].scatter(ts[c_cond][c_mouse]["Post"][peaks_mk801[c_cond][c_mouse]["Post"]], p_data[c_cond][c_mouse]["Post"][peaks_mk801[c_cond][c_mouse]["Post"]], c="red")
#ax.set_ylim(-0.025,0.01)
#plt.axis("off")
#plt.savefig(figure_dir + r"\example_01mgml_mk801.eps", bbox_inches="tight")


#%%


peak_rate_df = []
for condition in peaks_mk801:
    c_condition_peaks = peaks_mk801[condition]
    
    for mouse in c_condition_peaks:
        c_peaks_baseline = c_condition_peaks[mouse]["Baseline"].size
        c_time_baseline = ts[condition][mouse]["Baseline"][-1] - ts[condition][mouse]["Baseline"][0]
        c_peak_rate_baseline = c_peaks_baseline/(c_time_baseline/60)
        
        c_peaks_post = c_condition_peaks[mouse]["Post"].size
        c_time_post = ts[condition][mouse]["Post"][-1] - ts[condition][mouse]["Post"][0]
        c_peak_rate_post = c_peaks_post/(c_time_post/60)
        
        
        c_df = pd.DataFrame({"Mouse": mouse, "Baseline": [c_peak_rate_baseline],
                             "Post": [c_peak_rate_post], "Treatment": condition})
        peak_rate_df.append(c_df)

peak_rate_df = pd.concat(peak_rate_df).reset_index(drop=True)
peak_rate_df["Change"] = peak_rate_df["Post"] / peak_rate_df["Baseline"]

doses = ["saline", "0.1mgml", "1mgml", "2mgml", '4mgml']
peak_rate_df["Treatment"] = pd.Categorical(peak_rate_df["Treatment"], ordered=True, categories=doses)

fig, ax = plt.subplots(figsize=(8,8))
#sns.lineplot(x="Treatment", y="Change", data=peak_rate_df, marker="o", linewidth=5, markersize=16, errorbar="se", err_style="bars", c="olive")
sns.boxplot(x="Treatment", y="Change", data=peak_rate_df, palette=["olive"])
sns.swarmplot(x="Treatment", y="Change", data=peak_rate_df, s=12, c="silver",alpha=0.5, dodge=True)
ax.set_xlabel("MK801 Dose (mg/ml)")
ax.set_ylabel("Trans. rate (%Baseline)")
ax.set_xticklabels(["Sal", "0.1", "1", "2", "4"])
sns.despine()
#plt.savefig(figure_dir + r"\Figure2_transient_quant_alldoses.eps", bbox_inches="tight")

#%%

scaling = "density"
freq_lim = 5

base_power = {}
post_power = {}
power_change = {}

for condition in p_data:
    base_power[condition] = {}
    post_power[condition] = {}
    power_change[condition] = {}
    
    for mouse in p_data[condition]:
        c_mouse = p_data[condition][mouse]
        
        c_baseline = c_mouse["Baseline"]
        c_post = c_mouse["Post"]

        c_base_psd = welch(c_baseline, 40, nperseg=512, scaling=scaling, detrend=False)
        c_post_psd = welch(c_post, 40, nperseg=512, scaling=scaling, detrend=False)

        c_base_freqs = c_base_psd[0][c_base_psd[0] < freq_lim]
        c_base_power = c_base_psd[1][c_base_psd[0] < freq_lim]
        base_power[condition][mouse] = c_base_power
        
        c_post_freqs = c_post_psd[0][c_post_psd[0] < freq_lim]
        c_post_power = c_post_psd[1][c_post_psd[0] < freq_lim]
        post_power[condition][mouse] = c_post_power
        
        change_power = c_post_power / c_base_power
        power_change[condition][mouse] = change_power
        

    power_change[condition] = pd.DataFrame(power_change[condition]).assign(Freqs=c_base_freqs, Treatment=condition)
    base_power[condition] = pd.DataFrame(base_power[condition]).assign(Freqs=c_base_freqs, Treatment=condition)
    post_power[condition] = pd.DataFrame(post_power[condition]).assign(Freqs=c_base_freqs, Treatment=condition)

    
#Concatenating
m_base_power = pd.concat([base_power[condition].melt(id_vars=["Freqs", "Treatment"]) for condition in base_power])
m_post_power = pd.concat([post_power[condition].melt(id_vars=["Freqs", "Treatment"]) for condition in post_power])

#Converting to decibels
m_base_power_2 = copy.deepcopy(m_base_power)
m_post_power_2 = copy.deepcopy(m_post_power)

m_base_power_2["value"] = 10*np.log10(m_base_power["value"])
m_post_power_2["value"] = 10*np.log10(m_post_power["value"])

doses = ["saline", "0.1mgml", "1mgml", "2mgml", '4mgml']
m_base_power_2["Treatment"] = pd.Categorical(m_base_power_2["Treatment"], ordered=True, categories=doses)
m_post_power_2["Treatment"] = pd.Categorical(m_post_power_2["Treatment"], ordered=True, categories=doses)


#Here we plot the baseline and post infusion power
fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Freqs", y="value", hue="Treatment", data=m_base_power_2, linewidth=5, errorbar="se", legend=False)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Baseline PSD (dB)")
sns.despine()
#plt.savefig(figure_dir + r"\Figure2_baselinepower_alldoses.eps")

fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Freqs", y="value", hue="Treatment", data=m_post_power_2, linewidth=5, errorbar="se")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Post-MK801 PSD (dB)")
sns.despine()
#plt.savefig(figure_dir + r"\Figure2_postpower_alldoses.eps")


#We calculate the change in power with respect to baseline
m_power_change = copy.deepcopy(m_post_power)
m_power_change["value"] = 10*np.log10(m_base_power["value"]/m_post_power["value"])

#Here we plot it
fig, ax = plt.subplots(figsize=(8,8))
sns.lineplot(x="Freqs", y="value", hue="Treatment", data=m_power_change, linewidth=5, errorbar="se")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"$\Delta$ power (dB)")
sns.despine()
#plt.savefig(figure_dir + r"\Figure2_changepower_alldoses.eps")

#Quantify change in power
avg_power_change = m_power_change.groupby(["Treatment", "variable"]).mean().reset_index(drop=False).dropna()

fig, ax = plt.subplots(figsize=(8,8))
sns.boxplot(x="Treatment", y="value", data=avg_power_change, palette=["olive"])
sns.swarmplot(x="Treatment", y="value", data=avg_power_change, palette=["silver"], s=10)
ax.set_xlabel("MK801 Dose (mg/ml)")
ax.set_ylabel("Trans. rate (%Baseline)")
ax.set_xticklabels(["Sal", "0.1", "1", "2", "4"])
ax.set_ylabel("Change in power (dB)")
sns.despine()
#plt.savefig(figure_dir + r"\Figure2_changepower_quant_alldoses.eps")