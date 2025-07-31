# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:21:43 2025

@author: Alex
"""

import pickle
import invian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.signal import find_peaks, periodogram, spectrogram, filtfilt, butter, find_peaks_cwt, correlate, welch, peak_prominences, resample


figure_dir = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\cell_type_photometry_systemic_mk801\Figure_panels"
file_loc = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\cell_type_photometry_systemic_mk801\Data"
file_loc = r"C:\Users\alexmacal\Desktop\Legaria_etal_2025_data\Cell_type_openfield_mk801"
file_loc2 = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\ephys_photom_systemic_MK801_OF\Data\mk801\raw_data"

plt.rcParams.update({'font.size': 32, 'figure.autolayout': True, 'lines.linewidth': 2})

#%%

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

def butter_filter(signal, filt_type, freqs, sr, order=3):
    b,a = butter(order, freqs, btype=filt_type, fs=sr)
    y=filtfilt(b, a, signal, padtype="even")
    
    return y

#%%

other_fileloc = r"F:\Alex"

file = pd.read_csv(other_fileloc + r"\M484_D1_IP_MK801_OF_022725.csv")

fig, ax = plt.subplots()
ax.plot(file["Time"].iloc[100:], file["Fluorescence"].iloc[100:])

fig, ax = plt.subplots()
ax.plot(file["Time"].iloc[100:], file["Isosbestic"].iloc[100:])


test_file2 = pd.read_csv(file_loc + r"\M484_D1_IP_MK801_OF_022725.csv")

#%%

files = {
    "Saline": {
        "C97M1": pd.read_csv(file_loc + r"\c97m1_arrayfiber_saline_091823.csv"),
        "C97M2": pd.read_csv(file_loc + r"\c97m2_arrayfiber_saline_091923.csv"),
        "GA11": pd.read_csv(file_loc + r"\GA11_saline_042524.csv"),
        "GA12": pd.read_csv(file_loc + r"\GA12_saline_042624.csv"),
        "GAC3F1": pd.read_csv(file_loc + r"\GAC3F1_saline_112024.csv")
        },
    "Non-Selective": {
        "C97M1": pd.read_csv(file_loc + r"\C97M1_arrayfiber_mk801_090523.csv"),
        "C97M2": pd.read_csv(file_loc + r"\C97M2_arrayfiber_mk801_090523.csv"),
        "C97M3": pd.read_csv(file_loc + r"\C97M3_arrayfiber_mk801_090523.csv"),
        "C97M4": pd.read_csv(file_loc + r"\C97M4_arrayfiber_mk801_091523.csv"),
        "GA11": pd.read_csv(file_loc + r"\GA11_mk801_of_050124.csv"),
        "GA12": pd.read_csv(file_loc + r"\GA12_mk801_of_041924.csv")
        },
    "D1-Cre": {
        "F491": pd.read_csv(file_loc + r"\F491_D1_MK801_OFl_010625.csv").iloc[40:,:],
        "F488": pd.read_csv(file_loc + r"\F488_D1_MK801_OF_010525.csv"),
        "F490": pd.read_csv(file_loc + r"\F490_D1_MK801_OF_010525.csv").iloc[40:-800,:],
        "M484": pd.read_csv(file_loc + r"\M484_D1_IP_MK801_OF_022725.csv"),
        "F692": pd.read_csv(file_loc + r"\F692_D1_MK801_031825.csv")
        },
    "A2a-Cre": {
        "C79M1": pd.read_csv(file_loc + r"\C79M1_A2a_MK801_OF_042423.csv"),
        "C79M2": pd.read_csv(file_loc + r"\C79M2_A2a_MK801_OF_042523.csv"),
        "M456": pd.read_csv(file_loc + r"\M456_A2a_MK801_OF_010225.csv"),
        "M521": pd.read_csv(file_loc + r"\M521_A2a_MK801_OF_010225.csv").iloc[:-40,:],
        "M524": pd.read_csv(file_loc + r"\M524_A2a_MK801_OF_010525.csv").iloc[40:,:]
        }
    }

inj_times = {
    "Saline": {
        "C97M1": [58360],
        "C97M2": [47770],
        "GA11": [54970],
        "GA12": [44360],
        "GAC3F1": [57310]
        },
    "Non-Selective": {
        "C97M1": [42180],
        "C97M2": [52330],
        "C97M3": [59310],
        "C97M4": [62850],
        "GA11": [45900],
        "GA12": [66200]
        },
    "D1-Cre": {
        "F491": [48800],
        "F488": [63500],
        "F490": [58900],
        "M484": [61350],
        "F692": [60210]
        },
    "A2a-Cre": {
        "C79M1": [46420],
        "C79M2": [41400],
        "M456": [47941],
        "M521": [62850],
        "M524": [46800]
        },
    }

#%%

"""
Here we double-check that everything looks okay.
"""
strain = "Non-Selective"
mouse =  "GA11"

fig, ax = plt.subplots(2,1)
ax[0].plot(files[strain][mouse]["Time"], files[strain][mouse]["Fluorescence"])
ax[1].plot(files[strain][mouse]["Time"], files[strain][mouse]["Isosbestic"])


#%%

p_data = {}
ts = {}
for strain in files:
    p_data[strain] = {}
    ts[strain] = {}
    for mouse in files[strain]:

        c_data = files[strain][mouse]
        
        c_ts = c_data["Time"].iloc[150:].to_numpy()
        c_isos = c_data["Isosbestic"].iloc[150:].to_numpy()
        c_gcamp = c_data["Fluorescence"].iloc[150:].to_numpy()
        c_sr = np.diff(c_ts).mean()
        
        regr = LinearRegression()
        regr.fit(c_isos.reshape(-1,1), c_gcamp.reshape(-1,1))
        c_hat = regr.predict(c_isos.reshape(-1,1))
        
        
        c_norm = (c_gcamp - c_hat[:,0])
        #c_norm_baseline = c_baseline_gcamp_2
        #c_norm_2 = c_norm
        c_norm_2 = butter_filter(c_norm, 'bandpass', (0.001, 6), (1/c_sr))
        z_norm = (c_norm_2 - c_norm_2.mean()) / c_norm_2.std()
        #z_norm = c_norm_2
    
        p_data[strain][mouse] = z_norm
        ts[strain][mouse] = c_ts
        
#%%

"""
Here we double-check that everything looks okay.
"""
strain = "Non-Selective"
mouse =  "GA11"

fig, ax = plt.subplots()
ax.plot(ts[strain][mouse], p_data[strain][mouse])

#%%

"""Here we align the data to the injection time using peri-event function"""
peh_ts = np.linspace(-900, 1800, 54000)
start = -900
end = 1800
aligned_data = {}
for strain in p_data:
    aligned_data[strain] = {}
    for mouse in p_data[strain]:
        print(strain, mouse)
        c_ts = ts[strain][mouse]
        c_data = p_data[strain][mouse]
        c_inj = inj_times[strain][mouse]
        
        c_peh = invian.contvar_peh(c_ts, c_data, c_inj, (start,end), 0.05)[0,:]
        aligned_data[strain][mouse] = c_peh
        

#%%

"""Here we find the peaks in all the recordings"""

#Find peaks
widths = np.arange(1,160)
min_snr = 7
prominence = 1
        
#Find all peaks in all the recordings
all_peaks = {}
all_peak_ts = {}
all_peak_proms = {}
for strain in aligned_data:
    all_peaks[strain] = {}
    all_peak_ts[strain] = {}
    all_peak_proms[strain] = {}
    for mouse in aligned_data[strain]:
        print(strain, mouse)
        c_data = aligned_data[strain][mouse]
        c_peaks = find_peaks(c_data, prominence=2)
        c_peak_ts = peh_ts[c_peaks[0]]
        all_peaks[strain][mouse] = c_peaks[0]
        all_peak_ts[strain][mouse] = c_peak_ts
        all_peak_proms[strain][mouse] = c_peaks[1]["prominences"]
        
#%%

"""Here we visualize the peaks to make sure everything looks fine"""

strain = "Non-Selective"
mouse = "GA12"

#Visualize peaks
fig, ax = plt.subplots()
ax.plot(peh_ts, aligned_data[strain][mouse])
ax.scatter(peh_ts[all_peaks[strain][mouse]], aligned_data[strain][mouse][all_peaks[strain][mouse]], c="red")
        
#%%

"""Bin number of bins into 5 minute bins and plot peak rate"""

#Bin the number of peaks in 5 minute bins
bins = np.arange(-900, 1801, 300)
all_peak_hist = []
for strain in all_peaks:
    for mouse in all_peak_ts[strain]:
        print(strain, mouse)
        c_peak_ts = all_peak_ts[strain][mouse]
        c_peak_hist = np.histogram(c_peak_ts, bins=bins)[0]
        c_peak_hist = (c_peak_hist / c_peak_hist[0]) * 100
        c_peak_hist_df = pd.DataFrame(c_peak_hist)
        c_peak_hist_df = c_peak_hist_df.assign(Bin=np.arange(-10,31,5), Strain=strain, Mouse=mouse)
        m_c_peak_hist_df  = pd.melt(c_peak_hist_df, id_vars=["Bin", "Strain", "Mouse"])
        all_peak_hist.append(m_c_peak_hist_df)
                
concat_peak_hist = pd.concat(all_peak_hist)
        

strains = ["Saline", "Non-Selective", "D1-Cre", "A2a-Cre"]
concat_peak_hist["Strain"] = pd.Categorical(concat_peak_hist["Strain"], ordered=True, categories=strains)


#Plotting of peak rate over time
fig, ax = plt.subplots(figsize=(10,8))
sns.lineplot(x="Bin", y="value", hue="Strain", data=concat_peak_hist, errorbar="se", marker="o", palette=["darkcyan", "olive", "mediumblue", "maroon"], 
             markersize=8, linewidth=3, legend=False)
ax.set_xlabel("Time from injection (mins)")
ax.set_ylabel("Transient Rate (%Base)")
ax.axvline(0, c="red", linestyle="--", linewidth=2)
sns.despine()

#plt.savefig(figure_dir + r"\celltype_mk801_timecourse.eps")

#Now we quantify and plot the tansient in the last 10 minutes
last_10_peak_hist = concat_peak_hist[concat_peak_hist["Bin"] > 21]
last_10_peak_hist = last_10_peak_hist.groupby(["Strain", "Mouse"]).mean().reset_index()

fig, ax = plt.subplots(figsize=(6,8))
sns.boxplot(x="Strain", y="value", data=last_10_peak_hist, palette=["darkcyan", "olive", "mediumblue", "maroon"])
sns.swarmplot(x="Strain", y="value", data=last_10_peak_hist, c="silver", size=10)
ax.set_xlabel("")
ax.set_ylabel("Trans. Rate (%Base)")
sns.despine()
ax.set_xticklabels(["Sal", "N.S.", "D1", "A2a"])

#plt.savefig(figure_dir + r"\celltype_mk801_boxplot.eps")

#%%
        

