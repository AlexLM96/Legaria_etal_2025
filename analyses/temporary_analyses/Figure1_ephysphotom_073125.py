# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:32:41 2023

@author: Alex
"""

import numpy as np
import invian
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from invian.io import nexio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tqdm
import nex
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks, spectrogram, convolve, periodogram, find_peaks_cwt, filtfilt, butter, correlate, peak_prominences
import statsmodels.api as sm
from scipy.stats import ttest_rel, ttest_ind

plt.rcParams.update({'font.size': 24, 'figure.autolayout': True})
figure_dir = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\ephys_photom_systemic_MK801_OF\Figure_panels"

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
    
def plot_xcorr(x, y, sr): 
    "Plot cross-correlation (full) between two signals."
    N = max(len(x), len(y)) 
    n = min(len(x), len(y)) 

    if N == len(y): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 
    
    c = correlate(x / np.std(x), y / np.std(y), 'full') / n
    
    return c, lags/sr
    

#%%

#file_loc = r"C:\Users\alex.legariamacal\Desktop\fiber_array_mk801_OF"
#file_loc = r"C:\Users\Alex\Desktop\fiber_array_mk801"
file_loc = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\ephys_photom_systemic_MK801_OF\Data\mk801\processed"
#file_loc = r"D:\fiber_array_mk801_OF"


mk801_files = {
    "C97M1": file_loc + "\\c97m1_arrayfiber_mk801_090523_sorted.nex5",
    "C97M2": file_loc + "\\c97m2_arrayfiber_mk801_090523_sorted.nex5",
    "C97M3": file_loc + "\\c97m3_arrayfiber_mk801_090523_sorted.nex5",
    "C97M4": file_loc + "\\c97m4_arrayfiber_mk801_091523_sorted.nex5",
    "GA11": file_loc + "\\GA11_mk801_of_050124_processed.nex5",
    "GA12": file_loc + "\\GA12_mk801_of_041924_sorted.nex5"
    }

#%%

"""Here we load and pre-process the mk801 data"""

variables = {
    "neuron": "all",
    "continuous": ["norm_gcamp1"],
    "event": ["KBD1"]
    }

data = {mouse: nexio.get_variables(mk801_files[mouse], variables) for mouse in mk801_files}
neurons = {mouse: data[mouse]["neuron"] for mouse in data}
photom = {mouse: data[mouse]["continuous"]["norm_gcamp1"] for mouse in data}
injections = {mouse: data[mouse]["event"]["KBD1"] for mouse in data}

#%%

"""Convolving spiking activity to make analogous signal to photometry and aligning to injection"""

neurons_mk801_peh = {}
avg_neurons_mk801_peh = {}
photom_mk801_peh = {}
for mouse in mk801_files:
    c_neurons = neurons[mouse]
    c_neuron_peh = {neuron: convolve1d(invian.spiking_peh(c_neurons[neuron], injections[mouse], (-1200,2400), 0.1), np.ones(1), mode="mirror")
             for neuron in c_neurons}
    z_neuron_peh = {neuron: (c_neuron_peh[neuron] - c_neuron_peh[neuron].mean()) / c_neuron_peh[neuron].std() for neuron in c_neuron_peh}
    zall_neuron_peh = np.vstack(list(z_neuron_peh.values()))
    neurons_mk801_peh[mouse] = zall_neuron_peh
    avg_neurons_mk801_peh[mouse] = zall_neuron_peh.mean(axis=0)
    
    c_photom = photom[mouse]
    c_photom_peh = invian.contvar_peh(c_photom[0], c_photom[1], injections[mouse], (-1200,2400), 0.1)[0,:-1]
    #dff_photom_peh = c_photom_peh /c_photom_peh.min()
    #z_photom_peh = (dff_photom_peh - dff_photom_peh.mean()) / dff_photom_peh.std()
    z_photom_peh = (c_photom_peh - c_photom_peh.mean()) / c_photom_peh.std()
    photom_mk801_peh[mouse] = z_photom_peh
    
#%%

fig, ax = plt.subplots()
ax.plot(photom_mk801_peh["GA11"])

#%%

"""Here we divide the data into three intervals: pre MK801 (-20 to 0 mins), and  post MK801 (20 to 40 mins)"""

pre_mk801_neurons = {mouse: neurons_mk801_peh[mouse][:,:12000] for mouse in neurons_mk801_peh}
pre_mk801_avg_neuron = {mouse: avg_neurons_mk801_peh[mouse][:12000] for mouse in avg_neurons_mk801_peh}
pre_mk801_photom = {mouse: photom_mk801_peh[mouse][:12000] for mouse in photom_mk801_peh}

post_mk801_neurons = {mouse: neurons_mk801_peh[mouse][:,24000:] for mouse in neurons_mk801_peh}
post_mk801_avg_neuron = {mouse: avg_neurons_mk801_peh[mouse][24000:] for mouse in avg_neurons_mk801_peh}
post_mk801_photom = {mouse: photom_mk801_peh[mouse][24000:] for mouse in photom_mk801_peh}

#%%

"""Here find how good can photometry be predicted from spiking activity in the baseline and MK801 period"""

pre_w_corrs = {}
post_w_corrs = {}
for mouse in mk801_files:
    c_pre_neurons = pre_mk801_neurons[mouse]
    c_pre_avg_neurons = pre_mk801_avg_neuron[mouse]
    c_pre_photom = pre_mk801_photom[mouse]

    c_post_neurons = post_mk801_neurons[mouse]
    c_post_avg_neurons = post_mk801_avg_neuron[mouse]
    c_post_photom = post_mk801_photom[mouse]
    
    print(f"{mouse} pre-r: {np.corrcoef(c_pre_avg_neurons, c_pre_photom)[0,1]**2}, post-r: {np.corrcoef(c_post_avg_neurons, c_post_photom)[0,1]**2}")
    
    c_pre_regr = LinearRegression()
    c_pre_regr.fit(c_pre_neurons.T, c_pre_photom.reshape(-1,1))
    c_pre_score = c_pre_regr.score(c_pre_neurons.T, c_pre_photom.reshape(-1,1))
    pre_w_corrs[mouse] = [c_pre_score]
    
    c_post_regr = LinearRegression()
    c_post_regr.fit(c_post_neurons.T, c_post_photom.reshape(-1,1))
    c_post_score = c_post_regr.score(c_post_neurons.T, c_post_photom.reshape(-1,1))
    post_w_corrs[mouse] = [c_post_score]
    
pre_w_corrs_df = pd.DataFrame(pre_w_corrs)
pre_w_corrs_df = pre_w_corrs_df.assign(Stage="Baseline", Model="Weighted Avg.")
m_pre_w_corrs = pd.melt(pre_w_corrs_df, id_vars=["Stage", "Model"])

post_w_corrs_df = pd.DataFrame(post_w_corrs)
post_w_corrs_df = post_w_corrs_df.assign(Stage="MK801", Model="Weighted Avg.")
m_post_w_corrs = pd.melt(post_w_corrs_df, id_vars=["Stage", "Model"])

all_corrs = pd.concat([m_pre_w_corrs, m_post_w_corrs])

fig, ax = plt.subplots(figsize=(4,6))
sns.boxplot(x="Stage", y="value", data = all_corrs, palette=["darkcyan", "olive"])
sns.swarmplot(x="Stage", y="value", data = all_corrs, palette=["silver", "silver"], s=10)
ax.set_xlabel("")
ax.set_ylabel(r"$R^2$")
sns.despine()
ax.set_xticklabels(["Base", "MK801"])
#plt.savefig(figure_dir + "\\Prepost_corrs.eps", bbox_inches="tight")


#%%

"""Here we find peaks and look at the change in prominence post vs. pre"""

widths = np.arange(1,240)
min_snr = 5
prominence = 1
ts = np.arange(-1200,2399.9,0.1)/60


#Find peaks
#spk_peaks = {mouse: find_adjust_peaks(avg_neurons_mk801_peh[mouse], widths, min_snr, 10, 0) for mouse in photom_mk801_peh}
#photom_peaks = {mouse: find_adjust_peaks(photom_mk801_peh[mouse], widths, min_snr, 10, 0) for mouse in photom_mk801_peh}
spk_peaks = {mouse: find_peaks(avg_neurons_mk801_peh[mouse], prominence=1)[0] for mouse in photom_mk801_peh}
photom_peaks = {mouse: find_peaks(photom_mk801_peh[mouse], prominence=1.5)[0] for mouse in photom_mk801_peh}


pre_spk_peaks = {mouse:  spk_peaks[mouse][ts[spk_peaks[mouse]] < 0] for mouse in pre_mk801_photom}
post_spk_peaks = {mouse: spk_peaks[mouse][ts[spk_peaks[mouse]] > 20]for mouse in post_mk801_photom}
pre_photom_peaks = {mouse: photom_peaks[mouse][ts[photom_peaks[mouse]] < 0] for mouse in pre_mk801_photom}
post_photom_peaks = {mouse: photom_peaks[mouse][ts[photom_peaks[mouse]] > 20] for mouse in post_mk801_photom}

#Frequency
freq_pre_spk = {mouse: pre_spk_peaks[mouse].size for mouse in pre_spk_peaks}
freq_post_spk = {mouse: post_spk_peaks[mouse].size for mouse in post_spk_peaks}
freq_pre_photom = {mouse: pre_photom_peaks[mouse].size for mouse in pre_photom_peaks}
freq_post_photom = {mouse: post_photom_peaks[mouse].size for mouse in post_photom_peaks}

ratio_freq_spk_peaks = pd.DataFrame({mouse: [(freq_post_spk[mouse]/freq_pre_spk[mouse])] for mouse in freq_pre_spk})
ratio_freq_spk_peaks["Signal"] = "Spiking"
m_ratio_freq_spk_peaks = pd.melt(ratio_freq_spk_peaks, id_vars=["Signal"])

ratio_freq_photom_peaks = pd.DataFrame({mouse: [(freq_post_photom[mouse]/freq_pre_photom[mouse])] for mouse in freq_pre_photom})
ratio_freq_photom_peaks["Signal"] = "Photometry"
m_ratio_freq_photom_peaks = pd.melt(ratio_freq_photom_peaks, id_vars=["Signal"])

ratio_freq_peaks_both = pd.concat([m_ratio_freq_spk_peaks, m_ratio_freq_photom_peaks])
ratio_freq_peaks_both["value"] *= 100

fig, ax = plt.subplots(figsize=(4,6))
sns.boxplot(x="Signal", y="value", data = ratio_freq_peaks_both, palette=["steelblue", "goldenrod"])
sns.swarmplot(x="Signal", y="value", data = ratio_freq_peaks_both, s=10, palette=["silver", "silver"])
ax.set_ylabel(r"Peak Freq %")
ax.set_xlabel("")
ax.set_xticklabels(["Spk", "Photom"])
sns.despine()
plt.savefig(figure_dir + "\\peak_freq.eps", bbox_inches="tight")


#%%

s_mouse = "GA11"

c_all_spk_peaks = np.concatenate([pre_spk_peaks[s_mouse], post_spk_peaks[s_mouse]])
c_all_photom_peaks = np.concatenate([pre_photom_peaks[s_mouse], post_photom_peaks[s_mouse]])

fig, ax = plt.subplots(2,1, figsize=(12,6))
ax[0].plot(np.arange(-1200,2399.9,0.1)/60, avg_neurons_mk801_peh[s_mouse])
ax[0].scatter((np.arange(-1200,2399.9,0.1)/60)[spk_peaks[s_mouse]], avg_neurons_mk801_peh[s_mouse][spk_peaks[s_mouse]], c="red", alpha=0.5)
ax[0].axvline(0, color="red", linestyle="--")
ax[0].set_ylabel("Z-Score")
#ax[0].set_xticks([])
ax[0].axvspan(-20, 0,-0.5,2, color="darkcyan", alpha=0.3)
ax[0].axvspan(20,40,-0.5,2, color="olive", alpha=0.3)
ax[0].set_ylim(-1,2.7)
ax[1].plot(np.arange(-1200,2399.9,0.1)/60, hp_filter(photom_mk801_peh[s_mouse],0.005,10), color="goldenrod")
ax[1].scatter((np.arange(-1200,2399.9,0.1)/60)[photom_peaks[s_mouse]], hp_filter(photom_mk801_peh[s_mouse],0.005,10)[photom_peaks[s_mouse]], c="red", alpha=0.5)
ax[1].axvline(0, color="red", linestyle="--")
ax[1].axvspan(-20, 0,-0.5,2, color="darkcyan", alpha=0.3)
ax[1].axvspan(20,40,-0.5,2, color="olive", alpha=0.3)
ax[1].set_ylabel("Z-Score")
ax[1].set_xlabel("Time from MK801 Injection (minutes)")
ax[1].set_ylim(-3,6.2)
sns.despine()
plt.savefig(figure_dir + "\\sample_rec.eps", bbox_inches="tight")

#%%

fig, ax = plt.subplots()
ax.hist(avg_neurons_mk801_peh[s_mouse][spk_peaks[s_mouse]])
ax.hist(photom_mk801_peh[s_mouse][photom_peaks[s_mouse]])

        
#%%

"""Here we find the spiking and calcium activity around spiking bursts during the baseline and MK801 period."""

ts = np.arange(-1200,2399.9,0.1)
pre_ts = np.arange(-1200,0,0.1)
post_ts = np.arange(1200, 2399.9, 0.1)

pre_spk_peh = {}
pre_photom_peh = {}
post_spk_peh = {}
post_photom_peh = {}
for mouse in mk801_files:
    print(mouse)
    c_avg_spk = avg_neurons_mk801_peh[mouse]
    c_photom = photom_mk801_peh[mouse]
    
    c_pre_spk_peaks = pre_spk_peaks[mouse]
    c_post_spk_peaks = post_spk_peaks[mouse]
    
    c_pre_spk_peh = invian.contvar_peh(ts, c_avg_spk, ts[c_pre_spk_peaks], (-3,3), 0.1)
    norm_pre_spk_peh = c_pre_spk_peh - c_pre_spk_peh[:,:20].mean(axis=1)[:,np.newaxis]
    pre_spk_peh[mouse] = norm_pre_spk_peh
    
    c_pre_photom_peh = invian.contvar_peh(ts, c_photom, ts[c_pre_spk_peaks], (-3,3), 0.1)
    norm_pre_photom_peh = c_pre_photom_peh - c_pre_photom_peh[:,:20].mean(axis=1)[:,np.newaxis]
    pre_photom_peh[mouse] = norm_pre_photom_peh
    
    c_post_spk_peh = invian.contvar_peh(ts, c_avg_spk, ts[c_post_spk_peaks], (-3,3), 0.1)
    norm_post_spk_peh = c_post_spk_peh - c_post_spk_peh[:,:20].mean(axis=1)[:,np.newaxis]
    post_spk_peh[mouse] = norm_post_spk_peh
    
    c_post_photom_peh = invian.contvar_peh(ts, c_photom, ts[c_post_spk_peaks], (-3,3), 0.1)
    norm_post_photom_peh = c_post_photom_peh - c_post_photom_peh[:,:20].mean(axis=1)[:,np.newaxis]
    post_photom_peh[mouse] = norm_post_photom_peh

#%%

"""Here we find bursts in the baseline period that match the MK801 period and look at the photometry response around these bursts"""

all_f_pre_spk_peh = {}
all_f_post_spk_peh = {}
all_f_pre_photom_peh = {}
all_f_post_photom_peh = {}
for mouse in pre_spk_peh:
    test_pre_spk_peh = pre_spk_peh[mouse]
    test_post_spk_peh = post_spk_peh[mouse]
    
    test_pre_photom_peh = pre_photom_peh[mouse]
    test_post_photom_peh = post_photom_peh[mouse]
    
    test_pre_max = np.max(test_pre_spk_peh[:,25:35], axis=1)
    test_post_max = np.max(test_post_spk_peh[:,25:35], axis=1)
    
    f_pre_spk_peh = []
    f_post_spk_peh = []
    f_pre_photom_peh = []
    f_post_photom_peh = []
    for i in range(test_post_max.shape[0]):
        c_post_max = test_post_max[i]
        c_nearest_max = np.min(np.abs(test_pre_max-c_post_max))
        
        if c_nearest_max < 0.1:
            c_nearest_max_idx = np.argmin(np.abs(test_pre_max-c_post_max))
            f_pre_spk_peh.append(test_pre_spk_peh[c_nearest_max_idx,:])
            f_post_spk_peh.append(test_post_spk_peh[i,:])
            
            f_pre_photom_peh.append(test_pre_photom_peh[c_nearest_max_idx,:])
            f_post_photom_peh.append(test_post_photom_peh[i,:])
            
    f_pre_spk_peh = np.vstack(f_pre_spk_peh)
    all_f_pre_spk_peh[mouse] = f_pre_spk_peh
    
    f_post_spk_peh = np.vstack(f_post_spk_peh)    
    all_f_post_spk_peh[mouse] = f_post_spk_peh
    
    f_pre_photom_peh = np.vstack(f_pre_photom_peh)
    all_f_pre_photom_peh[mouse] = f_pre_photom_peh
    
    f_post_photom_peh = np.vstack(f_post_photom_peh)
    all_f_post_photom_peh[mouse] = f_post_photom_peh
        
all_f_pre_spk_peh_df = pd.concat([pd.DataFrame(all_f_pre_spk_peh[mouse]).T.assign(Time=np.arange(-3,3,0.1), Mouse=mouse).melt(id_vars=["Time", "Mouse"])
                     for mouse in all_f_pre_spk_peh])
all_f_post_spk_peh_df = pd.concat([pd.DataFrame(all_f_post_spk_peh[mouse]).T.assign(Time=np.arange(-3,3,0.1), Mouse=mouse).melt(id_vars=["Time", "Mouse"])
                     for mouse in all_f_post_spk_peh])
all_f_pre_photom_peh_df = pd.concat([pd.DataFrame(all_f_pre_photom_peh[mouse]).T.assign(Time=np.arange(-3,3,0.1), Mouse=mouse).melt(id_vars=["Time", "Mouse"])
                     for mouse in all_f_pre_photom_peh])
all_f_post_photom_peh_df = pd.concat([pd.DataFrame(all_f_post_photom_peh[mouse]).T.assign(Time=np.arange(-3,3,0.1), Mouse=mouse).melt(id_vars=["Time", "Mouse"])
                     for mouse in all_f_post_photom_peh])
    
fig, ax = plt.subplots(figsize=(4,6))
sns.lineplot(x="Time", y="value", data=all_f_pre_spk_peh_df, c="steelblue", legend=False, linewidth=3)
sns.lineplot(x="Time", y="value", data=all_f_pre_photom_peh_df, c="goldenrod", legend=False, linewidth=3)
ax.set_ylabel("Z-Score")
#ax.set_title("Baseline")
ax.set_xlabel("Time (s)")
ax.set_ylim(-0.1, 0.9)
ax.set_yticks(np.arange(0,1,0.4))
sns.despine()
plt.savefig(figure_dir + "\\match_spk_peh_baseline.eps", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4,6))
sns.lineplot(x="Time", y="value", data=all_f_post_spk_peh_df, c="steelblue", legend=False, linewidth=3)
sns.lineplot(x="Time", y="value", data=all_f_post_photom_peh_df, c="goldenrod", legend=False, linewidth=3)
ax.set_ylabel("Z-Score")
#ax.set_title("MK801")
ax.set_xlabel("Time (s)")
ax.set_ylim(-0.1, 0.9)
ax.set_yticks(np.arange(0,1,0.4))
sns.despine()
plt.savefig(figure_dir + "\\match_spk_peh_mk801.eps", bbox_inches="tight")

#Finding the max amplitude
pre_spk_max = pd.DataFrame({mouse: [np.mean(all_f_pre_spk_peh[mouse][:,25:35].max(axis=1))] for mouse in all_f_pre_spk_peh})
m_pre_spk_max = pd.melt(pre_spk_max.assign(Stage="Base", Type="Spk"), id_vars=["Stage", "Type"])

post_spk_max = pd.DataFrame({mouse: [np.mean(all_f_post_spk_peh[mouse][:,25:35].max(axis=1))] for mouse in all_f_post_spk_peh})
m_post_spk_max = pd.melt(post_spk_max.assign(Stage="MK801", Type="Spk"), id_vars=["Stage", "Type"])

pre_photom_max = pd.DataFrame({mouse: [np.mean(all_f_pre_photom_peh[mouse][:,25:35].max(axis=1))] for mouse in all_f_pre_photom_peh})
m_pre_photom_max = pd.melt(pre_photom_max.assign(Stage="Base", Type="Photom"), id_vars=["Stage", "Type"])

post_photom_max = pd.DataFrame({mouse: [np.mean(all_f_post_photom_peh[mouse][:,25:35].max(axis=1))] for mouse in all_f_post_photom_peh})
m_post_photom_max = pd.melt(post_photom_max.assign(Stage="MK801", Type="Photom"), id_vars=["Stage", "Type"])

all_max = pd.concat([m_pre_photom_max, m_post_photom_max])

fig, ax = plt.subplots(figsize=(3.7,6))
sns.boxplot(x="Stage", y="value", data=all_max, palette=["darkcyan", "olive"])
sns.swarmplot(x="Stage", y="value", data=all_max, s=10, palette=["silver", "silver"], legend=False)
ax.set_ylabel("Photometry Z-Score")
ax.set_xlabel("")
ax.set_ylim(0,1.05)
#ax.set_yticks(np.arange(0.2,0.9,0.2))
sns.despine()
plt.legend("")
plt.savefig(figure_dir + "\match_boxplot.eps", bbox_inches="tight")