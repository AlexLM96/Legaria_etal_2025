# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:45:59 2024

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
#from influxdb import DataFrameClient
import copy
import math
from scipy.stats import ttest_rel, ttest_ind

plt.rcParams.update({'font.size': 28, 'figure.autolayout': True})

cat_size_order = CategoricalDtype(["Sal", "MK801_2ug", "MK801_4ug"], ordered=True)

figure_dir = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\Dissertation_figures\Figure_panels"

#%%

file_loc = r"C:\Users\Alex\Desktop\ICV_mk801_all"
file_loc2 = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\Bandit_mk801_behavior\icv_4ug_MK801_bandit\Data_2"

saline_2ug_data = {
    "saline_1": {
        "C116F2": pd.read_csv(file_loc + "\\saline_2ug_C116F2_icv_060523.csv"),
        "C116F4": pd.read_csv(file_loc + "\\saline_2ug_C116F4_icv_060523.csv")
        },
    "saline_2": {
        "C75M1": pd.read_csv(file_loc + "\\saline_2ug_C75M1_icv_060623.csv"),
        "C75M3": pd.read_csv(file_loc + "\\saline_2ug_C75M3_icv_060623.csv"),       
        },
    "saline_3": {
        "C116F1": pd.read_csv(file_loc + "\\saline_2ug_C116F1_icv_061323.csv"),
        },
    "saline_4": {
        "C48M3": pd.read_csv(file_loc + "\\saline_2ug_C48M3_icv_062023.csv"),
        "C50F5": pd.read_csv(file_loc + "\\saline_2ug_C50F5_icv_062023.csv")
        },
    "saline_5": {
        "C50F1": pd.read_csv(file_loc + "\\saline_2ug_C50F1_icv_063023.csv")
        },
    "saline_6": {
        "C129F1": pd.read_csv(file_loc + "\\saline_2ug_C129F1_icv_091223.csv"),
        "C84M2": pd.read_csv(file_loc + "\\saline_2ug_C84M2_icv_091223.csv")
        },
    "saline_7": {
        "C86F1": pd.read_csv(file_loc + "\\saline_2ug_C86F1_icv_091823.csv"),
        "C86F2": pd.read_csv(file_loc + "\\saline_2ug_C86F2_icv_091823.csv")
        }
    }

saline_2ug_injectiontimes = {
    "saline_1": datetime.datetime(2023, 6, 5, 19),
    "saline_2": datetime.datetime(2023, 6, 6, 18, 30),
    "saline_3": datetime.datetime(2023, 6, 13, 17, 45),
    "saline_4": datetime.datetime(2023, 6, 20, 18, 5),
    "saline_5": datetime.datetime(2023, 6, 30, 16, 45),
    "saline_6": datetime.datetime(2023, 9, 12, 19, 10),
    "saline_7": datetime.datetime(2023, 9, 18, 18, 45)
    }


p_saline_2ug_data = {}
for session in saline_2ug_data:
    print(f"Processing {session}")
    c_session = saline_2ug_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_saline_2ug_data[session] = p_session
    
#MK801 2ug
mk801_2ug_data = {
    "mk801_1": {
        "C48M3": pd.read_csv(file_loc + "\\mk801_2ug_C48M3_icv_060623.csv")
        },
    "mk801_2": {
        "C116F2": pd.read_csv(file_loc + "\\mk801_2ug_C116F2_icv_061323.csv"),
        "C116F4": pd.read_csv(file_loc + "\\mk801_2ug_C116F4_icv_061323.csv")
        },
    "mk801_3": {
        "C75M1": pd.read_csv(file_loc + "\\mk801_2ug_C75M1_icv_062023.csv"),
        "C116F1": pd.read_csv(file_loc + "\\mk801_2ug_C116F1_icv_m062023.csv")
        },
    "mk801_4": {
        "C75M3": pd.read_csv(file_loc + "\\mk801_2ug_C75M3_icv_062623.csv"),
        "C50F1": pd.read_csv(file_loc + "\\mk801_2ug_C50F1_icv_062623.csv"),
        "C50F5": pd.read_csv(file_loc + "\\mk801_2ug_C75M3_icv_062623.csv")
        },
    "mk801_5": {
        "C86F1": pd.read_csv(file_loc + "\\mk801_2ug_C86F1_icv_091223.csv"),
        "C86F2": pd.read_csv(file_loc + "\\mk801_2ug_C86F2_icv_091223.csv")
        },
    "mk801_6": {
        "C129F1": pd.read_csv(file_loc + "\\mk801_2ug_C129F1_icv_091823.csv"),
        "C84M2": pd.read_csv(file_loc + "\\mk801_2ug_C84M2_icv_091823.csv")
        }
    }

mk801_2ug_injectiontimes = {
    "mk801_1": datetime.datetime(2023, 6, 6, 18, 30),
    "mk801_2": datetime.datetime(2023, 6, 13, 16, 22),
    "mk801_3": datetime.datetime(2023, 6, 20, 18, 45),
    "mk801_4": datetime.datetime(2023, 6, 26, 18, 45),
    "mk801_5": datetime.datetime(2023, 9, 12, 19, 10),
    "mk801_6": datetime.datetime(2023, 9, 18, 18, 45)
    }

p_mk801_2ug_data = {}
for session in mk801_2ug_data:
    print(f"Processing {session}")
    c_session = mk801_2ug_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_mk801_2ug_data[session] = p_session

#
saline_4ug_data = {
    "saline_1": {
        "C116F3": pd.read_csv(file_loc + "\\saline_4ug_C116F3_icv_053023.csv"),
        },
    "saline_2": {
        "C48M3": pd.read_csv(file_loc + "\\saline_4ug_C48M3_icv_060123.csv"),
        },
    "saline_3": {
        "C50F5": pd.read_csv(file_loc + "\\saline_4ug_C50F5_icv_060523.csv"),
        "C116F1": pd.read_csv(file_loc + "\\saline_4ug_C116F1_icv_060523.csv"),
        },
    "saline_4": {
        "C86F1": pd.read_csv(file_loc + "\\saline_4ug_C86F1_icv_092523.csv"),
        "C86F2": pd.read_csv(file_loc + "\\saline_4ug_C86F2_icv_092523.csv")
        },
    "saline_5": {
        "C129F1": pd.read_csv(file_loc + "\\saline_4ug_C129F1_icv_100423.csv"),
        "C84M2": pd.read_csv(file_loc + "\\saline_4ug_C84M2_icv_100423.csv")
        },
    "saline_6": {
        "FCK4M1": pd.read_csv(file_loc2 + r"\saline_4ug_FCK4M1_111224_01.CSV"),
        "FCK4M3": pd.read_csv(file_loc2 + r"\saline_4ug_FCK4M3_111224_00.CSV"),
        },
    "saline_7": {
        "FCK3F1": pd.read_csv(file_loc2 + r"\saline_4ug_FCK3F1_111324_00.CSV")
        },
    "saline_8": {
        "FCK4M2": pd.read_csv(file_loc2 + r"\saline_4ug_FCK4M2_111924_02.CSV"),
        "FCK3F3": pd.read_csv(file_loc2 + r"\saline_4ug_FCK3F3_111924_00.CSV")
        },
    "saline_9": {
        "FCK5M3": pd.read_csv(file_loc2 + r"\saline_4ug_FCK5M3_010925_01.CSV"),
        "FCK5M4": pd.read_csv(file_loc2 + r"\saline_4ug_FCK5M4_010925_01.CSV")
        },
    }

saline_4ug_injectiontimes = {
    "saline_1": datetime.datetime(2023, 5, 30, 18, 45),
    "saline_2": datetime.datetime(2023, 6, 1, 18, 30),
    "saline_3": datetime.datetime(2023, 6, 5, 19, 15),
    "saline_4": datetime.datetime(2023, 9, 25, 19, 40),
    "saline_5": datetime.datetime(2023, 10, 4, 19, 10),
    "saline_6": datetime.datetime(2024, 11, 12, 19, 45),
    "saline_7": datetime.datetime(2024, 11, 13, 20, 55),
    "saline_8": datetime.datetime(2024, 11, 19, 18, 43), 
    "saline_9": datetime.datetime(2025, 1, 9, 17, 12) 
    }

p_saline_4ug_data = {}
for session in saline_4ug_data:
    print(f"Processing {session}")
    c_session = saline_4ug_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_saline_4ug_data[session] = p_session
    
#C84M2 FED was off by 1 hour on 10/04. Here we fix it
p_saline_4ug_data["saline_5"]["C84M2"].iloc[:,0] = p_saline_4ug_data["saline_5"]["C84M2"].iloc[:,0] - datetime.timedelta(hours=1)

#MK801 Data    
mk801_4ug_data = {
    "mk801_1": {
        "C50F5": pd.read_csv(file_loc + "\\mk801_4ug_C50F5_icv_053023.csv"),
        "C116F2": pd.read_csv(file_loc + "\\mk801_4ug_C116F2_icv_053023.csv"),
        "C116F4": pd.read_csv(file_loc + "\\mk801_4ug_C116F4_icv_053023.csv")
        },
    "mk801_2": {
        "C75M1": pd.read_csv(file_loc + "\\mk801_4ug_C75M1_icv_060123.csv"),
        "C75M2": pd.read_csv(file_loc + "\\mk801_4ug_C75M2_icv_060123.csv"),
        "C75M3": pd.read_csv(file_loc + "\\mk801_4ug_C75M3_icv_060123.csv"),
        },
    "mk801_3": {
        "C129F1": pd.read_csv(file_loc + "\\mk801_4ug_C129F1_icv_092523.csv"),
        "C84M2": pd.read_csv(file_loc + "\\mk801_4ug_C84M2_icv_092523.csv")
        },
    "mk801_4": {
        "C86F1": pd.read_csv(file_loc + "\\mk801_4ug_C86F1_icv_100423.csv"),
        "C86F2": pd.read_csv(file_loc + "\\mk801_4ug_C86F2_icv_100423.csv")
        },
    "mk801_5": {
        "FCK4M1": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK4M1_112524_00.CSV"),
        "FCK4M3": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK4M3_112524_01.CSV"),
        },
    "mk801_6": {
        "FCK3F1": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK3F1_111824_00.CSV")
        },
    "mk801_7": {
        "FCK4M2": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK4M2_112524_00.CSV"),
        "FCK3F3": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK3F3_112524_01.CSV")
        },
    "mk801_8": {
        "FCK5M3": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK5M3_011725_00.csv"),
        "FCK5M4": pd.read_csv(file_loc2 + r"\mk801_4ug_FCK5M4_011625_00.csv")
        },
    }

mk801_4ug_injectiontimes = {
    "mk801_1": datetime.datetime(2023, 5, 30, 18, 45),
    "mk801_2": datetime.datetime(2023, 6, 1, 18, 30),
    "mk801_3": datetime.datetime(2023, 9, 25, 19, 40),
    "mk801_4": datetime.datetime(2023, 10, 4, 19, 10),
    "mk801_5": datetime.datetime(2024, 11, 25, 20, 20),
    "mk801_6": datetime.datetime(2024, 11, 18, 20, 37),
    "mk801_7": datetime.datetime(2024, 11, 25, 20, 20),
    "mk801_8": datetime.datetime(2025, 1, 16, 17, 59)
    }

p_mk801_4ug_data = {}
for session in mk801_4ug_data:
    print(f"Processing {session}")
    c_session = mk801_4ug_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        f_mouse.iloc[:,0] = pd.to_datetime(f_mouse.iloc[:,0])
        p_session[mouse] = f_mouse
    
    p_mk801_4ug_data[session] = p_session
    

#%%

"""Here we get the slice of the data that goes from 0 to 8 hours after the injection."""

post_hours = 8

# The saline groups
saline_2ug_slices = {}
for session in p_saline_2ug_data:
    c_session = p_saline_2ug_data[session]
    c_injection = saline_2ug_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    saline_2ug_slices[session] = s_session
       
# The MK801 groups                               
mk801_2ug_slices = {}
for session in p_mk801_2ug_data:
    c_session = p_mk801_2ug_data[session]
    c_injection = mk801_2ug_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    mk801_2ug_slices[session] = s_session


# The saline groups
saline_4ug_slices = {}
for session in p_saline_4ug_data:
    c_session = p_saline_4ug_data[session]
    c_injection = saline_4ug_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    saline_4ug_slices[session] = s_session
       
# The MK801 groups                               
mk801_4ug_slices = {}
for session in p_mk801_4ug_data:
    c_session = p_mk801_4ug_data[session]
    c_injection = mk801_4ug_injectiontimes[session]
    s_session = {mouse: c_session[mouse][np.logical_and(c_session[mouse].iloc[:,0] < c_injection + datetime.timedelta(hours=post_hours),
                                                        c_session[mouse].iloc[:,0] > c_injection)] for mouse in c_session}
    mk801_4ug_slices[session] = s_session

#%%

"""Reversal PEHs"""

m_sal_2ug_pehs = {}
for session in saline_2ug_slices:
    print(session)
    c_session = saline_2ug_slices[session]
    c_pehs = {mouse: f3b.reversal_peh(c_session[mouse], (-10,11)).mean(axis=0) for mouse in c_session}
    c_pehs_df = pd.DataFrame(c_pehs)
    c_pehs_df = c_pehs_df.assign(Trial=np.arange(-10,11), Treatment="Sal")
    m_c_pehs = pd.melt(c_pehs_df, id_vars=["Trial", "Treatment"])
    m_sal_2ug_pehs[session] = m_c_pehs

m_sal_2ug_pehs = pd.concat(m_sal_2ug_pehs.values())

m_mk801_2ug_pehs = {}
for session in mk801_2ug_slices:
    print(session)
    c_session = mk801_2ug_slices[session]
    c_pehs = {mouse: f3b.reversal_peh(c_session[mouse], (-10,11)).mean(axis=0) for mouse in c_session}
    c_pehs_df = pd.DataFrame(c_pehs)
    c_pehs_df = c_pehs_df.assign(Trial=np.arange(-10,11), Treatment="MK801_2ug")
    m_c_pehs = pd.melt(c_pehs_df, id_vars=["Trial", "Treatment"])
    m_mk801_2ug_pehs[session] = m_c_pehs

m_mk801_2ug_pehs = pd.concat(m_mk801_2ug_pehs.values())
m_all_2ug_pehs = pd.concat([m_sal_2ug_pehs, m_mk801_2ug_pehs])

#Accuracy of trials prior to reversal
all_2ug_pre_pehs = m_all_2ug_pehs[m_all_2ug_pehs["Trial"] < 0]
mean_2ug_pre_pehs = all_2ug_pre_pehs.groupby(["Treatment", "variable"]).mean().reset_index()
mean_2ug_pre_pehs["Treatment"] = mean_2ug_pre_pehs["Treatment"].astype(cat_size_order)
mean_2ug_pre_pehs = mean_2ug_pre_pehs.sort_values(by="Treatment")

#Now the 4 ug
m_sal_4ug_pehs = {}
for session in saline_4ug_slices:
    print(session)
    c_session = saline_4ug_slices[session]
    c_pehs = {mouse: f3b.reversal_peh(c_session[mouse], (-10,11)).mean(axis=0) for mouse in c_session}
    c_pehs_df = pd.DataFrame(c_pehs)
    c_pehs_df = c_pehs_df.assign(Trial=np.arange(-10,11), Treatment="Sal")
    m_c_pehs = pd.melt(c_pehs_df, id_vars=["Trial", "Treatment"])
    m_sal_4ug_pehs[session] = m_c_pehs

m_sal_4ug_pehs = pd.concat(m_sal_4ug_pehs.values())



m_mk801_4ug_pehs = {}
for session in mk801_4ug_slices:
    print(session)
    c_session = mk801_4ug_slices[session]
    c_pehs = {mouse: f3b.reversal_peh(c_session[mouse], (-10,11)).mean(axis=0) for mouse in c_session}
    c_pehs_df = pd.DataFrame(c_pehs)
    c_pehs_df = c_pehs_df.assign(Trial=np.arange(-10,11), Treatment="MK801_4ug")
    m_c_pehs = pd.melt(c_pehs_df, id_vars=["Trial", "Treatment"])
    m_mk801_4ug_pehs[session] = m_c_pehs

m_mk801_4ug_pehs = pd.concat(m_mk801_4ug_pehs.values())
m_all_4ug_pehs = pd.concat([m_sal_4ug_pehs, m_mk801_4ug_pehs])

#Accuracy of trials prior to reversal
all_4ug_pre_pehs = m_all_4ug_pehs[m_all_4ug_pehs["Trial"] < 0]
mean_4ug_pre_pehs = all_4ug_pre_pehs.groupby(["Treatment", "variable"]).mean().reset_index()
mean_4ug_pre_pehs["Treatment"] = mean_4ug_pre_pehs["Treatment"].astype(cat_size_order)
mean_4ug_pre_pehs = mean_4ug_pre_pehs.sort_values(by="Treatment")

# Here we pool everything
m_all_pehs = pd.concat([m_sal_2ug_pehs, m_sal_4ug_pehs, m_mk801_2ug_pehs, m_mk801_4ug_pehs])

fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(x="Trial", y="value", hue="Treatment", data=m_all_pehs, linewidth=4, palette=["darkcyan", "chocolate", "olive"], errorbar="se")
ax.set_ylabel("P(high port)")
ax.set_xlabel("Trial from reversal")
ax.set_yticks(np.arange(0.3, 1, 0.2))
sns.despine()
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
#plt.savefig(figure_dir + "\\rev_peh_2doses_icv_mk801.eps", bbox_inches="tight")

#
#Accuracy of trials prior to reversal
all_pre_pehs = m_all_pehs[m_all_pehs["Trial"] < 0]
mean_pre_pehs = all_pre_pehs.groupby(["Treatment", "variable"]).mean().reset_index()
mean_pre_pehs["Treatment"] = mean_pre_pehs["Treatment"].astype(cat_size_order)
mean_pre_pehs = mean_pre_pehs.sort_values(by="Treatment")


fig, ax = plt.subplots(figsize=(6, 8))
sns.boxplot(x="Treatment", y="value", data=mean_pre_pehs, palette=["darkcyan", 
            "chocolate", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.swarmplot(x="Treatment", y="value", data=mean_pre_pehs, palette=["silver", "silver", "silver"], s=10)
ax.set_ylabel("Pre-reversal accuracy")
ax.set_xlabel("")
ax.set_yticks(np.arange(0.3, 1, 0.2))
ax.set_xticklabels(["Sal", "2ug", "4ug"])
sns.despine()
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
#plt.savefig(figure_dir + "\\pre_rev_2doses_icv_mk801.eps", bbox_inches="tight")

#%%

saline_2ug_ws = {}
for session in saline_2ug_slices:
    c_session = saline_2ug_slices[session]
    c_session_ws = {mouse: [f3b.win_stay(c_session[mouse])] for mouse in c_session}
    saline_2ug_ws = saline_2ug_ws | c_session_ws
    
saline_2ug_ws = pd.DataFrame(saline_2ug_ws).T
saline_2ug_ws["Treatment"] = "Sal"
saline_2ug_ws = saline_2ug_ws.reset_index()
saline_2ug_ws.columns = ["Mouse", "win-stay", "Treatment"]

# Count of ws after mk801 injections
mk801_2ug_ws = {}
for session in mk801_2ug_slices:
    c_session = mk801_2ug_slices[session]
    c_session_ws = {mouse: [f3b.win_stay(c_session[mouse])] for mouse in c_session}
    mk801_2ug_ws = mk801_2ug_ws | c_session_ws
    
mk801_2ug_ws = pd.DataFrame(mk801_2ug_ws).T.reset_index()
mk801_2ug_ws["Treatment"] = "MK801_2ug"
mk801_2ug_ws.columns = ["Mouse", "win-stay", "Treatment"]

#
saline_4ug_ws = {}
for session in saline_4ug_slices:
    c_session = saline_4ug_slices[session]
    c_session_ws = {mouse: [f3b.win_stay(c_session[mouse])] for mouse in c_session}
    saline_4ug_ws = saline_4ug_ws | c_session_ws
    
saline_4ug_ws = pd.DataFrame(saline_4ug_ws).T
saline_4ug_ws["Treatment"] = "Sal"
saline_4ug_ws = saline_4ug_ws.reset_index()
saline_4ug_ws.columns = ["Mouse", "win-stay", "Treatment"]


# Count of ws after mk801 injections
mk801_4ug_ws = {}
for session in mk801_4ug_slices:
    c_session = mk801_4ug_slices[session]
    c_session_ws = {mouse: [f3b.win_stay(c_session[mouse])] for mouse in c_session}
    mk801_4ug_ws = mk801_4ug_ws | c_session_ws
    
mk801_4ug_ws = pd.DataFrame(mk801_4ug_ws).T
mk801_4ug_ws["Treatment"] = "MK801_4ug"
mk801_4ug_ws = mk801_4ug_ws.reset_index()
mk801_4ug_ws.columns = ["Mouse", "win-stay", "Treatment"]


# We pool all the numbers
all_ws = pd.concat([saline_2ug_ws, saline_4ug_ws, mk801_2ug_ws, mk801_4ug_ws], axis=0)
all_ws["Treatment"] = all_ws["Treatment"].astype(cat_size_order)
all_ws = all_ws.groupby(["Treatment", "Mouse"]).mean().reset_index()
all_ws = all_ws.sort_values(by="Treatment")

fig, ax = plt.subplots(figsize=(6, 8))
sns.boxplot(x="Treatment", y="win-stay", data=all_ws, palette=[
            "darkcyan", "chocolate", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.swarmplot(x="Treatment", y="win-stay", data=all_ws, palette=["silver", "silver", "silver"], s=10)
ax.set_ylabel("Win-stay")
ax.set_xlabel("")
ax.set_xticklabels(["Sal", "2ug", "4ug"])
ax.set_ylim(0,1)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\ws_2doses_icv_mk801.eps", bbox_inches="tight")


#%%

saline_2ug_pcoeffs = {}
for session in saline_2ug_slices:
    c_session = saline_2ug_slices[session]
    c_session_sidep = {mouse: f3b.side_prewards(c_session[mouse]) for mouse in c_session}
    c_session_preX = {mouse: f3b.create_X(c_session[mouse], c_session_sidep[mouse], 5)
                      for mouse in c_session_sidep}
    c_session_pcoeffs = {mouse: f3b.logit_regr(c_session_preX[mouse]).params for mouse in c_session_preX}
    saline_2ug_pcoeffs = saline_2ug_pcoeffs | c_session_pcoeffs
    
saline_2ug_pcoeffs = pd.DataFrame(saline_2ug_pcoeffs)
saline_2ug_pcoeffs = saline_2ug_pcoeffs.assign(Treatment="Sal", Trial=np.flip(np.arange(-5, 0, 1)))
m_saline_2ug_pcoeffs = pd.melt(saline_2ug_pcoeffs, id_vars=["Treatment", "Trial"])

#Saline from 4ug
saline_4ug_pcoeffs = {}
for session in saline_4ug_slices:
    c_session = saline_4ug_slices[session]
    c_session_sidep = {mouse: f3b.side_prewards(c_session[mouse]) for mouse in c_session}
    c_session_preX = {mouse: f3b.create_X(c_session[mouse], c_session_sidep[mouse], 5)
                      for mouse in c_session_sidep}
    c_session_pcoeffs = {mouse: f3b.logit_regr(c_session_preX[mouse]).params for mouse in c_session_preX}
    saline_4ug_pcoeffs = saline_4ug_pcoeffs | c_session_pcoeffs
    
saline_4ug_pcoeffs = pd.DataFrame(saline_4ug_pcoeffs)
saline_4ug_pcoeffs = saline_4ug_pcoeffs.assign(Treatment="Sal", Trial=np.flip(np.arange(-5, 0, 1)))
m_saline_4ug_pcoeffs = pd.melt(saline_4ug_pcoeffs, id_vars=["Treatment", "Trial"])

# Count of ws after mk801 injections
mk801_2ug_pcoeffs = {}
for session in mk801_2ug_slices:
    print(session)
    c_session = mk801_2ug_slices[session]
    c_session_sidep = {mouse: f3b.side_prewards(c_session[mouse]) for mouse in c_session}
    c_session_preX = {mouse: f3b.create_X(c_session[mouse], c_session_sidep[mouse], 5)
                      for mouse in c_session_sidep}
    c_session_pcoeffs = {mouse: f3b.logit_regr(c_session_preX[mouse]).params for mouse in c_session_preX}
    mk801_2ug_pcoeffs = mk801_2ug_pcoeffs | c_session_pcoeffs
    
mk801_2ug_pcoeffs = pd.DataFrame(mk801_2ug_pcoeffs)
mk801_2ug_pcoeffs = mk801_2ug_pcoeffs.assign(Treatment="MK801_2ug", Trial=np.flip(np.arange(-5, 0, 1)))
m_mk801_2ug_pcoeffs = pd.melt(mk801_2ug_pcoeffs, id_vars=["Treatment", "Trial"])

#Th4 4ug
mk801_4ug_pcoeffs = {}
for session in mk801_4ug_slices:
    print(session)
    c_session = mk801_4ug_slices[session]
    c_session_sidep = {mouse: f3b.side_prewards(c_session[mouse]) for mouse in c_session}
    c_session_preX = {mouse: f3b.create_X(c_session[mouse], c_session_sidep[mouse], 5)
                      for mouse in c_session_sidep}
    c_session_pcoeffs = {mouse: f3b.logit_regr(c_session_preX[mouse]).params for mouse in c_session_preX}
    mk801_4ug_pcoeffs = mk801_4ug_pcoeffs | c_session_pcoeffs
    
mk801_4ug_pcoeffs = pd.DataFrame(mk801_4ug_pcoeffs)
mk801_4ug_pcoeffs = mk801_4ug_pcoeffs.assign(Treatment="MK801_4ug", Trial=np.flip(np.arange(-5, 0, 1)))
m_mk801_4ug_pcoeffs = pd.melt(mk801_4ug_pcoeffs, id_vars=["Treatment", "Trial"])


m_all_pcoeffs = pd.concat([m_saline_2ug_pcoeffs, m_saline_4ug_pcoeffs, m_mk801_2ug_pcoeffs, m_mk801_4ug_pcoeffs])

fig, ax = plt.subplots(figsize=(8, 8))
sns.pointplot(x="Trial", y="value", hue="Treatment", data=m_all_pcoeffs,
              palette=["darkcyan", "chocolate", "olive"], errwidth=3, errorbar="se")
for line in ax.lines:
    line.set_linewidth(4)
ax.set_xlabel("Trial in past")
ax.set_ylabel("Regr. Coefficient")
ax.set_yticks(np.arange(0,2.01,0.5))
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
sns.despine()
plt.savefig(figure_dir + r"\plogreg_2doses_icv_mk801.eps", bbox_inches="tight")


all_paucs = m_all_pcoeffs.groupby(by=["Treatment", "Trial", "variable"]).mean().reset_index()
all_paucs = all_paucs.groupby(by=["Treatment", "variable"]).sum().reset_index()
all_paucs["Treatment"] = all_paucs["Treatment"].astype(cat_size_order)
all_paucs = all_paucs.sort_values(by="Treatment")

# Plotting
fig, ax = plt.subplots(figsize=(6, 8))
sns.boxplot(x="Treatment", y="value", data=all_paucs, palette=[
            "darkcyan", "chocolate", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.swarmplot(x="Treatment", y="value", data=all_paucs,
              palette=["silver", "silver", "silver"], s=10)
ax.set_ylabel("Regr. Coeff. AUC")
ax.set_xlabel("")
ax.set_yticks(np.arange(-1,5,1))
ax.set_ylim(-1,5)
ax.set_xticklabels(["Sal", "2ug", "4ug"])
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
plt.savefig(figure_dir + r"\paucs_2doses_icv_mk801.eps", bbox_inches="tight")