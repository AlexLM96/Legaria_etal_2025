# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:30:06 2023

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import fed3bandit as f3b
import copy
from scipy.stats import ttest_rel

plt.rcParams.update({'font.size': 24, 'figure.autolayout': False})

from pandas.api.types import CategoricalDtype
cat_size_order = CategoricalDtype(["Sal", "MK801"], ordered=True)

figure_dir = r"C:\Users\Alex\Box\Kravitz Lab Box Drive\Alex\Projects\Thesis project\Systemic_MK801_bandit\figure_panels"



#%%

file_loc = r"C:\Users\Alex\Desktop\Legaria_etal_2025\data\temporary_data\Figure_s6"
file_loc_mk801 = r"C:\Users\Alex\Desktop\mk801_systemic\MK801_cohort"
post_hours = 8

saline_data = {
    "saline_1": {
        "C48M1": pd.read_csv(file_loc + "\\saline_C48M1_022723.csv"),
        "C48M3": pd.read_csv(file_loc + "\\saline_C48M3_022723.csv"),
        "C50F1": pd.read_csv(file_loc + "\\saline_C50F1_022723.csv"),
        "C50F5": pd.read_csv(file_loc + "\\saline_C50F5_022723.csv"),
        "C116F2": pd.read_csv(file_loc + "\\saline_C116F2_022723.csv"),
        "C116F4": pd.read_csv(file_loc + "\\saline_C116F4_022723.csv")
        },
    "saline_2": {
        "C75M1": pd.read_csv(file_loc + "\\saline_C75M1_021723_2.csv"),
        "C75M2": pd.read_csv(file_loc + "\\saline_C75M2_021723_2.csv"),
        "C75M3": pd.read_csv(file_loc + "\\saline_C75M3_021723_2.csv"),
        "C116AF1": pd.read_csv(file_loc + "\\saline_C116AF1_021723_2.csv"),
        "C116AF3": pd.read_csv(file_loc + "\\saline_C116AF3_021723_2.csv"),
        "C116AF5": pd.read_csv(file_loc + "\\saline_C116AF5_021723_2.csv"),
        }
    }

saline_injectiontimes = {
    "saline_1": datetime.datetime(2023,2,27,19,15),
    "saline_2": datetime.datetime(2023,2,20,17,45)
    }


p_saline_data = {}
for session in saline_data:
    print(f"Processing {session}")
    c_session = saline_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        s_mouse = f_mouse[np.logical_and(f_mouse.iloc[:,0] > saline_injectiontimes[session], 
                                         f_mouse.iloc[:,0] < (saline_injectiontimes[session] + datetime.timedelta(hours=post_hours)))]
        p_session[mouse] = s_mouse
    
    p_saline_data[session] = p_session

mk801_data = {
    "mk801_1": {
        "C75M1": pd.read_csv(file_loc + "\\mk801_C75M1_022723.csv"),
        "C75M2": pd.read_csv(file_loc + "\\mk801_C75M2_022723.csv"),
        "C75M3": pd.read_csv(file_loc + "\\mk801_C75M3_022723.csv"),
        "C116AF1": pd.read_csv(file_loc + "\\mk801_C116AF1_022723.csv"),
        "C116AF3": pd.read_csv(file_loc + "\\mk801_C116AF3_022723.csv"),
        "C116AF5": pd.read_csv(file_loc+ "\\mk801_C116AF5_022723.csv")
        },
    "mk801_2": {
        "C48M3": pd.read_csv(file_loc + "\\mk801_C48M3_021723_2.csv"),
        "C50F1": pd.read_csv(file_loc + "\\mk801_C50F1_021723_2.csv"),
        "C50F5": pd.read_csv(file_loc + "\\mk801_C50F5_021723_2.csv"),
        "C116F2": pd.read_csv(file_loc + "\\mk801_C116F2_021723_2.csv"),
        "C116F4": pd.read_csv(file_loc + "\\mk801_C116F4_021723_2.csv")
        }
    }

mk801_injectiontimes = {
    "mk801_1": datetime.datetime(2023,2,27,19,15),
    "mk801_2": datetime.datetime(2023,2,20,17,45)
    }

p_mk801_data = {}
for session in mk801_data:
    print(f"Processing {session}")
    c_session = mk801_data[session]
    p_session = {}
    for mouse in c_session:
        f_mouse = f3b.filter_data(c_session[mouse])
        s_mouse = f_mouse[np.logical_and(f_mouse.iloc[:,0] > mk801_injectiontimes[session], 
                                         f_mouse.iloc[:,0] < (mk801_injectiontimes[session] + datetime.timedelta(hours=post_hours)))]
        p_session[mouse] = s_mouse
    
    p_mk801_data[session] = p_session

#%%

"""Here we get examples of the behavior."""

c50f5_sal = p_saline_data["saline_1"]["C50F5"].iloc[:205,:]
sal_probs = c50f5_sal["Session_type"].iloc[5:] / 100
sal_choices = f3b.binned_paction(c50f5_sal, 5)

c50f5_mk801 = p_mk801_data["mk801_2"]["C50F5"].iloc[:205,:]
mk801_probs = c50f5_mk801["Session_type"].iloc[5:] / 100
mk801_choices = f3b.binned_paction(c50f5_mk801, 5)

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(np.arange(len(sal_probs)),sal_probs, c='red', linewidth=3)
ax.plot(np.arange(len(sal_probs)), sal_choices, c='darkcyan', linewidth=3)
plt.axis("off")
#plt.savefig(figure_dir + "\\example_sal.eps", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(np.arange(len(mk801_probs)),mk801_probs, c='red', linewidth=3)
ax.plot(np.arange(len(mk801_probs)), mk801_choices, c='olive', linewidth=3)
plt.axis("off")
#plt.savefig(figure_dir + "\\example_mk801.eps", bbox_inches="tight")

#%%

"""Here we get the reversals of saline and mk801"""

m_sal_pehs = {}
for session in p_saline_data:
    print(session)
    c_session = p_saline_data[session]
    c_pehs = {mouse: f3b.reversal_peh(c_session[mouse], (-10,11)).mean(axis=0) for mouse in c_session}
    c_pehs_df = pd.DataFrame(c_pehs)
    c_pehs_df = c_pehs_df.assign(Trial=np.arange(-10,11), Treatment="Sal")
    m_c_pehs = pd.melt(c_pehs_df, id_vars=["Trial", "Treatment"])
    m_sal_pehs[session] = m_c_pehs

m_sal_pehs = pd.concat(m_sal_pehs.values())

m_mk801_pehs = {}
for session in p_mk801_data:
    print(session)
    c_session = p_mk801_data[session]
    c_pehs = {mouse: f3b.reversal_peh(c_session[mouse], (-10,11)).mean(axis=0) for mouse in c_session}
    c_pehs_df = pd.DataFrame(c_pehs)
    c_pehs_df = c_pehs_df.assign(Trial=np.arange(-10,11), Treatment="MK801")
    m_c_pehs = pd.melt(c_pehs_df, id_vars=["Trial", "Treatment"])
    m_mk801_pehs[session] = m_c_pehs

m_mk801_pehs = pd.concat(m_mk801_pehs.values())

# Here we pool everything
m_all_pehs = pd.concat([m_sal_pehs, m_mk801_pehs])

fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(x="Trial", y="value", hue="Treatment", data=m_all_pehs, linewidth=4, palette=["darkcyan", "olive"], errorbar="se", legend=False)
ax.set_ylabel("P(high port)")
ax.set_xlabel("Trial from reversal")
ax.set_yticks(np.arange(0.3, 1, 0.2))
sns.despine()
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
#plt.savefig(figure_dir + "\\rev_peh_systemic_mk801.eps", bbox_inches="tight")

#Accuracy of trials prior to reversal
all_pre_pehs = m_all_pehs[m_all_pehs["Trial"] < 0]
mean_pre_pehs = all_pre_pehs.groupby(["Treatment", "variable"]).mean().reset_index()
mean_pre_pehs["Treatment"] = mean_pre_pehs["Treatment"].astype(cat_size_order)
mean_pre_pehs = mean_pre_pehs.sort_values(by="Treatment")


fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="Treatment", y="value", data=mean_pre_pehs, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="Treatment", y="value", data=mean_pre_pehs, palette=["silver", "silver"], s=10)
ax.set_ylabel("Pre-reversal accuracy")
ax.set_xlabel("")
ax.set_yticks(np.arange(0.3, 1, 0.2))
sns.despine()
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
#plt.savefig(figure_dir + "\\pre_rev_peh_systemic_mk801.eps", bbox_inches="tight")

#Do paired ttest
mk801_rev_ttest = ttest_rel(mean_pre_pehs["value"][mean_pre_pehs["Treatment"]=="Sal"], mean_pre_pehs["value"][mean_pre_pehs["Treatment"]=="MK801"], nan_policy="omit")

#%%

"""Here we get the number of pellets"""

# Count of pellets after saline injections
saline_pellets = {}
for session in p_saline_data:
    c_session = p_saline_data[session]
    c_session_pellets = {mouse: [f3b.count_pellets(c_session[mouse])] for mouse in c_session}
    saline_pellets = saline_pellets | c_session_pellets
    
saline_pellets = pd.DataFrame(saline_pellets).T

# Count of pellets after mk801 injections
mk801_pellets = {}
for session in p_mk801_data:
    c_session = p_mk801_data[session]
    c_session_pellets = {mouse: [f3b.count_pellets(c_session[mouse])] for mouse in c_session}
    mk801_pellets = mk801_pellets | c_session_pellets
    
mk801_pellets = pd.DataFrame(mk801_pellets).T

# We pool all the numbers
all_pellets = pd.concat([saline_pellets, mk801_pellets], axis=1).reset_index()
all_pellets.columns = ["Mouse", "Sal", "MK801"]
m_all_pellets = pd.melt(all_pellets, id_vars="Mouse")
m_all_pellets["variable"] = m_all_pellets["variable"].astype(cat_size_order)
m_all_pellets = m_all_pellets.sort_values(by="variable")

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_pellets, palette=[
            "darkcyan", "olive", "salmon"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_pellets, palette=["silver", "silver", "silver"], s=10)
ax.set_ylabel("Pellets")
ax.set_xlabel("")
sns.despine()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.set_ylim(0,180)
ax.set_yticks(np.arange(0,180,40))
#plt.savefig(figure_dir + "\\pellets_systemic_mk801.eps", bbox_inches="tight")

mk801_pellets_ttest = ttest_rel(m_all_pellets["value"][m_all_pellets["variable"]== "Sal"], 
                                m_all_pellets["value"][m_all_pellets["variable"] == "MK801"])

#%%

"""Here we get the number of pokes"""

saline_pokes = {}
for session in p_saline_data:
    c_session = p_saline_data[session]
    c_session_pokes = {mouse: [f3b.count_pokes(c_session[mouse])] for mouse in c_session}
    saline_pokes = saline_pokes | c_session_pokes
    
saline_pokes = pd.DataFrame(saline_pokes).T

# Count of pokes after mk801 injections
mk801_pokes = {}
for session in p_mk801_data:
    c_session = p_mk801_data[session]
    c_session_pokes = {mouse: [f3b.count_pokes(c_session[mouse])] for mouse in c_session}
    mk801_pokes = mk801_pokes | c_session_pokes
    
mk801_pokes = pd.DataFrame(mk801_pokes).T

# We pool all the numbers
all_pokes = pd.concat([saline_pokes, mk801_pokes], axis=1).reset_index()
all_pokes.columns = ["Mouse", "Sal", "MK801"]
m_all_pokes = pd.melt(all_pokes, id_vars="Mouse")
m_all_pokes["variable"] = m_all_pokes["variable"].astype(cat_size_order)
m_all_pokes = m_all_pokes.sort_values(by="variable")

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_pokes, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_pokes,
              palette=["silver", "silver"], s=10)
ax.set_ylabel("pokes")
ax.set_xlabel("")
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.set_ylim(0,340)
ax.set_yticks(np.arange(0,340,80))
sns.despine()
#plt.savefig(figure_dir + "\\pokes_systemic_mk801.eps", bbox_inches="tight")

mk801_pokes_ttest = ttest_rel(m_all_pokes["value"][m_all_pokes["variable"]=="Sal"], m_all_pokes["value"][m_all_pokes["variable"]=="MK801"], nan_policy="omit")

#%%

"""Here we get the number of pokes per pellet"""

m_all_ppp = copy.deepcopy(m_all_pokes)
m_all_ppp["value"] = m_all_pokes["value"] / m_all_pellets["value"]

# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_ppp, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_ppp, palette=["silver", "silver"], s=10)
ax.set_ylabel("Pokes/Pellet")
ax.set_xlabel("")
# ax.set_yticks([1.7,2.1,2.5])
sns.despine()
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
#plt.savefig(figure_dir + "\\ppp_systemic_mk801.eps", bbox_inches="tight")

# Do paired ttest
mk801_ppp_ttest = ttest_rel(m_all_ppp["value"][m_all_ppp["variable"]== "Sal"], m_all_ppp["value"][m_all_ppp["variable"] == "MK801"])

#%%

"""Here we look for the win-stay"""

saline_ws = {}for session in p_saline_data:
    c_session = p_saline_data[session]
    c_session_ws = {mouse: [f3b.win_stay(c_session[mouse])] for mouse in c_session}
    saline_ws = saline_ws | c_session_ws
    
saline_ws = pd.DataFrame(saline_ws).T

# Count of ws after mk801 injections
mk801_ws = {}
for session in p_mk801_data:
    c_session = p_mk801_data[session]
    c_session_ws = {mouse: [f3b.win_stay(c_session[mouse])] for mouse in c_session}
    mk801_ws = mk801_ws | c_session_ws
    
mk801_ws = pd.DataFrame(mk801_ws).T

# We pool all the numbers
all_ws = pd.concat([saline_ws, mk801_ws], axis=1).reset_index()
all_ws.columns = ["Mouse", "Sal", "MK801"]
m_all_ws = pd.melt(all_ws, id_vars="Mouse")
m_all_ws["variable"] = m_all_ws["variable"].astype(cat_size_order)
m_all_ws = m_all_ws.sort_values(by="variable")

fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_ws, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_ws, palette=["silver", "silver"], s=10, jitter=True)
ax.set_ylabel("Win-stay")
ax.set_xlabel("")
ax.set_ylim(0,1)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\ws_systemic_mk801.eps", bbox_inches="tight")

#Do paired ttest
mk801_ws_ttest = ttest_rel(m_all_ws["value"][m_all_ws["variable"]=="Sal"], m_all_ws["value"][m_all_ws["variable"]=="MK801"], nan_policy="omit")

#%%

"""Here we perform the logistic regression for wins"""

saline_pcoeffs = {}
for session in p_saline_data:
    c_session = p_saline_data[session]
    c_session_sidep = {mouse: f3b.side_prewards(c_session[mouse]) for mouse in c_session}
    c_session_preX = {mouse: f3b.create_X(c_session[mouse], c_session_sidep[mouse], 5)
                      for mouse in c_session_sidep}
    c_session_pcoeffs = {mouse: f3b.logit_regr(c_session_preX[mouse]).params for mouse in c_session_preX}
    saline_pcoeffs = saline_pcoeffs | c_session_pcoeffs
    
saline_pcoeffs = pd.DataFrame(saline_pcoeffs)
saline_pcoeffs = saline_pcoeffs.assign(Treatment="Sal", Trial=np.flip(np.arange(-5, 0, 1)))
m_saline_pcoeffs = pd.melt(saline_pcoeffs, id_vars=["Treatment", "Trial"])

# Count of ws after mk801 injections
mk801_pcoeffs = {}
for session in p_mk801_data:
    print(session)
    c_session = p_mk801_data[session]
    c_session_sidep = {mouse: f3b.side_prewards(c_session[mouse]) for mouse in c_session}
    c_session_preX = {mouse: f3b.create_X(c_session[mouse], c_session_sidep[mouse], 5)
                      for mouse in c_session_sidep}
    c_session_pcoeffs = {mouse: f3b.logit_regr(c_session_preX[mouse]).params for mouse in c_session_preX}
    mk801_pcoeffs = mk801_pcoeffs | c_session_pcoeffs
    
mk801_pcoeffs = pd.DataFrame(mk801_pcoeffs)
mk801_pcoeffs = mk801_pcoeffs.assign(Treatment="MK801", Trial=np.flip(np.arange(-5, 0, 1)))
m_mk801_pcoeffs = pd.melt(mk801_pcoeffs, id_vars=["Treatment", "Trial"])

m_all_pcoeffs = pd.concat([m_saline_pcoeffs, m_mk801_pcoeffs])

# We plot the regression coefficients
fig, ax = plt.subplots(figsize=(8, 8))
sns.pointplot(x="Trial", y="value", hue="Treatment", data=m_all_pcoeffs,
              palette=["darkcyan", "olive"], errwidth=3, errorbar="se")
for line in ax.lines:
    line.set_linewidth(4)
ax.set_xlabel("Trial in past")
ax.set_ylabel("Regr. Coefficient")
ax.set_yticks(np.arange(0,2.01,0.5))
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\plogreg_systemic_mk801.eps", bbox_inches="tight")

all_paucs = m_all_pcoeffs.groupby(by=["Treatment", "variable"]).sum().reset_index()
all_paucs["Treatment"] = all_paucs["Treatment"].astype(cat_size_order)
all_paucs = all_paucs.sort_values(by="Treatment")


# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="Treatment", y="value", data=all_paucs, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="Treatment", y="value", data=all_paucs,
              palette=["silver", "silver"], s=10, dodge=True)
ax.set_ylabel("Regr. Coeff. AUC")
ax.set_xlabel("")
# ax.set_yticks(np.arange(1,5,1))
ax.set_ylim(0,5)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\plogreg_auc_systemic_mk801.eps", bbox_inches="tight")

#Do paired ttest
mk801_pauc_ttest = ttest_rel(all_paucs["value"][all_paucs["Treatment"]=="Sal"], all_paucs["value"][all_paucs["Treatment"]=="MK801"], nan_policy="omit")



#%%

"""Here we look for the lose-shift"""

saline_ls = {}
for session in p_saline_data:
    c_session = p_saline_data[session]
    c_session_ls = {mouse: [f3b.lose_shift(c_session[mouse])] for mouse in c_session}
    saline_ls = saline_ls | c_session_ls
    
saline_ls = pd.DataFrame(saline_ls).T

# Count of ws after mk801 injections
mk801_ls = {}
for session in p_mk801_data:
    c_session = p_mk801_data[session]
    c_session_ls = {mouse: [f3b.lose_shift(c_session[mouse])] for mouse in c_session}
    mk801_ls = mk801_ls | c_session_ls
    
mk801_ls = pd.DataFrame(mk801_ls).T

# We pool all the numbers
all_ls = pd.concat([saline_ls, mk801_ls], axis=1).reset_index()
all_ls.columns = ["Mouse", "Sal", "MK801"]
m_all_ls = pd.melt(all_ls, id_vars="Mouse")
m_all_ls["variable"] = m_all_ls["variable"].astype(cat_size_order)
m_all_ls = m_all_ls.sort_values(by="variable")

fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="variable", y="value", data=m_all_ls, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="variable", y="value", data=m_all_ls, palette=["silver", "silver"], s=10, jitter=True)
ax.set_ylabel("Lose-Shift")
ax.set_xlabel("")
ax.set_yticks(np.arange(0.5, 1, 0.2))
# ax.set_ylim(0.4,1)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\ls_systemic_mk801.eps", bbox_inches="tight")

#Do paired ttest
mk801_ls_ttest = ttest_rel(m_all_ls["value"][m_all_ls["variable"]=="Sal"], m_all_ls["value"][m_all_ls["variable"]=="MK801"])

#%%

"""Here we perform the logistic regression for losses"""

saline_ncoeffs = {}
for session in p_saline_data:
    c_session = p_saline_data[session]
    c_session_siden = {mouse: f3b.side_nrewards(c_session[mouse]) for mouse in c_session}
    c_session_prenX = {mouse: f3b.create_X(c_session[mouse], c_session_siden[mouse], 5)
                      for mouse in c_session_siden}
    c_session_ncoeffs = {mouse: f3b.logit_regr(c_session_prenX[mouse]).params for mouse in c_session_prenX}
    saline_ncoeffs = saline_ncoeffs | c_session_ncoeffs
    
saline_ncoeffs = pd.DataFrame(saline_ncoeffs)
saline_ncoeffs = saline_ncoeffs.assign(Treatment="Sal", Trial=np.flip(np.arange(-5, 0, 1)))
m_saline_ncoeffs = pd.melt(saline_ncoeffs, id_vars=["Treatment", "Trial"])

# Count of ws after mk801 injections
mk801_ncoeffs = {}
for session in p_mk801_data:
    print(session)
    c_session = p_mk801_data[session]
    c_session_siden = {mouse: f3b.side_nrewards(c_session[mouse]) for mouse in c_session}
    c_session_prenX = {mouse: f3b.create_X(c_session[mouse], c_session_siden[mouse], 5)
                      for mouse in c_session_siden}
    c_session_ncoeffs = {mouse: f3b.logit_regr(c_session_prenX[mouse]).params for mouse in c_session_prenX}
    mk801_ncoeffs = mk801_ncoeffs | c_session_ncoeffs
    
mk801_ncoeffs = pd.DataFrame(mk801_ncoeffs)
mk801_ncoeffs = mk801_ncoeffs.assign(Treatment="MK801", Trial=np.flip(np.arange(-5, 0, 1)))
m_mk801_ncoeffs = pd.melt(mk801_ncoeffs, id_vars=["Treatment", "Trial"])

m_all_ncoeffs = pd.concat([m_saline_ncoeffs, m_mk801_ncoeffs])

# We plot the regression coefficients
fig, ax = plt.subplots(figsize=(8, 8))
sns.pointplot(x="Trial", y="value", hue="Treatment", data=m_all_ncoeffs,
              palette=["darkcyan", "olive"], errwidth=3, errorbar="se")
for line in ax.lines:
    line.set_linewidth(4)
ax.set_xlabel("Trial in past")
ax.set_ylabel("Regr. Coefficient")
# ax.set_yticks(np.arange(0,2.01,0.5))
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\nlogreg_systemic_mk801.eps", bbox_inches="tight")

all_naucs = m_all_ncoeffs.groupby(by=["Treatment", "variable"]).sum().reset_index()
all_naucs["Treatment"] = all_naucs["Treatment"].astype(cat_size_order)
all_naucs = all_naucs.sort_values(by="Treatment")


# Plotting
fig, ax = plt.subplots(figsize=(5, 8))
sns.boxplot(x="Treatment", y="value", data=all_naucs, palette=[
            "darkcyan", "olive"], boxprops={"linewidth": 2.5}, whiskerprops={"linewidth": 2.5})
sns.stripplot(x="Treatment", y="value", data=all_naucs,
              palette=["silver", "silver"], s=10, dodge=True)
ax.set_ylabel("Regr. Coeff. AUC")
ax.set_xlabel("")
# ax.set_yticks(np.arange(1,5,1))
ax.set_ylim(-1,5)
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
sns.despine()
#plt.savefig(figure_dir + "\\nlogreg_auc_systemic_mk801.eps", bbox_inches="tight")

#Do paired ttest
mk801_nauc_ttest = ttest_rel(all_naucs["value"][all_naucs["Treatment"]=="Sal"], all_naucs["value"][all_naucs["Treatment"]=="MK801"])
