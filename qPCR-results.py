# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:26:56 2021

@author: Nate
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = "D:/projects/project-7_RNA/RealRNAData/csvs/"
file_name = "fitting_results.csv"
df = pd.read_csv(path+file_name)

#%%
fig, ax = plt.subplots(1,1, figsize=(9,9))
Xs, Ys, yerrs, labels = [],[],[], []
XWTs, YWTs, yWTerrs, WTlabels = [], [], [],[]
for i in range(5):
    if i >= 1 and i < 4: 
        Xs.append(i)
        Ys.append(df.iloc[i]['kA_50'])
        yerrs.append([df.iloc[i]['kA_2.5'], df.iloc[i]['kA_97.5']])
        labels.append(df.iloc[i]['Row'])
    elif i ==0 or i ==4:
        XWTs.append(i)
        YWTs.append(df.iloc[i]['kA_50'])
        yWTerrs.append([df.iloc[i]['kA_2.5'], df.iloc[i]['kA_97.5']])
        labels.append(df.iloc[i]['Row'])
    
ax.errorbar(Xs, Ys, np.asarray(yerrs).transpose(), fmt='o', color='red')
ax.errorbar(XWTs, YWTs, np.asarray(yWTerrs).transpose(), fmt='o', color='purple')
ax.set_xticks(range(5))
ax.set_xticklabels(labels, size=14)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)

ax.set_xlabel('Sequence', size = 20)
ax.set_ylabel('Log[kA] (95% Confidence Interval)', size=20)

# ax.set_yscale('log')
ax.set_title("Figure 7: qPCR Activity Measurements", size=24, weight = 'bold')
    