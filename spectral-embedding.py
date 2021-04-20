# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:27:02 2021

@author: Nate
"""


#%% IO
from rna_analysis_classes import data_import, distribution_data, sample_list, all_sites_eps
import random
import pandas as pd
import ranVar as rV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.manifold import SpectralEmbedding, MDS
from sklearn.cluster import KMeans, SpectralClustering
import seqlogo as seql
import matplotlib.patches as mpatches
#%% Variables #################################
path = "D:/projects/project-7_RNA/RealRNAData/csvs/"
file = "YWFLIVM-merged_baseline-normalized.csv"
keep = ['seq', 'Fam2.1', 'Fam2.2', 'Fam1A.1', 'Fam1B.1', 'Fam3.1'] + ['r_'+i for i in ['BIO', 'BLO', 'BMO', 'BVO', 'BFO', 'BWO', 'BYO']]
#%% Import ##############################
domain = path+file
combined_data = data_import(domain)
dist_dat = distribution_data()
dist_dat.import_data(combined_data, keep)
#%% Distribution Init ###################
family_name = 'Fam1B.1'
token = family_name
samples = [rV.sample({'seq' : i}) for i in dist_dat.families[family_name][0]['seq']]
substrates = ['r_'+i for i in ['BIO', 'BLO', 'BMO', 'BVO', 'BFO', 'BWO', 'BYO']]
binary_substrates = ['bin_'+i for i in ['BIO', 'BLO', 'BMO', 'BVO', 'BFO', 'BWO', 'BYO']]
print("\n Initializing Data")
#%% Alphabet Creation ###########################
alphabet = {'A' : 0, 'T' : 1, 'G' : 2, 'C' : 3}
dealphabet = {0 : 'A', 1 : 'T', 2 : 'G', 3 : 'C'}
double_alphabet = {}
count = 0
doublealph_ticks = []
#################################################
for i in ['A', 'T', 'G', 'C']:
    for j in ['A', 'T', 'G', 'C']:
        doublealph_ticks.append(i+j)
        double_alphabet.update({i+j : count})
        count += 1
## Assign Residue Labels to Samples #############        
for i in samples:
    for site in range(21):
        i.add_entry({'single'+str(site) : alphabet[i.data['seq'][site]]})
    for site1 in range(21):
        for site2 in range(21):
            i.add_entry({'double' + str(site1)+'x'+str(site2) : double_alphabet[i.data['seq'][site1]+i.data['seq'][site2]]})
#%% Sample List Generation ###########################
samp_list = sample_list(samples, dist_dat, substrates)
print("\n Sample list initialized")
#%% Make Data  Set
n_neighbors = 4
X = np.asarray([[i.data['r_BYO'], i.data['r_BWO'], i.data['r_BFO'], i.data['r_BIO'], i.data['r_BLO'], i.data['r_BMO'], i.data['r_BVO']] for i in samp_list.samples])
se_embedding = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
mds_embedding = MDS(n_components=2, max_iter=100, n_init=1)
se_X_transformed = se_embedding.fit_transform(X)
mds_X_transformed = mds_embedding.fit_transform(X)
#%% Clustering
n_clusters = 6
random_state = 42
k_cluster = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
sc_cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=random_state, affinity='nearest_neighbors', n_neighbors=n_neighbors).fit(X)
for i in range(len(samp_list.samples)):
    samp_list.samples[i].data['k_cluster'] = k_cluster.labels_[i]
    samp_list.samples[i].data['sc_cluster'] = sc_cluster.labels_[i]
    
k_clusters = {}
for i in range(n_clusters):
    k_clusters[i] = []
    
for i in samp_list.samples:
    k_clusters[i.data['k_cluster']].append(i)

cluster_lists = {}
for i in range(n_clusters):
    cluster_lists[i] = sample_list(k_clusters[i], dist_dat, substrates)
    cluster_lists[i].percentage_sequence()
    seql.seqlogo(cluster_lists[i].cpm, format='png')
#%%    
activities = {}
for n in range(n_clusters):
    activities[n] = {}
    for substrate in substrates:
        activities[n][substrate] = []
        for j in cluster_lists[n].samples:
            activities[n][substrate].append(j.data[substrate])
#%% Kernel Density Estimation
def kde(act):
    act = np.asarray(act).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=2.5).fit(act)
    x_d = np.linspace(min(act)[0], max(act)[0], 3100)
    logprob = kde.score_samples(x_d[:,None])
    ############################
    fig, ax = plt.subplots()
    ax.fill_between(x_d, np.exp(logprob), alpha=0.5, color='tab:red')
    ax.plot(act, np.full_like(act, -0.0), '|k', markeredgewidth=1.0, color='tab:red')
    ############################
    ax.set_title('KDE of '+str(family_name)+' Activities')
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('Activity, r')
    return fig, ax

colors = {'r_BIO' : 'tab:red', 'r_BLO' : 'tab:orange', 'r_BMO':'yellow', 'r_BVO' : 'tab:green', 'r_BFO' : 'tab:blue', 'r_BWO' : 'tab:cyan', 'r_BYO' : 'tab:purple'}
substra = ['r_BYO', 'r_BFO', 'r_BWO']
def clus_vis(activities, n, substrates, colors):
    fig, ax = plt.subplots()
    for subs in substrates:
        act = np.asarray(activities[n][subs]).reshape(-1,1)
        kde = KernelDensity(kernel='gaussian', bandwidth=2.5).fit(act)
        x_d = np.linspace(min(act)[0], max(act)[0], 3100)
        logprob = kde.score_samples(x_d[:,None])
        ##################
        ax.fill_between(x_d, np.exp(logprob), alpha=0.5, color=colors[subs])
        ax.plot(act, np.full_like(act, -0.0), '|k', markeredgewidth=1.0, color=colors[subs])
        ax.set_title('KDE of Fam1B.1 Activities: Cluster '+str(n))
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Activity, r')
        ##################
        bio_patch = mpatches.Patch(color='tab:red', label='BIO')
        blo_patch = mpatches.Patch(color='tab:orange', label='BLO')
        bmo_patch = mpatches.Patch(color='yellow', label='BMO')
        bvo_patch = mpatches.Patch(color='tab:green', label='BVO')
        bfo_patch = mpatches.Patch(color='tab:blue', label='BFO')
        bwo_patch = mpatches.Patch(color='tab:cyan', label='BWO')
        byo_patch = mpatches.Patch(color='tab:purple', label='BYO')
        ###################
        ax.legend(handles=[bio_patch, blo_patch, bmo_patch, bvo_patch, bfo_patch, bwo_patch, byo_patch])
    return fig, ax
        

#%%
fig, axs = plt.subplots(2,2)
#######
axs[0][0].set_title("Laplacian Eigenmap")
axs[0][0].scatter(se_X_transformed[:,0], se_X_transformed[:,1], c=k_cluster.labels_)
axs[0][0].set_ylabel('Coordinate A')
#######
axs[0][1].set_title("Multidimensional Scaling")
axs[0][1].yaxis.set_label_position('right')
axs[0][1].yaxis.tick_right()
axs[0][1].set_ylabel('k-Means Clusters')
axs[0][1].scatter(mds_X_transformed[:,0], mds_X_transformed[:,1], c=k_cluster.labels_)
#######
axs[1][0].scatter(se_X_transformed[:,0], se_X_transformed[:,1], c=sc_cluster.labels_)
axs[1][0].set_ylabel('Coordinate A')
axs[1][0].set_xlabel('Coordinate B')
#######
axs[1][1].yaxis.set_label_position('right')
axs[1][1].yaxis.tick_right()
axs[1][1].set_ylabel('S.E. Clusters')
axs[1][1].set_xlabel('Coordinate B')
axs[1][1].scatter(mds_X_transformed[:,0], mds_X_transformed[:,1], c=sc_cluster.labels_)


