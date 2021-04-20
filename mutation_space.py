# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:42:01 2021

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
#%% Mutation Space Class
class mutation:
    def __init__(self, WT, M, data):
        self.WT = WT
        self.M = M
        self.data = {}.update(data)
        
class mutations:
    
    def __init__(self, samp_list, substrates):
        self.samp_list = samp_list
        self.muts = {}
        for seq_1 in range(len(self.samp_list.samples)):
            for seq_2 in range(seq_1, len(self.samp_list.samples)):
                output = self.create_mutation(self.samp_list.samples[seq_1], self.samp_list.samples[seq_2], substrates)
                if output[-1] not in self.muts.keys():
                    self.muts[output[-1]] = []
                self.muts[output[-1]].append(output)    
        
    def create_mutation(self, seq_1, seq_2, substrates, verbose=False):
        deltas = {}
        for key in substrates:
            a = seq_1.data[key]
            b = seq_2.data[key]
            deltas[key] = a - b
            if verbose==True:
                print((a,b,deltas[key]))
        identity = []
        loci = []
        order = 0
        for n in range(len(seq_1.data['seq'])):
            if seq_1.data['seq'][n] != seq_2.data['seq'][n]:
                order += 1
                identity.append(seq_1.data['seq'][n] + str(n)+ seq_2.data['seq'][n])
                loci.append(n)
        return [identity, loci, deltas, order]
    
class seedscape:
    
    def __init__(self, sequence, samp_list, substrates):
        ##################################################################
        self.reference = sequence
        self.samp_list = samp_list
        self.muts = {}
        self.substrates = substrates
        self.single_muts = {}
        ##################################################################
        for seq in samp_list.samples:
            output = self.create_mutation(self.reference, seq, substrates)
            if output[-1] not in self.muts.keys():
                self.muts[output[-1]] = []
            self.muts[output[-1]].append(output)
            if output[-1] == 1:
                self.single_muts[output[0][0]] = output[2]
    
    def create_mutation(self, seq_1, seq_2, substrates, verbose=False):
        deltas = {}
        for key in substrates:
            a = seq_1.data[key]
            b = seq_2.data[key]
            deltas[key] = a - b
            if verbose==True:
                print((a,b,deltas[key]))
        identity = []
        loci = []
        order = 0
        for n in range(len(seq_1.data['seq'])):
            if seq_1.data['seq'][n] != seq_2.data['seq'][n]:
                order += 1
                identity.append(seq_1.data['seq'][n] + str(n)+ seq_2.data['seq'][n])
                loci.append(n)
        return [identity, loci, deltas, order]
    
    def deconstruct(self, mutation, substrates):
        output = []
        for code in mutation:
            output.append(self.single_muts[code])
        adds = {}    
        for sub in substrates:
            for mut in output:
                if sub not in adds.keys():
                    adds[sub] = 0
                adds[sub] += mut[sub]
        
        return output, adds    
    
    def full_deconstruction(self, substrates):
        deconstructions = {}
        for doub in self.muts[2]:
            token = ''
            for c in doub[0]:
                token += c
            deconstructions[token] = [self.deconstruct(doub[0], substrates), doub[1], doub[2]]
        self.deconstructions = deconstructions
        
    def find_decon(self, sites):
        output = []
        for key in self.deconstructions.keys():
            if self.deconstructions[key][1] == sites:
                output.append([self.deconstructions[key][0][-1]['r_BYO'], self.deconstructions[key][-1]['r_BYO']])
        return np.asarray(output)
               
#%% Seeds #############################
### pk1A S-1A.1-a CTACTTCAAACAATCGGTCTG
### pk1B S-1B.1-a CCACACTTCAAGCAATCGGTC
### pk2  S-2.1-a  ATTACCCTGGTCATCGAGTGA
### pk2  S-2.2-a  ATTCACCTAGGTCATCGGGTG
### pk3  S-3.1    AAGTTTGCTAATAGTCGCAAG
#######################################

def find_seed(sequence):
    for i in samp_list.samples:
        if i.data['seq'] == sequence:
            return i

ref = find_seed('CCACACTTCAAGCAATCGGTC')
seed_muts = seedscape(ref, samp_list, substrates)
seed_muts.full_deconstruction(['r_BYO'])
#%%
def differences(sites):
    output = seed_muts.find_decon(sites)[:,0] - seed_muts.find_decon(sites)[:,1]
    return output, np.average(output), np.std(output)

site_pairs = []
for i in range(21):
    for j in range(i, 21):
        if i != j:
            site_pairs.append([i,j])

diffs = {}
for pair in site_pairs:
    diffs[tuple(pair)] = differences(pair)
    
avgs = {}
for i in diffs.keys():
    avgs[i] = diffs[i][-2]
    
stds = {}
for i in diffs.keys():
    stds[i] = diffs[i][-1]

avgmatdata = np.zeros((21, 21))
for i in range(21):
    for j in range(21):
        try:
            avgmatdata[j][i] = avgs[(i,j)] 
        except:
            avgmatdata[j][i] = -1
            
stdmatdata = np.zeros((21, 21))
for i in range(21):
    for j in range(21):
        try:
            stdmatdata[j][i] = stds[(i,j)] 
        except:
            stdmatdata[j][i] = -1
            
def create_image(avgs, stds):
    value = -1
    masked_array = np.ma.masked_where(avgs == value, avgs)
    cmap = matplotlib.cm.winter
    cmap.set_bad(color='black')
    
    plt.imshow(masked_array, cmap=cmap)
    
            
#%%    
muts = mutations(samp_list, substrates)
#%%
def searchmut(mutation, substrate):
    out = [i for i in muts.muts[1] if i[0] == [mutation]]
    acts = [i[2][substrate] for i in out]
    return out, acts, np.std(acts)

test1 = [i for i in muts.muts[1] if 11 in i[1]]
test2 = [i for i in muts.muts[2] if (11 in i[1]) and (2 in i[1])]


#%%
deltas_1 = []
for substrate in substrates:
    deltas_1.append([i[2][substrate] for i in muts.muts[1]])

deltas_1 = np.asarray(deltas_1).transpose()
#%% Manifold Embedding
n_neighbors=25
n_clusters = 27
random_state = 42
#####
print("Embedding Manifolds")
se_embedding = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
mds_embedding = MDS(n_components=2, max_iter=100, n_init=1)
se_X_transformed = se_embedding.fit_transform(deltas_1)
mds_X_transformed = mds_embedding.fit_transform(deltas_1)
#%% Clustering
print("Clustering")
k_cluster = KMeans(n_clusters=n_clusters, random_state=random_state).fit(deltas_1)
sc_cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=random_state, affinity='nearest_neighbors', n_neighbors=n_neighbors).fit(deltas_1)
for i in range(len(muts.muts[1])):
    muts.muts[1][i][2]['k_cluster'] = k_cluster.labels_[i]
    muts.muts[1][i][2]['sc_cluster'] = sc_cluster.labels_[i]

#%%    
k_clusters = {}
for i in range(n_clusters):
    k_clusters[i] = []
    
for i in muts.muts[1]:
    k_clusters[i[2]['sc_cluster']].append(i)

# cluster_lists = {}
# for i in range(n_clusters):
#     cluster_lists[i] = sample_list(k_clusters[i], dist_dat, substrates)

activities = {}
for n in range(n_clusters):
    activities[n] = {}
    for substrate in substrates:
        activities[n][substrate] = []
        for j in k_clusters[n]:
            activities[n][substrate].append(j[2][substrate])
print("Clusters Complete")
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
    ax.set_title('KDE of Fam1A.1 Activities')
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('Activity, r')
    return fig, ax

def kde_loci(cluster):
    act = [i[1][0] for i in k_clusters[cluster]]
    act = np.asarray(act).reshape(-1,1)
    kde = KernelDensity(kernel='tophat', bandwidth=0.5).fit(act)
    x_d = np.linspace(min(act)[0], max(act)[0], 3100)
    logprob = kde.score_samples(x_d[:,None])
    ############################
    fig, ax = plt.subplots()
    ax.fill_between(x_d, np.exp(logprob), alpha=0.5, color='navy')
    ax.plot(act, np.full_like(act, -0.0), '|k', markeredgewidth=1.0, color='indigo')
    ############################
    ax.set_xticks(range(21))
    ax.set_xticklabels(range(27,48))
    ############################
    ax.set_title('KDE of Cluster Mutation Loci: Cluster '+str(cluster))
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('Residue Number')
    return fig, ax
#%%


colors = {'r_BIO' : 'red', 'r_BLO' : 'tab:orange', 'r_BMO':'yellow', 'r_BVO' : 'lime', 'r_BFO' : 'turquoise', 'r_BWO' : 'darkblue', 'r_BYO' : 'purple'}
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
        ax.set_title('KDE of '+str(family_name)+'Mutation Activity Change: Cluster '+str(n))
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Activity, r')
        ##################
        bio_patch = mpatches.Patch(color='red', label='BIO')
        blo_patch = mpatches.Patch(color='tab:orange', label='BLO')
        bmo_patch = mpatches.Patch(color='yellow', label='BMO')
        bvo_patch = mpatches.Patch(color='lime', label='BVO')
        bfo_patch = mpatches.Patch(color='turquoise', label='BFO')
        bwo_patch = mpatches.Patch(color='darkblue', label='BWO')
        byo_patch = mpatches.Patch(color='purple', label='BYO')
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
#%%

    
            
            
