# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:07:47 2021

@author: Nate
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:44:17 2021

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
import matplotlib
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
family_name = 'Fam1A.1'
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
#%% Class Assignments ################################
#### r_BYO of seed 2.1B = 272.56562902655156
samp_list.double_class('r_BYO', [4.0, 329.5446518108852])
random.shuffle(samp_list.samples)
print("\n Sample list initialized")
#####################################
#%% Calculation of Single Site RE ###
#####################################
seqVar = rV.ranVar(samp_list.samples)
H, Hexp, X = [], [], []
terms, H_dict, terms_exp, model_terms = {}, {}, {}, {}
#####################################
for n in range(21): 
    ##############
    X.append(n)
    ##############
    seqVar.hist2d('essential', 'single'+str(n), bins=(2,4))
    H.append(seqVar.rent)
    H_dict.update({n : seqVar.rent})
    terms.update({n : seqVar.re_component_matrix})
    ##############
    seqVar.hist2d('expert', 'single'+str(n), bins=(2,4))
    Hexp.append(seqVar.rent)
    terms_exp.update({n : seqVar.re_component_matrix})
#####################################
for i in terms.keys():
    model_terms.update({i : {} })
    for j in range(terms[i].shape[0]):
        model_terms[i].update({j:{}})
        for k in range(terms[i].shape[1]):
            model_terms[i][j].update({dealphabet[k] : terms[i][j][k]})
#%% Epistasis Calculation ###########
eps_data = all_sites_eps(seqVar, 'expert')
#%% Visualizations ##################
### matrix should be eps_data[0] or eps_data[1] for epistasis or associated surprisal sum, respectively
### title should be a string that is the title of the graphic
def gen_mat_graphic(matrix, title):
    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(matrix, cmap='winter')
    ax.set_title(title, size = 20)
    ax.set_xlabel("Site 1", size=16)
    ax.set_ylabel("Site 2", size=16)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    ax.set_xticklabels(range(27,48))
    ax.set_yticklabels(range(27,48))
    return fig, ax
#%% Double Site Term Analysis Graphic
### Target is the (M,N) site pair for which the surprisal events are to be analyzed. Terms is the dictionary associated
### with eps_data[2] that is passed from running all_sites_eps
def dsite_term(target, terms):
    target = target
    fix, ax = plt.subplots(figsize=(15,15))
    ax.imshow(terms[target], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target]):
         ax.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax.set_title("Epistasis Terms Matrix: Site ({site1},{site2})".format(site1=target[0]+27, site2=target[1]+27), y= 1.0, size=20)
    ax.set_xlabel("Genotype Class", size=16)
    ax.set_yticks(range(2))
    ax.set_yticklabels(['Inactivity', 'Activity'], size=14)
    ax.set_xticks(range(16))
    ax.set_xticklabels(doublealph_ticks, size=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Activity Class", size=16)
    return fig, ax
#%% Single Site RE Graphic ###########
def single_site_graphic(X, Hexp):
    fig, ax = plt.subplots(figsize=(7,7))
    #########################
    ax.plot(X, Hexp, color = 'tab:red')
    ax.set_ylabel("Single Site Mutual Information, bits", size=16)
    ax.set_xlabel("Site", size=16)
    ax.set_xticks(range(21))
    ax.set_xticklabels(range(27,48), size=14)
    #########################
    ax.set_title('Single Site Genotype Variable Analysis', size=20)
    return fig, ax
#%% Figure 1
def create_figure_5(matrix, X, Hexp, target, terms):
    grid = plt.GridSpec(2,2, wspace=0.2, hspace=0.1)
    fig = plt.figure(figsize = (13,8))
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[0,1])
    ax3 = fig.add_subplot(grid[1,:2])
    fig.suptitle('Figure 5: S1A Epistasis & Mutual Information', size=24, weight='bold')
    #########################
    value = 0.0
    masked_array = np.ma.masked_where(matrix == value, matrix)
    cmap_avg = matplotlib.cm.cool
    cmap_avg.set_bad(color='black')
    #########################
    pos = ax1.imshow(masked_array, cmap=cmap_avg, interpolation='none')
    fig.colorbar(pos, ax = ax1, shrink=0.75)
    ax1.set_title('Epistasic Divergence, $\epsilon$', size = 20)
    ax1.set_xlabel("Site M", size=16)
    ax1.set_ylabel("Site N", size=16)
    ax1.set_xticks(range(21))
    ax1.set_yticks(range(21))
    ax1.set_xticklabels(range(27,48))
    ax1.set_yticklabels(range(27,48))
    for label in ax1.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax1.text(-0.3, 1.0, 'A', transform=ax1.transAxes, size=20, weight='bold')
    ax1.text(1.25, 0.35, r'$\epsilon$, bits', transform=ax1.transAxes, size=20, weight='bold', rotation='vertical')
    #########################
    ax2.plot(X, Hexp, color = 'tab:green')
    ax2.set_title("Mutual Information")
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_ylabel("Mutual Information, bits", size=16)
    ax2.set_xlabel("Site", size=16)
    ax2.set_xticks(range(21))
    ax2.set_xticklabels(range(27,48), size=12)
    for label in ax2.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    #########################
    ax2.set_title('Mutual Information', size=20)
    ax2.text(1.55, 1.0, 'B', transform=ax1.transAxes, size=20, weight='bold')
    #########################
    ax1.text(-0.3, -0.3, 'C', transform=ax1.transAxes, size=20, weight='bold')
    ax3.imshow(terms[target], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target]):
         ax3.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax3.set_title("Epistasis Terms Matrix: Site ({site1},{site2})".format(site1=target[0]+27, site2=target[1]+27), y= 1.0, size=20)
    ax3.set_xlabel("Genotype Class", size=16)
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['Inactivity', 'Activity'], size=14)
    ax3.set_xticks(range(16))
    ax3.set_xticklabels(doublealph_ticks, size=14)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax3.set_ylabel("Activity Class", size=16)
    
create_figure_5(eps_data[0], X, Hexp, (2,0), eps_data[2])
#%% Kernel Density Estimation
activities = np.asarray([i.data['r_BYO'] for i in samp_list.samples if ((i.data['r_BYO'] < 30.0) and (i.data['r_BYO'] > 0.0))]).reshape(-1,1)
kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(activities)
x_d = np.linspace(0.0, 30.0, 3100)
logprob = kde.score_samples(x_d[:,None])
############################
plt.fill_between(x_d, np.exp(logprob), alpha=1.0)
plt.plot(activities, np.full_like(activities, -0.0), '|k', markeredgewidth=1, color='red')
plt.ylim(-0.0, 0.05)
plt.xlim(5.0, 30.0)
############################
plt.title('KDE of Fam1A.1 Activities')
plt.ylabel('Probability Density')
plt.xlabel('Activity, r')
