# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:44:17 2021

@author: Nate
"""

#%% IO
from rna_analysis_classes import data_import, distribution_data, sample_list, all_sites_eps, seedscape
import random
import pandas as pd
import ranVar as rV
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#%% Variables #################################
path = "D:/projects/project-7_RNA/RealRNAData/csvs/"
file = "YWFLIVM-merged_baseline-normalized.csv"
keep = ['seq', 'Fam2.1', 'Fam2.2', 'Fam1A.1', 'Fam1B.1', 'Fam3.1'] + ['r_'+i for i in ['BIO', 'BLO', 'BMO', 'BVO', 'BFO', 'BWO', 'BYO']]
keep_trip = ['seq', 'Fam2.1', 'Fam2.2', 'Fam1A.1', 'Fam1B.1', 'Fam3.1', 'bs_kA_2.5%_BYO', 'bs_kA_50%_BYO', 'bs_kA_97.5%_BYO']
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
seed_seq = 'CCACACTTCAAGCAATCGGTC'
print("\n Sample list initialized")
#%% Class Assignments ################################
#### r_BYO of seed 2.1B CCACACTTCAAGCAATCGGTC = 272.56562902655156
samp_list.double_class('r_BYO', [4.0, 272.56562902655156])
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
ess_data = all_sites_eps(seqVar, 'essential')
#####################################
#%%
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
#%% ######################################################            
def create_figure_1(avgs, stds, exp, ess):
    ### Masking
    value_avg = -1
    masked_array_avg = np.ma.masked_where(avgs == value_avg, avgs)
    cmap_avg = matplotlib.cm.YlOrRd
    cmap_avg.set_bad(color='black')
    ### 
    value_std = -1
    masked_array_std = np.ma.masked_where(stds == value_std, stds)
    cmap_std = matplotlib.cm.YlOrRd
    cmap_std.set_bad(color='black')
    ###
    value_exp = 0.0
    masked_array_exp = np.ma.masked_where(exp == value_exp, exp)
    cmap_exp = matplotlib.cm.cool
    cmap_exp.set_bad(color='black')
    ###
    value_ess = 0.0
    masked_array_ess = np.ma.masked_where(ess == value_ess, ess)
    cmap_ess = matplotlib.cm.cool
    cmap_ess.set_bad(color='black')
    ###
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    fig.suptitle("Figure 1: Comparison of E.D. & Additive Deviation", size=24)
    ###
    a00 = axs[0][0].imshow(masked_array_avg, cmap=cmap_avg)
    axs[0][0].set_title(r'Nonadditive $\mu$', size = 17)    
    axs[0][0].set_ylabel("Site N", size=16)    
    axs[0][0].set_yticks(range(21))
    axs[0][0].set_xticklabels(range(27,48))
    axs[0][0].set_yticklabels(range(27,48), size=12)
    for label in axs[0][0].yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    axs[0][0].text(0.75, 0.75, 'A', transform=axs[0][0].transAxes, size=24, weight='bold', c='white')
    axs[0][0].axes.get_xaxis().set_visible(False)
    axs[0][0].text(1.25, 0.2, r'$\mu$, mol/min', transform=axs[0][0].transAxes, size=20, weight='bold', rotation='vertical')

    fig.colorbar(a00, ax = axs[0][0], shrink=0.8)
    ###
    axs[0][1].set_title('Nonadditive $\sigma$', size = 17)
    a01 = axs[0][1].imshow(masked_array_std, cmap=cmap_std)
    axs[0][1].text(2.25, 0.75, 'B', transform=axs[0][0].transAxes, size=24, weight='bold', c='white')
    axs[0][1].axes.get_xaxis().set_visible(False)
    axs[0][1].axes.get_yaxis().set_visible(False)
    axs[0][1].text(1.25, 0.2, r'$\sigma$, mol/min', transform=axs[0][1].transAxes, size=20, weight='bold', rotation='vertical')
    
    fig.colorbar(a01, ax = axs[0][1], shrink=0.8)
    ###
    a10 = axs[1][0].imshow(masked_array_exp, cmap=cmap_exp)
    axs[1][0].set_title(r'Epistatic Divergence, $\theta$=$a_{seed}$', size = 17)
    axs[1][0].set_xlabel("Site M", size=16)
    
    axs[1][0].set_xticks(range(21))
    axs[1][0].set_xticklabels(range(27,48), size=12)
    for label in axs[1][0].xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    axs[1][0].set_yticklabels(range(27,48), size=12)
    axs[1][0].set_yticks(range(21))
    for label in axs[1][0].yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    
    axs[1][0].text(0.75, -0.55, 'C', transform=axs[0][0].transAxes, size=24, weight='bold', c='white')
    axs[1][0].set_ylabel("Site N", size=16)
    axs[1][0].text(1.25, 0.35, r'$\epsilon$, bits', transform=axs[1][0].transAxes, size=20, weight='bold', rotation='vertical')
    
    fig.colorbar(a10, ax = axs[1][0], shrink=0.8)
    ###
    axs[1][1].set_title(r'Epistatic Divergence, $\theta$=$a_{active}$', size=17)
    a11 = axs[1][1].imshow(masked_array_ess, cmap=cmap_ess)
    axs[1][1].set_xlabel("Site M", size=16)
    axs[1][1].set_xticks(range(21))
    axs[1][1].set_xticklabels(range(27,48), size=12)
    for label in axs[1][1].xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    axs[1][1].text(2.25, -0.55, 'D', transform=axs[0][0].transAxes, size=24, weight='bold', c='white')
    axs[1][1].axes.get_yaxis().set_visible(False)
    axs[0][0].text(1.25, 0.35, r'$\epsilon$, bits', transform=axs[1][1].transAxes, size=20, weight='bold', rotation='vertical')
    
    fig.colorbar(a11, ax = axs[1][1], shrink=0.8)
    
create_figure_1(avgmatdata, stdmatdata, eps_data[0], ess_data[0])
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
    fig, ax = plt.subplots(figsize=(15,15))
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
def create_figure_2(matrix, X, Hexp):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,6.5))
    fig.suptitle('Figure 2: Epistasis & Mutual Information', size=24, weight='bold')
    #########################
    value = 0.0
    masked_array = np.ma.masked_where(matrix == value, matrix)
    cmap_avg = matplotlib.cm.cool
    cmap_avg.set_bad(color='black')
    #########################
    pos = ax1.imshow(masked_array, cmap=cmap_avg, interpolation='none')
    fig.colorbar(pos, ax = ax1, shrink=0.75)
    ax1.set_title(r'Epistasic Divergence, $\epsilon$', size = 20)
    ax1.set_xlabel("Site M", size=16)
    ax1.set_ylabel("Site N", size=16)
    ax1.set_xticks(range(21))
    ax1.set_yticks(range(21))
    ax1.set_xticklabels(range(27,48))
    ax1.set_yticklabels(range(27,48))
    ax1.text(-0.1, 1.0, 'A', transform=ax1.transAxes, size=20, weight='bold')
    ax1.text(1.2, 0.35, r'$\epsilon$, bits', transform=ax1.transAxes, size=20, weight='bold', rotation='vertical')
    #########################
    ax2.plot(X, Hexp, color = 'tab:green')
    ax2.set_title("Mutual Information")
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_ylabel("Single Site Mutual Information, bits", size=16)
    ax2.set_xlabel("Site", size=16)
    ax2.set_xticks(range(21))
    ax2.set_xticklabels(range(27,48), size=12)
    #########################
    ax2.set_title('Mutual Information', size=20)
    ax2.text(1.55, 1.0, 'B', transform=ax1.transAxes, size=20, weight='bold')
    
create_figure_2(eps_data[0], X, Hexp)
#%%
def create_figure_3(target1, target2, target3, terms):
    ##############################
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(15,7))
    fig.suptitle('Figure 3: Epistasis Matrix Contributions', size=24, weight='bold')
    ax1.imshow(terms[target1], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target1]):
         ax1.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax1.set_title("Sites ({site1},{site2})".format(site1=target1[0]+27, site2=target1[1]+27), y= 1.0, size=20)
    ax1.set_yticks(range(2))
    ax1.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax1.set_xticks([])
    ax1.set_xticklabels(doublealph_ticks, size=14)
    ax1.text(-0.03, 1.0, 'A', transform=ax1.transAxes, size=20, weight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ##############################
    ax2.imshow(terms[target2], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target2]):
         ax2.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax2.set_xticklabels(doublealph_ticks, size=14)
    ax2.set_title("Sites ({site1},{site2})".format(site1=target2[0]+27, site2=target2[1]+27), y= 1.0, size=20)
    ax2.set_yticks(range(2))
    ax2.set_xticks([])
    ax2.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax2.text(-0.03, -0.3, 'B', transform=ax1.transAxes, size=20, weight='bold')
    ax2.set_ylabel("Activity Class", size=16)
    ##############################
    ax3.imshow(terms[target3], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target3]):
         ax3.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax3.set_title("Sites ({site1},{site2})".format(site1=target3[0]+27, site2=target3[1]+27), y= 1.0, size=20)
    ax3.set_xlabel("Genotype Class", size=16)
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax3.set_xticks(range(16))
    ax3.set_xticklabels(doublealph_ticks, size=14)
    ax2.text(-0.03, -1.6, 'C', transform=ax1.transAxes, size=20, weight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
create_figure_3((11,4), (11,2), (4,2), eps_data[2])
#%%
def create_figure_3(target1, target2, target3, terms):
    ##############################
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(15,7))
    fig.suptitle('Figure 3: Epistasis Matrix Contributions', size=24, weight='bold')
    ax1.imshow(terms[target1], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target1]):
         ax1.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax1.set_title("Sites ({site1},{site2})".format(site1=target1[0]+27, site2=target1[1]+27), y= 1.0, size=20)
    ax1.set_yticks(range(2))
    ax1.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax1.set_xticks([])
    ax1.set_xticklabels(doublealph_ticks, size=14)
    ax1.text(-0.03, 1.0, 'A', transform=ax1.transAxes, size=20, weight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ##############################
    ax2.imshow(terms[target2], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target2]):
         ax2.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax2.set_xticklabels(doublealph_ticks, size=14)
    ax2.set_title("Sites ({site1},{site2})".format(site1=target2[0]+27, site2=target2[1]+27), y= 1.0, size=20)
    ax2.set_yticks(range(2))
    ax2.set_xticks([])
    ax2.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax2.text(-0.03, -0.3, 'B', transform=ax1.transAxes, size=20, weight='bold')
    ax2.set_ylabel("Activity Class", size=16)
    ##############################
    ax3.imshow(terms[target3], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target3]):
         ax3.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax3.set_title("Sites ({site1},{site2})".format(site1=target3[0]+27, site2=target3[1]+27), y= 1.0, size=20)
    ax3.set_xlabel("Genotype Class", size=16)
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax3.set_xticks(range(16))
    ax3.set_xticklabels(doublealph_ticks, size=14)
    ax2.text(-0.03, -1.6, 'C', transform=ax1.transAxes, size=20, weight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#%% Supplementary Figure
def create_suppfig_1(target1, target2, target3, terms):
    ##############################
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(15,7))
    fig.suptitle('Figure S1: Additional Epistasis Matrix Contributions', size=24, weight='bold')
    ax1.imshow(terms[target1], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target1]):
         ax1.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax1.set_title("Sites ({site1},{site2})".format(site1=target1[0]+27, site2=target1[1]+27), y= 1.0, size=20)
    ax1.set_yticks(range(2))
    ax1.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax1.set_xticks([])
    ax1.set_xticklabels(doublealph_ticks, size=14)
    ax1.text(-0.03, 1.0, 'A', transform=ax1.transAxes, size=20, weight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ##############################
    ax2.imshow(terms[target2], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target2]):
         ax2.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax2.set_xticklabels(doublealph_ticks, size=14)
    ax2.set_title("Sites ({site1},{site2})".format(site1=target2[0]+27, site2=target2[1]+27), y= 1.0, size=20)
    ax2.set_yticks(range(2))
    ax2.set_xticks([])
    ax2.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax2.text(-0.03, -0.3, 'B', transform=ax1.transAxes, size=20, weight='bold')
    ax2.set_ylabel("Activity Class", size=16)
    ##############################
    ax3.imshow(terms[target3], cmap = 'winter')
    for (i,j), z in np.ndenumerate(terms[target3]):
         ax3.text(j, i, '{:0.3f}'.format(round(z, 4)), ha='center', va='center', size = 12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax3.set_title("Sites ({site1},{site2})".format(site1=target3[0]+27, site2=target3[1]+27), y= 1.0, size=20)
    ax3.set_xlabel("Genotype Class", size=16)
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['Lesser', 'Superior'], size=14)
    ax3.set_xticks(range(16))
    ax3.set_xticklabels(doublealph_ticks, size=14)
    ax2.text(-0.03, -1.6, 'C', transform=ax1.transAxes, size=20, weight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")    
create_suppfig_1((11,3), (4,3), (3,2), eps_data[2])

