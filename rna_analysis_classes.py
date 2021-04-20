# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:45:48 2021

@author: Nate
"""
import pandas as pd
import ranVar as rV
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as pltc
import pickle
from progressbar import ProgressBar
import scipy as sp
import copy
import seqlogo as seql
import networkx as nx
import sklearn as skl
from networkx.utils import pairwise
import itertools
import math
#########################################
class data_import:
    # Simple Object for importing data   
    def __init__(self, file_name):       
        self.df = pd.read_csv(file_name)   
    def drop(self, key):
        self.df.drop(key)


#########################################
class distribution_data:
    
    def __init__(self):
        pass  
    
    def import_data(self, data_import, keep):
        self.combined_data = data_import
        self.data = []        
        for i in range(len(self.combined_data.df)):
            self.data.append(seq(self.combined_data.df.iloc[i]))           
        for label in self.combined_data.df.columns:
            if label not in keep:
                self.combined_data.df = self.combined_data.df.drop(label, axis=1)              
        self.column_labels = self.combined_data.df.columns
        self.families = {}               
        for fam_name in ['Fam2.1', 'Fam2.2', 'Fam1A.1', 'Fam1B.1', 'Fam3.1']:
            self.family = []
            for sequence in range(len(self.combined_data.df)):
                if self.combined_data.df[fam_name][sequence] <= 2:
                    self.family.append(list(self.combined_data.df.iloc[sequence]))
            self.families.update({fam_name : [pd.DataFrame(self.family, columns = self.column_labels)]})
            self.family_names = self.families.keys()
            
    def add_family(self, data_file, keep):
        self.combined_data = data_import
        self.data = []      
        for i in range(len(self.combined_data.df)):
            self.data.append(seq(self.combined_data.df.iloc[i]))          
        for label in self.combined_data.df.columns:
            if label not in keep:
                self.combined_data.df = self.combined_data.df.drop(label, axis=1)              
        self.column_labels = self.combined_data.df.columns              
        for fam_name in ['Fam2.1', 'Fam2.2', 'Fam1A.1', 'Fam1B.1', 'Fam3.1']:
            self.family = []
            for sequence in range(len(self.combined_data.df)):
                if self.combined_data.df[fam_name][sequence] <= 2:
                    self.family.append(list(combined_data.df.iloc[sequence]))
            self.families.update({fam_name : [pd.DataFrame(self.family, columns = self.column_labels)]})
            self.family_names = self.families.keys()
###########################################
class seq:
    def __init__(self, data, index = 'NA', stats_value = 'N/A'):
        self.atts = {}
        self.index = index
        self.sequence = data['seq']
        self.list_seq = list(self.sequence)
        self.seq_len = len(self.sequence)
        
    def update_att(self, key, value):
        self.atts.update({key : value})
###########################################          
class sample_list:  
    def __init__(self, samples, data_distribution, substrates):
        self.data_distribution = data_distribution
        self.substrates = substrates
        self.supe = None
        pbar = ProgressBar()
        self.samples = samples
        for i in pbar(self.samples):
            for j in substrates:                 
                i.add_entry({j : float(data_distribution.combined_data.df.loc[data_distribution.combined_data.df['seq'] == i.data['seq']][j])})            
    def hamming_mat(self):
        self.ham_mat = np.zeros((len(self.samples), len(self.samples)))
        helper = helper_functions()
        for i in range(self.ham_mat.shape[0]):
            for j in range(self.ham_mat.shape[1]):
                self.ham_mat[i][j] = helper.hamming(self.samples[i].data['seq'], self.samples[j].data['seq'])            
    def lower(self, sample, ham_distance, label):
        low = []
        helper = helper_functions()
        for i in self.samples:
            if (ham_distance == helper.hamming(sample.data['seq'], i.data['seq']))  and (i.data[label] < sample.data[label]):
                low.append(i)
        return low    
    def sort(self, label):
        self.samples.sort(key = lambda x:x.data[label])
    def class_matrix(self, bin_labels):
        self.overlap_matrix = np.zeros((2,2))
        for i in self.samples:
            self.overlap_matrix[i.data[bin_labels[0]]][i.data[bin_labels[1]]] += 1
        self.overlap_matrix /= len(self.samples)        
    def contained_class(self, label):
        for i in self.samples:
            if i in self.subwindow.samples:
                i.add_entry({label : 1})
            else:
                i.add_entry({label : 0})                
    def extract_above(self, label, thresh):
        self.percentage_sequence()
        self.subwindow = sample_list([i for i in self.samples if (i.data[label] > thresh)], self.data_distribution, self.substrates)
        self.subwindow.percentage_sequence(background = None)        
    def double_class(self, label, threshes):
        self.extract_above(label, threshes[0])
        self.contained_class('essential')
        self.extract_above(label, threshes[1])
        self.contained_class('expert')                
    def extract_window(self, labels, lower_bound, upper_bound):
        self.percentage_sequence()
        self.subwindow = sample_list([i for i in self.samples if ((i.data[labels[0]] > lower_bound and i.data[labels[1]] > lower_bound) and (i.data[labels[0]] < upper_bound and i.data[labels[1]] < upper_bound))], self.data_distribution, self.substrates)
        self.subwindow.percentage_sequence(background = None)        
    def extract_circle(self, label, center, radius):
        self.percentage_sequence()
        self.subwindow = sample_list([i for i in self.samples if (i.data[label] > (center - radius) and i.data[label] < (center + radius)) ], self.data_distribution, substrates)
        try:
            self.subwindow.percentage_sequence(background = None)
        except:
            print('Warning, an empty subwindow was generated')        
    def subsample(self, target):
        sublist = []
        for i in self.samples:
            ref_dict = {k:i.data[k] for k in list(target.keys()) if k in i.data}
            if ref_dict == target:
                sublist.append(i)
        self.subwindow = sample_list(sublist, self.data_distribution, self.substrates)  
    def percentage_sequence(self, background = None):
        seq_breakdown = {}
        for i in range(len(self.samples[0].data['seq'])):
            local_dict = {'A' : 0, 'C' : 0, 'G' : 0, 'T' : 0}
            for j in self.samples:
                local_dict[j.data['seq'][i]] += 1
            for k in local_dict.keys():
                local_dict[k] /= len(self.samples)
            seq_breakdown.update({ i : local_dict})
        self.seq_breakdown = seq_breakdown
        self.df_pwf = pd.DataFrame.from_dict(self.seq_breakdown)
        self.ar_pwf = self.df_pwf.to_numpy().transpose()
        self.cpm = seql.CompletePm(self.ar_pwf, background = background)
        return seq_breakdown    
    
    def subwindow_complexity(self):
        self.subwindow.percentage_sequence()
        vector = self.subwindow.cpm.ppm.to_numpy().reshape(84,)
        entropy = 0.0
        for i in vector:
            if i != 0.0:
                entropy += -(i)*np.log2(i)
        return entropy  
#####################################
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
    
#####################################
def functional_form(cond_1, cond_2, cond_12):
    try:
        eps = float(cond_12*(math.log(cond_1, 2)+math.log(cond_2, 2)-math.log(cond_12, 2)))
        sur = float((math.log(cond_1, 2)+math.log(cond_2, 2)-math.log(cond_12, 2))) 
    except:
        eps = 0.0
        sur = 0.0      
    return -eps, -sur      
#####################################
def epistasis(rVar, activity_label, genotype1, genotype2):
    #####
    rV1 = copy.deepcopy(rVar)
    rV2 = copy.deepcopy(rVar)
    rV12 = copy.deepcopy(rVar)
    #####
    rV1.hist2d(activity_label, 'single'+str(genotype1), (2,4))
    rV2.hist2d(activity_label, 'single'+str(genotype2), (2,4))
    rV12.hist2d(activity_label, 'double'+str(genotype1)+'x'+str(genotype2), (2,16))
    #####
    pdf1 = rV1.pdf2d
    pdf2 = rV2.pdf2d
    pdf12 = rV12.pdf2d
    #####
    #####
    pdf12_copy = copy.deepcopy(pdf12)
    shape_pdf12 = np.reshape(pdf12_copy, (2,4,4))
    #####
    marginal_1 = np.sum(pdf1, 0)
    marginal_2 = np.sum(pdf2, 0)
    marginal_12 = np.sum(pdf12, 0)
    #####
    shape_marginal_12 = np.reshape(copy.deepcopy(marginal_12), (4,4))
    #####
    #####
    eps_mat = np.zeros((2,4,4))
    sur_mat = np.zeros((2,4,4))
    #####
    for i in range(2):
        ##################
        for j in range(4):
            ##################
            for k in range(4):
                cond1 = float(pdf1[i][j]/marginal_1[j])
                cond2 = float(pdf2[i][k]/marginal_2[k])
                cond12 = float(shape_pdf12[i][j][k]/shape_marginal_12[j][k])
                if float(str(cond12)) == float(cond12):
                    calc = functional_form(cond1, cond2, cond12)
                    eps_mat[i][j][k] = float(calc[0])
                    sur_mat[i][j][k] = float(calc[1])
                elif str(cond12) == 'nan':
                    eps_mat[i][j][k] = 0.0
                    sur_mat[i][j][k] = 0.0
                else:
                    print("Warning. Something very strange is happening in epistasis calculation")

    return eps_mat, sur_mat, (cond1, cond2, cond12)
#####################################    
def all_sites_eps(rVar, activity_label):
    eps_sum_matrix = np.zeros((21,21))
    sur_sum_matrix = np.zeros((21,21))
    terms_matrix = {}
    conds = {}    
    for m in range(21):
        for n in range(21):
            if m > n:
                print((m,n))
                ep_tot = epistasis(rVar, activity_label, m, n)
                ep = ep_tot[0]
                sur = ep_tot[1]
                cprob = ep_tot[2]
                conds.update({(m,n) : cprob})                
                if m != n:
                    eps_sum_matrix[m][n] = np.sum(ep)
                    sur_sum_matrix[m][n] = np.sum(sur)
                    terms_matrix.update({(m,n) : np.reshape(ep, (2,16))})                    
                print(eps_sum_matrix[m][n])                
    return eps_sum_matrix, sur_sum_matrix, terms_matrix, conds