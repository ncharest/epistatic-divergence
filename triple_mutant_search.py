# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:47:12 2021

@author: Nate
"""
#%% Import
import pandas as pd
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import copy
#%%
file_name = 'byo-variant-k-seq-results-all.csv'
path = "D:/projects/project-7_RNA/RealRNAData/csvs/"
target = path+file_name

seed_seq = 'CCACACTTCAAGCAATCGGTC'

def ham(seq1, seq2):
    count = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            count += 1

    return count

def mutant(mutations, ref):
    mut = copy.deepcopy(ref)
    mut = list(mut)
    for i in mutations:
        mut[i[0]] = i[1]
    return "".join(mut)
        

class seq_data:
    def __init__(self, target):
        self.df = pd.read_csv(target)
        
    def find_order(self, ref, dis):
        self.order_df = {}
        pbar = ProgressBar()
        for i in pbar(range(len(self.df))):
            if ham(ref, self.df.iloc[i]['seq']) == dis:

                self.order_df[self.df.iloc[i]['seq']] = self.df.iloc[i].to_dict()
#%%                
test = seq_data(target)           
test.find_order(seed_seq, 3)

for i in range(len(test.df)):
    if test.df.iloc[i]['seq'] == seed_seq:
        print(i)
#%%

keys = ['bs_kA_2.5%', 'bs_kA_50%', 'bs_kA_97.5%']

trip_dict = {}
for i in test.order_df.keys():
    trip_dict[i] = {}
    for j in keys:
        trip_dict[i][j] = test.order_df[i][j]
        
look_at = [[[2,'C'], [4,'G'], [11,'A']],[[2,'T'], [4,'G'], [11,'A']], [[2,'G'], [4,'G'], [11,'A']], [[2,'G'], [3,'A'], [11,'A']], [[2,'G'], [3,'T'], [11,'A']], [[2,'G'], [3,'G'], [11,'A']]]
look_seq = [mutant(i, seed_seq) for i in look_at]
        
medians = []
errors = []
data_points = []
for i in trip_dict.keys():
    data_points.append([trip_dict[i]['bs_kA_50%'],[trip_dict[i]['bs_kA_2.5%'], trip_dict[i]['bs_kA_97.5%']]])

data_points.sort(key=lambda x:x[0], reverse=True)

l_medians = []
l_errors = []
l_data_points = []
for i in look_seq:
    l_data_points.append([trip_dict[i]['bs_kA_50%'],[trip_dict[i]['bs_kA_2.5%'], trip_dict[i]['bs_kA_97.5%']]])

xs = []
medians = []
errors = []
for i in range(len(data_points)):
    xs.append(i)
    medians.append(data_points[i][0])
    errors.append(data_points[i][1])

errors = np.asarray(errors).transpose()



extent = 25

fig, ax = plt.subplots()
ax.errorbar(xs[:extent], medians[:extent], errors[:,:extent], ecolor='Red')
ax.scatter(xs[:extent], medians[:extent])
