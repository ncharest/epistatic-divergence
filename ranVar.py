# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:15:14 2020

@author: Nate
"""

import numpy as np
import math
import sklearn.neighbors as KernelDensity
#%%
class sample:
    
    def __init__(self, value):
        
        self.data = value
    
    def add_entry(self, value):
        
        ### Value should be a dictionary of style {label : value}...obviously...
        
        self.data.update(value)

class ranVar:
    
    def __init__(self, data_set):
        
        ### data_set should be a python list of samples taken from a random variable
        
        try:
            self.data_set = list(data_set)
        except ValueError:
            print("data_set should be a python list containing samples from the random variable")
    
    def hist(self, label, num_bins, base = 2):
        
        ### num_bins is the number of histogram bins used to estimate the probability distribution
        ### for the variable
        
        try:
            num_bins = int(num_bins)
        except ValueError:
            print("num_bins must be a valid integer")
            
        self.pdf = np.histogram([i.data[label] for i in self.data_set], bins = num_bins)[0]/len(self.data_set)
        point_entropies = []
        
        for i in self.pdf:
            try:
                point_entropies.append(i*math.log(i, base))
            except:
                pass
            
        self.self_entropy = - sum(point_entropies)
        
        
    def hist2d(self, label1, label2, bins = (25,2), base = 2):
        
        try:
            bins = tuple(bins)
        except ValueError:
            print("bins must be a tuple of dim 2 (see np.histogram2d doc)")
            
        self.pairs = np.asarray([[i.data[label1], i.data[label2]] for i in self.data_set])
        
        self.pdf2d = np.histogram2d(self.pairs.transpose()[0], self.pairs.transpose()[1], bins)[0]/len(self.data_set)
        self.pdf2d_i = np.sum(self.pdf2d, 1)
        self.pdf2d_j = np.sum(self.pdf2d, 0)
        
        self.mutual_ex = np.zeros(bins)
        for i in range(bins[0]):
            for j in range(bins[1]):
                self.mutual_ex[i][j] = self.pdf2d_i[i]*self.pdf2d_j[j]
        
        self.ment_i = 0.0
        self.ment_j = 0.0
        
        for i in self.pdf2d_i:
            try:
                self.ment_i += (-i*math.log(i, base))
            except:
                self.ment_i += 0.0
                
        for j in self.pdf2d_j:
            try:
                self.ment_j += (-j*math.log(j, base))
            except:
                self.ment_j += 0.0
            
        
        self.joint_entropy = []
        self.relative_entropy = []
        self.conditional_entropy_i = []
        self.conditional_entropy_j = []
        self.re_component_matrix = np.zeros((self.pdf2d.shape[0], self.pdf2d.shape[1]))
        self.surprisal_matrix = np.zeros((self.pdf2d.shape[0], self.pdf2d.shape[1]))
        self.re_terms = {}
        
        for i in range(self.pdf2d.shape[0]):
            for j in range(self.pdf2d.shape[1]):

                try:
                    self.joint_entropy.append(-self.pdf2d[i][j]*math.log(self.pdf2d[i][j], base))
                    self.relative_entropy.append(self.pdf2d[i][j]*math.log((self.pdf2d[i][j]/self.mutual_ex[i][j]), base))
                    self.re_terms.update({ (i,j) : self.pdf2d[i][j]*math.log((self.pdf2d[i][j]/self.mutual_ex[i][j]), base) })
                    self.re_component_matrix[i][j] = (self.pdf2d[i][j]*math.log((self.pdf2d[i][j]/self.mutual_ex[i][j]), base))
                    self.surprisal_matrix[i][j] = (self.pdf2d[i][j]*math.log((self.pdf2d[i][j]/self.mutual_ex[i][j]), base))

                except:
                    self.re_terms.update({ (i,j) : 0 })
                    pass
                
        for i in range(self.pdf2d.shape[0]):
            for j in range(self.pdf2d.shape[1]):

                if self.pdf2d[i][j] > 0.0:
                    self.conditional_entropy_i.append(-self.pdf2d[i][j]*math.log((self.pdf2d[i][j]/self.pdf2d_i[i]),base))
                    

                else:
                    self.conditional_entropy_i.append(0.0)
            
        for i in range(self.pdf2d.shape[0]):
            for j in range(self.pdf2d.shape[1]):
                if self.pdf2d[i][j] > 0.0:
                    self.conditional_entropy_j.append(-self.pdf2d[i][j]*math.log((self.pdf2d[i][j]/self.pdf2d_j[j]),base))
                else:
                    self.conditional_entropy_j.append(0.0)
            
        self.jent = sum(self.joint_entropy)
        self.rent = sum(self.relative_entropy)
        self.cent_j = sum(self.conditional_entropy_i)
        self.cent_i = sum(self.conditional_entropy_j)
        
    def cross_learning(self, targets, num_classes):
        self.cross_learning = np.zeros((len(targets),len(targets)))
        for i in range(len(targets)):
            for j in range(len(targets)):
                self.hist2d(targets[i], targets[j], (num_classes,num_classes))
                self.cross_learning[i][j] = self.rent
        self.unaccounted = np.zeros(self.cross_learning.shape)

        for i in range(self.cross_learning.shape[0]):
            for j in range(self.cross_learning.shape[0]):
                if i != j:
                    self.unaccounted[i][j] = abs(sum([self.cross_learning[i][i],self.cross_learning[j][j]])  - (2*self.cross_learning[i][j]))
                if i == j:
                    self.unaccounted[i][j] = self.cross_learning[i][i]

        
