# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:45:48 2021

@author: Nate
"""
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
                    self.family.append(list(combined_data.df.iloc[sequence]))
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

