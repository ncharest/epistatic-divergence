# code snippets for figure plotting

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = "D:/projects/project-7_RNA/RealRNAData/csvs/"
file_name = "triple-data-w-wt.csv"

df = pd.read_csv(path+file_name)

#%%

def rank_plot(df, y, rank_by=None, pos_col=None, yerr_u=None, yerr_l=None, ascending=False, ax=None, **kwargs):
    
    plot_kws = {**dict(elinewidth=1, capsize=2, capthick=1), **kwargs}
    
    if pos_col is None:
        if rank_by is None:
            raise ValueError('Please provide rank_by for ranking or rank_col for pre-ranked dataframe')
        else:
            df = df.sort_values(by=rank_by, ascending=ascending)
            df['rank'] = np.arange(df.shape[0])
            pos_col = 'rank'
    else:
        if rank_by is None:
            pass
        else:
            raise ValueError('Please only provide one of rank_by or pos_col')

    if ax is None:
        ax = plt.subplots(1, 1, figsize=(8, 4))
    if yerr_u is not None and yerr_l is not None:
        ax.errorbar(x=df[pos_col], y=df[y], yerr=(df[y] - df[yerr_l], df[yerr_u] - df[yerr_l]), **plot_kws)
    else:
        ax.errorbar(x=df[pos_col], y=df[y], **plot_kws)
    
    return ax


anchor_seqs = {'s-1B.1-WT': 'CCACACTTCAAGCAATCGGTC',
               's-1A.1-WT': 'CTACTTCAAACAATCGGTCTG'}
triple_mutants_predicted = {
    '3T5G12A': 'CCTCGCTTCAAACAATCGGTC',
    '3G5G12A': 'CCGCGCTTCAAACAATCGGTC',
    '3C5G12A': 'CCCCGCTTCAAACAATCGGTC',
    '3C4A12A': 'CCCAACTTCAAACAATCGGTC',
    '3C4T12A': 'CCCTACTTCAAACAATCGGTC',
    '3C4G12A': 'CCCGACTTCAAACAATCGGTC'
}


triples_1b_results = pd.read_csv(path+file_name, index_col=0)  # Load the data here

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

triples_1b_results.sort_values(by=['bs_kA_50%'], ascending=False, inplace=True)
triples_1b_results['rank'] = np.arange(triples_1b_results.shape[0])

# Plot for triples
rank_plot(df=triples_1b_results.loc[~triples_1b_results.index.isin(anchor_seqs.values())],
          y='bs_kA_50%', pos_col='rank', ls=None, marker='.', markersize=2.5,
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax, color='navy',
          elinewidth=0.6, capsize=1, capthick=0.6, ecolor='cornflowerblue', alpha=0.05)

# Fake plot for legend
rank_plot(df=triples_1b_results.loc[[]], y='bs_kA_50%', pos_col='rank', ls=None, marker='.', markersize=2.5,
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax, color='navy',
          elinewidth=0.6, capsize=1, capthick=0.6, ecolor='cornflowerblue', alpha=1, label='Triple mutants')

# Plot for predicted triple mutants
rank_plot(df=triples_1b_results.loc[triple_mutants_predicted.values()], y='bs_kA_50%', pos_col='rank', 
          ls=None, marker='.', markersize=5,
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax,
          elinewidth=0.6, capsize=1, capthick=0.6, ecolor='peachpuff', color='darkorange', label='Predicted triple mutants')

# Plot for WT of 1A and 1B
rank_plot(df=triples_1b_results.loc[anchor_seqs.values()], y='bs_kA_50%', pos_col='rank', ls=None, marker='.', markersize=5,
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax, color='red',
          elinewidth=0.6, capsize=1, capthick=0.6, ecolor='lightcoral', label='WT')

# ax.set_yscale('log')
ax.set_xlabel('Sequence Rank (Median $kA$)', size=16)
ax.set_ylabel('$kA$ (95 Confidence Interval) mol/min', size=16)

ax.legend(ncol=3, loc=[0.4, 0.9])
# fig.savefig()    # Save fig

plt.show()


# Zoomed in
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# First 86 seqs
df = triples_1b_results.iloc[:85]

# All triples
rank_plot(df=df.loc[~df.index.isin(anchor_seqs.values())], y='bs_kA_50%', pos_col='rank', ls='',
          marker='o', markersize=3,
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax, color='navy',
          elinewidth=1, capsize=2, capthick=1, ecolor='cornflowerblue', alpha=0.9)

# Predicted triples
rank_plot(df=df.loc[triple_mutants_predicted.values()], y='bs_kA_50%', pos_col='rank',
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax, ls='', marker='o', markersize=3,
          elinewidth=1, capsize=2, capthick=1, ecolor='peachpuff', color='darkorange')

# WT
rank_plot(df=df.loc[anchor_seqs.values()], y='bs_kA_50%', pos_col='rank',
          yerr_u='bs_kA_97.5%', yerr_l='bs_kA_2.5%', ax=ax, ls='', marker='o', markersize=3,
          elinewidth=1, capsize=2, capthick=1, ecolor='lightcoral', color='red')
# Modifiers
mods = {'CCTCGCTTCAAACAATCGGTC': 70, 'CCGCGCTTCAAACAATCGGTC': 60,'CCCCGCTTCAAACAATCGGTC' : 80,'CCCAACTTCAAACAATCGGTC' : 115,'CCCTACTTCAAACAATCGGTC' : 70,'CCCGACTTCAAACAATCGGTC': 60}


# Add alias/name for zoomed in sequences below the bar
for ix in df.index:
    seq = df.loc[ix]
    if seq.name in triple_mutants_predicted.values():
        ax.annotate(seq['alias'], (seq['rank'], seq['bs_kA_97.5%']+mods[seq.name]),color='black', alpha=1.0,
                rotation=90, ha='center', va='bottom', fontsize=10) 
    elif seq.name in anchor_seqs.values():
        ax.annotate(seq['alias'], (seq['rank'], seq['bs_kA_2.5%']+90.0), color='black', alpha=0.75,
                rotation=90, ha='center', va='bottom', fontsize=10)
    else:
        pass
        # ax.text(s=seq['alias'] + '  ', x=seq['rank'], y=seq['bs_kA_2.5%'], color='navy',
        #         rotation=90, ha='center', va='top', fontsize=8)

# ax.set_yscale('log')
ax.set_xlabel('Sequence Rank (Median $kA$)', size=16)
ax.set_ylabel('$kA$ (95 Confidence Interval) mol/min', size=16)
ax.set_ylim([1, 500])
ax.set_xlim([-1, 85])

# fig.savefig()   # Save fig

plt.show()


