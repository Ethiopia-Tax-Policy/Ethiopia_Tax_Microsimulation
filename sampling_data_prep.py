"""
This is a file that allows sampling of a large dataset.
"""
import sys
sys.path.insert(0, 'C:/Users/wb305167/OneDrive - WBG/python_latest/Tax-Revenue-Analysis')
from stata_python import *
import pandas as pd
import numpy as np

pit_df=pd.read_csv('pit_ethiopia_big.csv')

df, df_count = my_qcut_equal_length(pit_df, 'Employment_Income', 100)

pit_df=pit_df.sort_values(by=['Employment_Income'])
pit_df=pit_df.reset_index()
# allocate the data into bins
pit_df['bin'] = pd.qcut(pit_df['Employment_Income'], 10, labels=False)
pit_df['weight']=1
# bin_ratio is the fraction of the number of records selected in each bin
# 1/10,...1/5, 1/1
bin_ratio=[10,10,10,10,10,10,10,5,2,1]
frames=[]
df={}
for i in range(len(bin_ratio)):
    # find out the size of each bin
    bin_size=len(pit_df[pit_df['bin']==i])//bin_ratio[i]
    # draw a random sample from each bin
    df[i]=pit_df[pit_df['bin']==i].sample(n=bin_size)
    df[i]['weight'] = bin_ratio[i]
    frames=frames+[df[i]]

pit_sample= pd.concat(frames)
pit_sample = pit_sample.sort_values(by=['Employment_Income'])
pit_sample['Year'] = 2022
pit_sample = pit_sample[['id_n', 'Year', 'Employment_Income', 'weight']]
pit_sample.to_csv('taxcalc/pit_ethiopia_sample.csv')

df_weight = pit_sample[['weight']]

df_weight.columns = ['WT2022']
df_weight['WT2023'] = df_weight['WT2022']
df_weight['WT2024'] = df_weight['WT2022']
df_weight['WT2025'] = df_weight['WT2022']
df_weight['WT2026'] = df_weight['WT2022']
df_weight['WT2027'] = df_weight['WT2022']
df_weight['WT2028'] = df_weight['WT2022']
df_weight['WT2029'] = df_weight['WT2022']
df_weight['WT2030'] = df_weight['WT2022']
df_weight['WT2031'] = df_weight['WT2022']
df_weight['WT2032'] = df_weight['WT2022']

df_weight.to_csv('taxcalc/pit_ethiopia_sample_weights.csv')

varlist = ['Employment_Income']

total_weight_sample = pit_sample['weight'].sum()
total_weight_population = pit_df['weight'].sum()
#comparing the statistic of the population and sample
for var in varlist:
    pit_sample['weighted_'+var] = pit_sample[var]*pit_sample['weight']
    sample_sum = pit_sample['weighted_'+var].sum()
    population_sum = pit_df[var].sum()
    print("            Sample Sum for ", var, " = ", sample_sum)
    print("        Population Sum for ", var, " = ", population_sum)
    print(" Sampling Error for Sum(%) ", var, " = ", "{:.2%}".format((population_sum-sample_sum)/population_sum))
    sample_mean = sample_sum/total_weight_sample
    population_mean = population_sum/total_weight_population
    print("           Sample Mean for ", var, " = ", sample_mean)
    print("       Population Mean for ", var, " = ", population_mean)
    print("Sampling Error for Mean(%) ", var, " = ", "{:.2%}".format((population_mean-sample_mean)/population_mean))    
