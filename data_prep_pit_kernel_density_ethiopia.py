# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 23:14:36 2025

@author: ssj34
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys

sys.path.insert(0, 'C:/Users/ssj34/Documents/OneDrive/python_latest/Tax-Revenue-Analysis')
from stata_python import *
from plot_charts import *

"""
Procedure to create synthetic dataset
1.  First generate the pit_ethiopia.csv file from the socio economic survey
2.  This file has gaps. so we replicate a smooth distribution
3.  This is done by finding the kernel density of the distribution of
    Employment_Income in pit_ethiopia
4.  Once we get the kde we generate a synthetic dataset of 100,000 observations
5.  We can do the Pareto correction of the synthetic data
6.  We calculate the pit from this synthetic dataset.
7.  This we then scale each observations by calculating the weights
    which uses the miltiplicative (factor tax collected from 
    the ministry of finance divided by the tax collection from the 
    synthetic dataset). 
8.  We re-run the weighted dataset in the microsim to check if we can generate

"""
# Load dataset
df = pd.read_csv('pit_ethiopia.csv')

plot_density_chart(df, 'Employment_Income', logx=True)

# Remove missing or zero values in Employment_Income
df = df[df["Employment_Income"].notna()]
df = df[df["Employment_Income"] > 0]

# Extract values and weights
income = df["Employment_Income"]
weights = df["weight"]

# Define the number of synthetic samples
num_samples = 100000

log_income = np.log(income)  # Take log to ensure all values are positive

kde_log = gaussian_kde(log_income, weights=weights)

kde_reg = gaussian_kde(income, weights=weights)

#synthetic_samples = kde_reg.resample(num_samples)[0]  

# Only Positive Values Sample from the KDE distribution
synthetic_log_samples = kde_log.resample(num_samples)[0]
synthetic_samples = np.exp(synthetic_log_samples)  # Convert back to original scale

# Fit KDE to synthetic samples
kde = gaussian_kde(synthetic_samples)

# Generate smooth KDE curve
x_vals_syn = np.linspace(min(synthetic_samples), max(synthetic_samples), 10000)
kde_values_syn = kde_reg(x_vals_syn)

# Generate smooth x values
x_vals = np.linspace(income.min(), income.max(), 10000)
kde_vals = kde_reg(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, kde_vals, label="KDE Estimate", color='blue')
plt.plot(x_vals_syn, kde_values_syn, label="Synthetic Data", color='red')
# Use log scale for better visualization
plt.xscale("log")
plt.xlabel("Income")
plt.ylabel("Density")
plt.title("KDE vs. Synthetic Data")
plt.legend()
plt.show()

# Create the Synthetic Dataset

pit_syn = pd.DataFrame({"Year": 2022, "Employment_Income": synthetic_samples, 'weight':1.0})

pit_syn = pit_syn.sort_values(by=['Employment_Income'])

pit_syn = pit_syn.reset_index(drop=True)

pit_syn['id_n'] = pit_syn.index

pit_syn = pit_syn[['id_n', 'Year', 'Employment_Income', 'weight']]

pit_syn.to_csv('pit_ethiopia_syn.csv')

plot_density_chart(pit_syn, 'Employment_Income', logx=True)

df_weight = pit_syn[['weight']]

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

df_weight.to_csv('pit_weights_ethiopia_syn.csv')

# reweight using tax projections calibrated
tax_collection_2021_22_billion = 84.560
# synthetic data has only 100,000 observations
tax_collection_syn_billion = 1.04

multiplicative_factor = tax_collection_2021_22_billion/tax_collection_syn_billion

pit_syn['weight'] = multiplicative_factor
pit_syn.to_csv('pit_ethiopia_syn.csv')

df_weight['WT2022'] = multiplicative_factor
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
df_weight.to_csv('pit_weights_ethiopia_syn.csv')
