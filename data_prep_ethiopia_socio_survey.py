# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 09:05:14 2025

@author: ssj34
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


my_dpi = 300

sys.path.insert(0, 'C:/Users/ssj34/Documents/OneDrive/python_latest/Tax-Revenue-Analysis')
from stata_python import *
from plot_charts import *

dir_name = "C:/Users/ssj34/Documents/OneDrive/python_latest/Microsimulation/Ethiopia_Tax_Microsimulation/"
# record serial number 9503 was corrected as it showed a person worked for the government
# but paid by the hour but it was clear that it was a monthly pament
file_name = "sect4_hh_w5_corrected.csv"
df_main = pd.read_csv(dir_name+file_name, index_col=False)

df_main['id_n']=df_main.index
df_main['Year']=2022

df_main = df_main.rename(columns={'saq14':'rural_urban', 'saq01':'region',
                                  's4q37':'months_worked', 's4q38':'weeks_worked', 
                                  's4q39':'hours_worked', 's4q40':'payment', 
                                  's4q41':'payment_frequency', 
                                  'pw_w5':'weight', 's4q34a':'job_desc', 
                                  's4q34b':'job_category', 's4q34c':'employer_activity', 
                                  's4q34d':'employer_activity_code', 
                                  's4q34':'employer_category', 's4q36':'formal'})

df_main['payment_multiplier'] = np.nan
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="1. HOUR"),
                                    (df_main['months_worked']*df_main['weeks_worked']
                                         *df_main['hours_worked']),
                                    df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="2. DAY"),
                                    7.0*(df_main['months_worked']*df_main['weeks_worked']),
                                    df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="3. WEEK"),
                                    (df_main['months_worked']*df_main['weeks_worked']),
                                    df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="4. FORTNIGHT"),
                                    0.5*(df_main['months_worked']*df_main['weeks_worked']),
                                    df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="5. MONTH"), 
                                    df_main['months_worked'],df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="6. QUARTER"), 
                                    4.0, df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="7. 1/2 YEAR"), 
                                    2.0, df_main['payment_multiplier'])
df_main['payment_multiplier'] = np.where(df_main['payment_multiplier'].isna()
                                    &(df_main['payment_frequency']=="8. YEAR"), 
                                    1.0, df_main['payment_multiplier'])

df_main['Employment_Income'] = df_main['payment']*df_main['payment_multiplier']

df_main['Other_Income_Federal'] = 0.0 
df_main['Other_Income_Regional'] = 0.0

df = df_main[['id_n', 'Year', 'rural_urban', 'region', 'Employment_Income', 
              'job_desc', 'job_category', 'employer_activity', 
              'employer_activity_code', 'employer_category', 'formal', 
              'weight']]

total_weight = df['weight'].sum()
df = df[df['Employment_Income'].notna()]
non_zero_payment_weight = df['weight'].sum()
df_weight = df[['weight']]

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

df.to_csv('taxcalc/pit_ethiopia.csv', index=False)
df_weight.to_csv('taxcalc/pit_weights_ethiopia.csv', index=False)