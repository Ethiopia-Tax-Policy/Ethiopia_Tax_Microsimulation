# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:48:34 2022

@author: wb305167
"""
import sys
sys.path.insert(0, 'C:/Users/ssj34/Documents/OneDrive/python_latest/Tax-Revenue-Analysis')
from stata_python import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math, random
import scipy
import numba
import time
from scipy.stats import gaussian_kde

def expand_df(df, Z, id_n, weight):
    df = df[df[Z].notna()]   
    df = df[df[Z]>0]    
    df = df[[Z, id_n, weight]]
    df_expanded = df.loc[df.index.repeat(df[weight])].reset_index(drop=True)     
    df_expanded = df_expanded[[id_n, Z]].sort_values(by=Z)
    return df_expanded

def calc_a(df, Z):
    ''' calculates the value of alpha using the Diamond-Saez approximation
    '''
    #df_a = calc_a(df_hhs_new, 'totinc')
    df = df.sort_values(by=Z, ascending=False)  
    df['Z*+'] = df[Z].cumsum()
    df['N'] = np.arange(len(df))
    df['Zm'] = df['Z*+']/df['N']
    df['a'] = df['Zm']/(df['Zm']-df[Z])    
    df = df.sort_values(by=Z)
    return df

def pareto_dist(max_val, a):
    x = max_val
    df = pd.DataFrame(columns = ['totinc'])
    df = df.append({'totinc' : x}, ignore_index=True)
    i=1
    while (x>=10000):
        x = ((df['totinc'].iloc[-1])/i)*(((a-1)/a)+(i-1))
        #x = ((a-1)/a)*df.loc[:, 'totinc'].values.mean()
        df = df.append({'totinc' : x}, ignore_index=True)
        print(x)
        i=i+1
    return df

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def taxamt(income, rt1, rt2, rt3, rt4, rt5, bk1, bk2, bk3, bk4, bk5):
    """
    Compute tax amount given the specified taxable income
    and the specified progressive tax rate structure.
    """
    tax_amount = (
        rt1 * min(income, bk1) +
        rt2 * min(bk2 - bk1, max(0, income - bk1)) +
        rt3 * min(bk3 - bk2, max(0, income - bk2)) +
        rt4 * min(bk4 - bk3, max(0, income - bk3)) +
        rt5 * min(bk5 - bk4, max(0, income - bk4))
    )
    return tax_amount

@numba.njit()
def taxamt_eth(income):
    """
    Compute tax amount given the specified taxable income
    and the specified progressive tax rate structure.
    """
    rt1 = 0.0
    bk1 = 7200
    rt2 = 0.10
    bk2 = 19800
    rt3 = 0.15
    bk3 = 38400
    rt4 = 0.20
    bk4 = 63000
    rt5 = 0.25
    bk5 = 93600
    rt6 = 0.30
    bk6 = 130800
    rt7 = 0.35
    bk7 = 4e99     
    tax_amount = (
        rt1 * min(income, bk1) +
        rt2 * min(bk2 - bk1, max(0, income - bk1)) +
        rt3 * min(bk3 - bk2, max(0, income - bk2)) +
        rt4 * min(bk4 - bk3, max(0, income - bk3)) +
        rt5 * min(bk5 - bk4, max(0, income - bk4)) +
        rt6 * min(bk6 - bk5, max(0, income - bk5)) +
        rt7 * min(bk7 - bk6, max(0, income - bk6))        
    )
    return tax_amount

def taxamt_alb1(income):
    """
    Compute tax amount given the specified taxable income
    and the specified progressive tax rate structure.
    """
    rt1 = 0.0
    bk1 = 30000
    rt2 = 0.13
    bk2 = 200000
    rt3 = 0.23
    bk3 = 2e99
    rt4 = 0.0
    bk4 = 3e99
    rt5 = 0.0
    bk5 = 4e99
    tax_amount = (
        rt1 * min(income, bk1) +
        rt2 * min(bk2 - bk1, max(0, income - bk1)) +
        rt3 * min(bk3 - bk2, max(0, income - bk2)) +
        rt4 * min(bk4 - bk3, max(0, income - bk3)) +
        rt5 * min(bk5 - bk4, max(0, income - bk4))
    )
    return tax_amount

def generate_distribution(U, alpha, num_points):
    def inverse_pareto_cdf(y, U, alpha):
        # Computed analytically
        #U = 1e6
        #alpha = 1.5
        # The CDF of a pareto is given by (y =) P(X<x) = 1-(U/x)^alpha for
        # x greater than U and ) otherwise
        # Given a probability y we can find the x using the inverse
        # distribution = U/(1-y)^(1/alpha)
        # so we are generating different x values by giving a random number
        # between 0 and 1 (probability) and using that for y
        eps = 1e-14
        if (y==0):
            return random.randrange(0, U-eps)
        else:
            return U*((1/(1-y))**(1/alpha))   
    
    def inverse_cdf1(y):
        # Computed analytically
        return math.sqrt(math.log(-1/(y - 1)))
    
    def sample_distribution(U, alpha):
        uniform_random_sample = random.random()
        return inverse_pareto_cdf(uniform_random_sample, U, alpha)

    x = [sample_distribution(U, alpha) for i in range(num_points)]
    #bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x, x, statistic='mean', bins=N)
    # imputed_values = 
    log_x =np.log(x)
    mean_x = round(np.mean(x),2)
    mean_log_x = np.log(np.mean(x))
    ax = plt.hist(log_x, bins=50)
    plt.axvline(mean_log_x, color='k', linestyle='dashed', linewidth=1)
    plt.annotate(str(mean_x), (mean_log_x, 500))
    plt.show()
    return x

import seaborn as sns
def plot_density_chart(df1, df1_desc, df2, df2_desc, variable, title=None, xlabel=None, logx=None, vline=None):
    if logx is None:
        display_variable = 'disp_'+variable
        df1.loc[:,display_variable] = df1[variable]
        df2.loc[:,display_variable] = df2[variable]
        df1_sns = df1[[display_variable]]
        df2_sns = df2[[display_variable]]
        if vline is not None:
            adj_vline = vline            
    else:
        min_val1 = df1[variable].min()
        if (min_val1<0):
            raise ValueError("Cannot Calculate Logs of Negative Values")
        elif np.isclose(df1[variable],np.zeros(len(df1)), atol=0.0001).any():
            # add 1 so that any zero values are adjusted to 1           
            df1[variable] += 1
        min_val2 = df2[variable].min()
        if (min_val2<0):
            raise ValueError("Cannot Calculate Logs of Negative Values")
        elif np.isclose(df2[variable],np.zeros(len(df2)), atol=0.0001).any():
            # add 1 so that any zero values are adjusted to 1           
            df2[variable] += 1
        display_variable = 'ln_'+variable
        df1.loc[:,display_variable] = np.log(df1[variable])
        df2.loc[:,display_variable] = np.log(df2[variable])
        df1_sns = df1[[display_variable]]
        df2_sns = df2[[display_variable]]
        if vline is not None:
            if (vline>0):
                adj_vline = np.log(vline)
            else:
                vline = None
        
    df1_sns.loc[:, 'source'] = df1_desc    
    df2_sns.loc[:, 'source'] = df2_desc
    df = pd.concat([df1_sns, df2_sns], axis=0) 
    sns.displot(df, kind='kde', x=display_variable, hue="source")
    if vline is not None:
        plt.axvline(adj_vline, color='b')
    if title is None:
        title = "Density Plot"
    plt.title(title)
    if xlabel is None:
        plt.xlabel(variable)
    else:
        plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.show()


def plot_density_chart1(df1, df1_desc, variable, title=None, xlabel=None, logx=None, vline=None):
    import seaborn as sns
    if logx is None:
        display_variable = 'disp_'+variable
        df1.loc[:,display_variable] = df1[variable]
        df1_sns = df1[[display_variable]]
        if vline is not None:
            adj_vline = vline            
    else:
        min_val1 = df1[variable].min()
        if (min_val1<0):
            raise ValueError("Cannot Calculate Logs of Negative Values")
        elif np.isclose(df1[variable],np.zeros(len(df1)), atol=0.0001).any():
            # add 1 so that any zero values are adjusted to 1           
            df1[variable] += 1

        display_variable = 'ln_'+variable
        df1.loc[:,display_variable] = np.log(df1[variable])
        df1_sns = df1[[display_variable]]
        if vline is not None:
            if (vline>0):
                adj_vline = np.log(vline)
            else:
                vline = None
        
    df1_sns.loc[:, 'source'] = df1_desc
    sns.displot(df1_sns, kind='kde', x=display_variable, hue="source")
    if vline is not None:
        plt.axvline(adj_vline, color='b')
    if title is None:
        title = "Density Plot"
    plt.title(title)
    if xlabel is None:
        plt.xlabel(variable)
    else:
        plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.show()
    
def centiles(df, ranking_col, centile_length):
    #df = df.rank(method='first')
    df1 = df[[ranking_col]]
    #df.sort_values('age_group')
    result, bins = pd.qcut(df[ranking_col], centile_length, labels=False, retbins=True)
    df['centile'] = result
    df1.loc[:,'centile'] = result
    df2 = df1.groupby(by=['centile']).sum()
    df2 = df2.rename(columns={ranking_col:'sum_'+ranking_col}) 
    df3 = df1.groupby(by=['centile']).count()
    df3 = df3.rename(columns={ranking_col:'count_'+ranking_col})
    df4 = df1.groupby(by=['centile']).mean()
    df4 = df4.rename(columns={ranking_col:'mean_'+ranking_col})
    df5 = df1.groupby(by=['centile']).median()
    df5 = df5.rename(columns={ranking_col:'median_'+ranking_col})    
    df6 = pd.concat([df2, df3, df4, df5], axis=1)
    #df6.loc[centile_length,:]=[0,0,0,0]
    return (df6, bins, df)

def centile_compare(df1, df1_desc, df2, df2_desc, variable, centile_length):    
    df_1, bins1, df1 = centiles(df1, variable, centile_length)
    df_centiles1 = df_1.rename(columns={'sum_'+variable: df1_desc+' Sum',
                                           'count_'+variable: df1_desc+' Count',
                                           'mean_'+variable: df1_desc+' Mean',
                                           'median_'+variable: df1_desc+' Median'})
    df_2, bins2, df2 = centiles(df2, variable, centile_length)
    df_centiles2 = df_2.rename(columns={'sum_'+variable: df2_desc+' Sum',
                                           'count_'+variable: df2_desc+' Count',
                                           'mean_'+variable: df2_desc+' Mean',
                                           'median_'+variable: df2_desc+' Median'})
    df_centiles = df_centiles1.join(df_centiles2)
    plt.figure(0)
    df_centiles.plot(y=[df1_desc+' Sum', df2_desc+' Sum'], kind='bar')
    plt.show()
    df_1.loc[centile_length,:]=[0,0,0,0]
    df_1['bin_edge'] = bins1
    df_2.loc[centile_length,:]=[0,0,0,0]
    df_2['bin_edge'] = bins2
    return (df_1, df_2, df1, df2)

@numba.jit()
def calc_pit(num_recs, totinc):
    pitax=np.zeros(num_recs)
    for i in numba.prange(0, num_recs):
        pitax[i] = taxamt_eth(totinc[i])
    return pitax

def summary_stats(df, variable, zero_bracket, title):
    # Summary Statistics
    df[variable] = df[variable].astype(float)
    len_df = len(df)
    pitax = calc_pit(len_df,df[variable].values)
    df = df.reset_index()
    df['tax_payer'] = np.where(df[variable]>zero_bracket,1,0)
    df['pitax'] = pitax
    df['post_tax_'+variable] = df[variable] - df['pitax']
    #df_tax['dedinted_frac']=np.where(df_tax['totinc']==0, 0, df_tax['dedinted']/df_tax['totinc'])
    #df_tax['dedmedex_frac']=np.where(df_tax['totinc']==0, 0, df_tax['dedmedex']/df_tax['totinc'])
    print('Pre Tax Gini ('+title+'): ', gini(df[variable].values))
    print('Post Tax Gini ('+title+'): ', gini(df['post_tax_'+variable].values))
    print('Tax Collection ('+title+'): ', df['pitax'].sum())
    print('Number of Taxpayers ('+title+'): ', df['tax_payer'].sum())
    return df

def summary_stats_slow(df, variable, zero_bracket, title):
    # Summary Statistics
    df[variable] = df[variable].astype(float)
    df = df.reset_index()
    df['tax_payer'] = np.where(df[variable]>zero_bracket,1,0)
    df['pitax'] = 0
    for i in range(len(df)):
        df['pitax'].values[i] = taxamt_alb1(df[variable].values[i])
    df['post_tax_'+variable] = df[variable] - df['pitax']
    #df_tax['dedinted_frac']=np.where(df_tax['totinc']==0, 0, df_tax['dedinted']/df_tax['totinc'])
    #df_tax['dedmedex_frac']=np.where(df_tax['totinc']==0, 0, df_tax['dedmedex']/df_tax['totinc'])
    print('Pre Tax Gini ('+title+'): ', gini(df['totinc'].values))
    print('Post Tax Gini ('+title+'): ', gini(df['post_tax_'+variable].values))
    print('Tax Collection ('+title+'): ', df['pitax'].sum())
    print('Number of Taxpayers ('+title+'): ', df['tax_payer'].sum())
    return df


def calc_alpha(df, s, r, variable, statistic):
    """
    Calculates the Maximum Likelihood Estimator for the pareto parameter
    alpha = sigma_fi/(fi*log(Mi)+sigmafi*log(Ms)-fr*log(Ms/U))
    See West(1985)
    """
    sigma_fi = 0
    fi_log_Mi = 0
    Ms = df.loc[s][statistic+'_'+variable]
    fr = df.loc[r]['count_'+variable]
    U = df.loc[r]['bin_edge']
    for i in range(s,r):
        sigma_fi = sigma_fi + df.loc[i]['count_'+variable]
        fi_log_Mi = fi_log_Mi + (df.loc[i]['count_'+variable]*
                         np.log(df.loc[i][statistic+'_'+variable]))

    alpha = sigma_fi/(fi_log_Mi-sigma_fi*np.log(Ms)-fr*np.log(Ms/U))
    return alpha

def survival_plot(df, variable):
    '''Survival Function Plot
    # P[x>X] = X^(-alpha)
    # log(P[x>X]) = -alpha*log[X]
    '''
    log_P_x_gt_X = []
    log_X = []
    len_df = len(df)
    for i in range(1, len_df, 100):
        X = df.loc[i,variable]
        P_x_gt_X = (len_df-i)/len_df
        #P_x_gt_X = len(df[df[variable]>X])/len_df
        log_P_x_gt_X = log_P_x_gt_X + [np.log(P_x_gt_X)]
        log_X = log_X + [np.log(X)]
    
    df1 = pd.DataFrame()
    df1['log_X'] = log_X
    df1['log_P_x_gt_X'] = log_P_x_gt_X
    
    z = np.polyfit(df1[df1['log_X']>13.0]['log_X'], df1[df1['log_X']>13.0]['log_P_x_gt_X'], 1) 
    poly_fn = np.poly1d(z)

    plt.plot(log_X, log_P_x_gt_X, 'o')
    plt.plot(df1[df1['log_X']>10.0]['log_X'], poly_fn(df1[df1['log_X']>10.0]['log_X']),"r--")
    plt.annotate('alpha = '+str(round(z[0],3)),
             (max(log_X)-5, max(log_P_x_gt_X)),
             size = 12)
    plt.show()
   
def smooth_dataset(df, field_name, weight_field):
    
    # Extract relevant columns
    income = df[field_name]
    weights = df[weight_field]
    
    # Compute total number of weighted observations
    total_weight = int(sum(weights))  # Convert to an integer for number of points
    
    # Perform KDE with weights
    kde = gaussian_kde(income, weights=weights)
    
    # Generate values for plotting (number of points = total weight)
    x_vals = np.linspace(income.min(), income.max(), total_weight)
    y_vals = kde(x_vals)  # Estimated density
    
    # Scale the KDE so that total observations equal the sum of weights
    y_vals *= sum(weights) / np.trapz(y_vals, x_vals)
    
    # Create DataFrame with smoothed values
    smoothed_df = pd.DataFrame({"Employment_Income": x_vals, "Density": y_vals})
    
    # Plot the original data (Histogram)
    plt.figure(figsize=(9, 5))
    plt.hist(income, bins=50, weights=weights, alpha=0.5, color='gray', label="Original Data (Histogram)")
    
    # Plot the KDE function
    plt.plot(x_vals, y_vals, label="Smoothed KDE", color="red", linewidth=2)
    
    # Use logarithmic scale for x-axis
    plt.xscale("log")
    
    # Labels and title
    plt.xlabel("Employment Income (Log Scale)")
    plt.ylabel("Density (scaled to total weight)")
    plt.title("Original vs. Smoothed Employment Income Distribution")
    plt.legend()
    plt.grid()
    
    # Show plot
    plt.show()
    return smoothed_df
    
    
    
filename = 'pit_ethiopia_syn.csv'
df_income_all = pd.read_csv(filename)
#smoothed_df = smooth_dataset(df_income_all, 'Employment_Income', 'weight')

#plot_density_chart(smoothed_df, "HHS", 'Employment_Income', "Density Plot for Ethiopia", "Income", logx=True, vline=None)

df_expanded = expand_df(df_income_all, 'Employment_Income', 'id_n', 'weight')
df_expanded = df_expanded.sort_values(by='Employment_Income')
df_expanded = df_expanded.reset_index(drop=True)
#df_expanded['totinc'] += 1
#df_expanded['ln_totinc'] = np.log(df_expanded['totinc'])

#plot_density_chart(df_expanded, "HHS", smoothed_df, "Smoothed HHS" , 'Employment_Income', "Density Plot for Ethiopia", "Income", logx=True, vline=None)

# Plot single Dataset
plot_density_chart1(df_expanded, "HHS", 'Employment_Income', "Density Plot for Ethiopia", "Income", logx=True, vline=10.3)

#Find the Centiles for the Data and the bin edges.
#Then replace the 90th pecentile of the Household Survey 
#with the Pareto estimates.
centile_length=20
ranking_col = "Employment_Income"
df_1, bins, df_expanded = centiles(df_expanded, ranking_col, centile_length)
df_1.loc[centile_length,:]=[0,0,0,0]
df_1['bin_edge'] = bins
print("HHS Original Data: \n", df_1)

r = 19 
# fr is the number of observations in the last centile group
# that needs to be replaced by a pareto distribution
fr=0 # initialize fr=0
for i in range(r, centile_length):
    fr = fr+ int(df_1.loc[i,'count_Employment_Income'])

# Find the bin edge of the last replace bin
U = df_1.loc[r,'bin_edge']

# alternatively fr could be 
df_hhs_cut = df_expanded[df_expanded['Employment_Income']>=U]
fr = len(df_hhs_cut)

s=15
alpha = calc_alpha(df_1, s, r,'Employment_Income','median')
print('alpha: ',alpha)
#alpha = 3.0
# Generate the pareto distribution
x = generate_distribution(U, alpha, fr)
df_pareto = pd.DataFrame(x, columns =['Employment_Income'])
df_pareto = df_pareto.sort_values(by='Employment_Income')
df_pareto = df_pareto.rename(columns={'Employment_Income': 'Employment_Income_New'})

# comparing the cut portion and the pareto
df_hhs_cut = df_hhs_cut.rename(columns={'Employment_Income': 'Employment_Income_Old'})
df_hhs_cut1 = df_hhs_cut[['id_n', 'Employment_Income_Old']].reset_index(drop=True)
df_pareto1 = df_pareto.reset_index(drop=True)

df_hhs_cut_pareto = df_hhs_cut1.join(df_pareto1)
# adjust the Income to ensure that the pareto adjustment is not below 
# the original dataset
df_hhs_cut_pareto['Employment_Income']=np.where(df_hhs_cut_pareto['Employment_Income_New']> 
                                                df_hhs_cut_pareto['Employment_Income_Old'], 
                                                df_hhs_cut_pareto['Employment_Income_New'], 
                                                df_hhs_cut_pareto['Employment_Income_Old'])

# keep the rows from the lower percentiles in the Household Survey
df_hhs_kept = df_expanded[df_expanded['Employment_Income']<U][['id_n','Employment_Income']]
# attach the pareto adjusted values
df_hhs_new = pd.concat([df_hhs_kept, df_hhs_cut_pareto[['id_n', 'Employment_Income']]], axis=0)
df_hhs_new = df_hhs_new.sort_values(by='Employment_Income').reset_index()
df_hhs_new = df_hhs_new.drop('index', axis=1)

# New density plots for the adjusted distribution
plot_density_chart(df_expanded, "HHS", df_hhs_new, "HHS Adjusted", 'Employment_Income', "Density Plot for Ethiopia", "Income", logx=True, vline=10.3)

#plot_density_chart(df_hhs_new, "HHS", df_tax, "Tax Return", 'totinc', "Density Plot for Albania", "Income", logx=True, vline=10.3)

#New centile plots for the adjusted distribution
df_1_updated, df_updated, df_expanded, df_hhs_new  = centile_compare(df_expanded, 'Household Survey', df_hhs_new, 'Household Survey Updated', 'Employment_Income', centile_length) 

# Summary statistics including income tax estimates from all datasets
print("HHS Adjusted Data: \n", df_updated)

#pitax=np.zeros(len(df_expanded))
start = time.time()
df_expanded = summary_stats(df_expanded, 'Employment_Income', 7200, "HHS Unadjusted")
df_hhs_new = summary_stats(df_hhs_new, 'Employment_Income', 7200, "HHS Adjusted")

df_hhs_new.to_csv("pit_ethiopia_big.csv")

end = time.time()
print ("Time elapsed numba:", end - start)
survival_plot(df_expanded,'Employment_Income')
survival_plot(df_hhs_new,'Employment_Income')


# Tax Distribution
df = pd.concat([tabulate(df_expanded, 'sum', ['pitax'], by='centile', title='HHS'),
                tabulate(df_hhs_new, 'sum', ['pitax'], by='centile', title='HHS Adj')], axis=1)

df.plot(kind='bar')
plt.show()
#df_a = calc_a(df_hhs_new, 'totinc')
#df_a['ln_totinc'] = np.log(df_a['totinc'])
#df_a.plot(kind='scatter', x='totinc', y='a')
#x = df_a[df_a['totinc']<5.0e+07].plot(kind='scatter', x='totinc', y='a')
#ax = df_a[df_a['totinc']<5.0e+07].plot(kind='scatter', x='ln_totinc', y='a')
#ax = df_a.plot(kind='scatter', x='ln_totinc', y='a')
#plt.plot()
#ax.axvline(1e6, color='b', linestyle='--')
#ax.axvline(cut_off, color='b', linestyle='--')









