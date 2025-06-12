import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'C:/Users/ssj34/Documents/OneDrive/python_latest/Tax-Revenue-Analysis')
from stata_python import *


def calc_pit(income_array, rate, bracket):
    def taxamt_eth(income, rate, bracket):
        """
        Compute tax amount given the specified taxable income
        and the specified progressive tax rate structure.
        """ 
        tax_amount = (
            rate[1] * min(income, bracket[1]) +
            rate[2] * min(bracket[2] - bracket[1], max(0, income - bracket[1])) +
            rate[3] * min(bracket[3] - bracket[2], max(0, income - bracket[2])) +
            rate[4] * min(bracket[4] - bracket[3], max(0, income - bracket[3])) +
            rate[5] * min(bracket[5] - bracket[4], max(0, income - bracket[4])) +
            rate[6] * min(bracket[6] - bracket[5], max(0, income - bracket[5])) +
            rate[7] * min(bracket[7] - bracket[6], max(0, income - bracket[6]))        
        )
        return tax_amount
    pitax=np.zeros(len(income_array))
    for i in range(0, len(income_array)):
        pitax[i] = taxamt_eth(income_array[i], rate, bracket)
    return pitax

# Load the dataset
df = pd.DataFrame()
#pd.options.display.float_format = "{:,.2f}".format
# generate 100 equally spaced incomes between 0 and 100,000
df['income'] = np.linspace(0, 30000, 1001)
# Current Schedules
# keep an extra 0 in the lists for rates and brackets to make the code work
rates = [0.0, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
brackets = [0, 7200, 19800, 38400, 63000, 93600, 130800, 4e99]
brackets_month = [x/12 for x in brackets]
df['tax'] = calc_pit(df['income'].values, rates, brackets_month)
df['avg_tax_rate1'] = df['tax']/df['income']
# Reform Option 1
# Yearly Income From	Upto	Rate
# 0	         19,200   0%
# 19,200	 36,000  15%
# 36,000	 60,000  20%
# 60,000	 132,000 30%
# 132,000		     35%
rates = [0.0, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
brackets = [0, 14400, 24000, 38400, 63000, 93600, 130800, 4e99]
brackets_month = [x/12 for x in brackets]
df['tax2'] = calc_pit(df['income'].values, rates, brackets_month)
df['avg_tax_rate2'] = df['tax2']/df['income']
# Reform Option 2
# Yearly Income From	Upto	Rate
# 0	        19,200	   0%
# 19,200	48,000	  15%
# 48,000	72,000	  20%
# 72,000	144,000	  32%
# 144,000	4,800,000 37%
# 4,800,000	and above 40%
rates = [0.0, 0.0, 0.15, 0.20, 0.25, 0.32, 0.37, 0.40]
brackets = [0, 19200, 48000, 72000, 108000, 144000, 4800000, 4e99]
brackets_month = [x/12 for x in brackets]
df['tax3'] = calc_pit(df['income'].values, rates, brackets_month)
df['avg_tax_rate3'] = df['tax3']/df['income']
plt.figure(figsize=(10, 5))
plt.plot(df["income"], df['avg_tax_rate1'], label="Current")
plt.plot(df["income"], df['avg_tax_rate2'], label="Option 1")
plt.plot(df["income"], df['avg_tax_rate3'], label="Option 2")
plt.xlabel("Monthly Income")
plt.ylabel("%")
plt.title("")
#plt.title("Average Tax Rate versus Monthly Income")
plt.legend()
plt.grid(False)
plt.show()

#plot_lorenz_curve(df['income'], "Lorenz Curve")
plot_kakwani_lorenz_curve(df['income'], df['tax'], "")
#plot_kakwani_lorenz_curve_reform(df['income'], df['tax'], df['tax2'], "Concentration Curve Current Tax", "Concentration Curve Tax Policy Option 1", "")
plot_kakwani_lorenz_curve_reform(df['income'], df['tax'], df['tax2'], 
                                 "Concentration Curve Current Tax", 
                                 "Concentration Curve Option 1", "",
                                 df['tax3'], "Concentration Curve Option 2")

