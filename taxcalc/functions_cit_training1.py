"""
Functions that calculate personal income tax liability.
"""
# CODING-STYLE CHECKS:
# pycodestyle functions.py
# pylint --disable=locally-disabled functions.py

import math
import copy
import numpy as np
from taxcalc.decorators import iterate_jit


@iterate_jit(nopython=True)
def calc_Gross_Income(Sales, Other_Income, Gross_Income):
    """
    Compute accounting profit from business
    """
    Gross_Income = Sales + Other_Income
    return Gross_Income

@iterate_jit(nopython=True)
def calc_Gross_Profit(Gross_Income, Cost_of_Goods, Gross_Profit):
    """
    Compute accounting profit from business
    """
    Gross_Profit = Gross_Income - Cost_of_Goods
    return Gross_Profit

@iterate_jit(nopython=True)
def calc_Net_Income(Gross_Profit, Total_Expense, Net_Income):
    """
    Compute accounting profit from business
    """
    Net_Income = Gross_Profit - Total_Expense
    return Net_Income

@iterate_jit(nopython=True)
def Op_WDV_depr(Op_WDV_Bld, Op_WDV_Intang, Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp):
    """
    Return the opening WDV of each asset class.
    """
    Op_WDV_Bld, Op_WDV_Intang, Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp = (Op_WDV_Bld, 
    Op_WDV_Intang, Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp)
    return (Op_WDV_Bld, Op_WDV_Intang, Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp)

@iterate_jit(nopython=True)
def Tax_depr_Bld(Op_WDV_Bld, Add_Bld, Excl_Bld, rate_depr_bld, Tax_depr_Bld):
    """
    Compute tax depreciation of building asset class.
    """
    Tax_depr_Bld = max(rate_depr_bld*(Op_WDV_Bld + Add_Bld - Excl_Bld),0)
    return Tax_depr_Bld

@iterate_jit(nopython=True)
def Tax_depr_Intang(Op_WDV_Intang, Add_Intang, Excl_Intang, rate_depr_intang, Tax_depr_Intang):
    """
    Compute tax depreciation of intangibles asset class
    """
    Tax_depr_Intang = max(rate_depr_intang*(Op_WDV_Intang + Add_Intang - Excl_Intang),0)
    return Tax_depr_Intang

@iterate_jit(nopython=True)
def Tax_depr_Mach(Op_WDV_Mach, Add_Mach, Excl_Mach, rate_depr_mach, Tax_depr_Mach):
    """
    Compute tax depreciation of Machinary asset class
    """
    Tax_depr_Mach = max(rate_depr_mach*(Op_WDV_Mach + Add_Mach - Excl_Mach),0)
    return Tax_depr_Mach

@iterate_jit(nopython=True)
def Tax_depr_Others(Op_WDV_Others, Add_Others, Excl_Others, rate_depr_others, Tax_depr_Others):
    """
    Compute tax depreciation of Other asset class
    """
    Tax_depr_Others = max(rate_depr_others*(Op_WDV_Others + Add_Others - Excl_Others),0)
    return Tax_depr_Others

@iterate_jit(nopython=True)
def Tax_depr_Comp(Op_WDV_Comp, Add_Comp, Excl_Comp, rate_depr_comp, Tax_depr_Comp):
    """
    Compute tax depreciation of Computer asset class
    """
    Tax_depr_Comp = max(rate_depr_comp*(Op_WDV_Comp + Add_Comp - Excl_Comp),0)
    return Tax_depr_Comp

@iterate_jit(nopython=True)
def Tax_depreciation(Tax_depr_Bld, Tax_depr_Intang, Tax_depr_Mach, Tax_depr_Others, Tax_depr_Comp, Tax_depr):
    """
    Compute total depreciation of all asset classes.
    """
    Tax_depr = Tax_depr_Bld + Tax_depr_Intang + Tax_depr_Mach + Tax_depr_Others + Tax_depr_Comp
    return Tax_depr

@iterate_jit(nopython=True)
def Cl_WDV_depr(Op_WDV_Bld, Add_Bld, Excl_Bld, Tax_depr_Bld, 
                Op_WDV_Intang, Add_Intang, Excl_Intang, Tax_depr_Intang,
                Op_WDV_Mach, Add_Mach, Excl_Mach, Tax_depr_Mach,
                Op_WDV_Others, Add_Others, Excl_Others, Tax_depr_Others,
                Op_WDV_Comp, Add_Comp, Excl_Comp, Tax_depr_Comp,
                Cl_WDV_Bld, Cl_WDV_Intang, Cl_WDV_Mach, Cl_WDV_Others, Cl_WDV_Comp):
    """
    Compute Closing WDV of each block of asset.
    """
    Cl_WDV_Bld = max((Op_WDV_Bld + Add_Bld - Excl_Bld),0) - Tax_depr_Bld
    Cl_WDV_Intang = max((Op_WDV_Intang + Add_Intang - Excl_Intang),0) - Tax_depr_Intang
    Cl_WDV_Mach = max((Op_WDV_Mach + Add_Mach - Excl_Mach),0) - Tax_depr_Mach
    Cl_WDV_Others = max((Op_WDV_Others + Add_Others - Excl_Others),0) - Tax_depr_Others
    Cl_WDV_Comp= max((Op_WDV_Comp + Add_Comp - Excl_Comp),0) - Tax_depr_Comp
    return (Cl_WDV_Bld, Cl_WDV_Intang, Cl_WDV_Mach, Cl_WDV_Others, Cl_WDV_Comp)

@iterate_jit(nopython=True)
def calc_Total_Deductions(Tax_depr, Total_Deductions):
    """
    Compute net taxable profits afer allowing deductions.
    """
    Total_Deductions = Tax_depr-Tax_depr
    return Total_Deductions

@iterate_jit(nopython=True)
def calc_Carried_Forward_Losses(Loss_CF_Prev, Loss_CB_Adj, CF_losses):
    """
    Compute net taxable profits afer allowing deductions.
    """
    CF_losses = Loss_CF_Prev+Loss_CB_Adj
    return CF_losses

@iterate_jit(nopython=True)
def calc_Total_Taxable_Profit(Net_Income, Total_Taxable_Profit):
    """
    Compute total taxable profits afer adding back non-allowable deductions.
    """
    Total_Taxable_Profit = Net_Income
    return Total_Taxable_Profit

@iterate_jit(nopython=True)
def calc_Taxable_Business_Income(Total_Taxable_Profit, Total_Deductions, Taxable_Business_Income):
    """
    Compute net taxable profits afer allowing deductions.
    """
    Taxable_Business_Income = Total_Taxable_Profit - Total_Deductions
    return Taxable_Business_Income

@iterate_jit(nopython=True)
def calc_Tax_base_CF_losses(Taxable_Business_Income, Loss_CFLimit, CF_losses,
    Loss_lag1, Loss_lag2, Loss_lag3, Loss_lag4, Loss_lag5, Loss_lag6, Loss_lag7, Loss_lag8,
    newloss1, newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8, Used_loss_total, Tax_base):
    
    """
    Compute net tax base afer allowing donations and losses.
    """
    BF_loss = np.array([Loss_lag1, Loss_lag2, Loss_lag3, Loss_lag4, Loss_lag5, Loss_lag6, Loss_lag7, Loss_lag8])
    
    Gross_Tax_base = Taxable_Business_Income

    if BF_loss.sum() == 0:
        BF_loss[0] = CF_losses

    N = int(Loss_CFLimit)
    if N == 0:
        (newloss1, newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8) = np.zeros(8)
        Used_loss_total = 0
        Tax_base = Gross_Tax_base
        
    else:
        BF_loss = BF_loss[:N]
                
        if Gross_Tax_base < 0:
            CYL = abs(Gross_Tax_base)
            Used_loss = np.zeros(N)
        elif Gross_Tax_base >0:
            CYL = 0
            Cum_used_loss = 0
            Used_loss = np.zeros(N)
            for i in range(N, 0, -1):
                GTI = Gross_Tax_base - Cum_used_loss
                Used_loss[i-1] = min(BF_loss[i-1], GTI)
                Cum_used_loss += Used_loss[i-1]
        elif Gross_Tax_base == 0:
            CYL=0
            Used_loss = np.zeros(N)
    
        New_loss = BF_loss - Used_loss
        Tax_base = Gross_Tax_base - Used_loss.sum()
        newloss1 = CYL
        Used_loss_total = Used_loss.sum()
        (newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8) = np.append(New_loss[:-1], np.zeros(8-N))

    return (Tax_base, newloss1, newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8, Used_loss_total)


@iterate_jit(nopython=True)
def calc_Net_tax_base(Tax_base, Net_tax_base):
    """
    Compute net tax base afer allowing deductions.
    """
    Net_tax_base = Tax_base
    return Net_tax_base

@iterate_jit(nopython=True)
def calc_Imputed_tax_base(Tax_Due, cit_rate_curr_law, Tax_base_Imputed):
    """
    Compute net tax base afer allowing donations and losses.
    """
    Tax_base_Imputed = Tax_Due/cit_rate_curr_law
    return Tax_base_Imputed

@iterate_jit(nopython=True)
def calc_Net_tax_base_behavior(cit_rate, cit_rate_curr_law, elasticity_cit_taxable_income_threshold,
                                elasticity_cit_taxable_income_value, Net_tax_base, 
                                Net_tax_base_behavior):
    """
    Compute net taxable profits afer allowing deductions.
    """
    NP = Net_tax_base   
    elasticity_taxable_income_threshold0 = elasticity_cit_taxable_income_threshold[0]
    elasticity_taxable_income_threshold1 = elasticity_cit_taxable_income_threshold[1]
    elasticity_taxable_income_threshold2 = elasticity_cit_taxable_income_threshold[2]
    elasticity_taxable_income_value0=elasticity_cit_taxable_income_value[0]
    elasticity_taxable_income_value1=elasticity_cit_taxable_income_value[1]
    elasticity_taxable_income_value2=elasticity_cit_taxable_income_value[2]
    if NP<=0:
        elasticity=0
    elif NP<elasticity_taxable_income_threshold0:
        elasticity=elasticity_taxable_income_value0
    elif NP<elasticity_taxable_income_threshold1:
        elasticity=elasticity_taxable_income_value1
    else:
        elasticity=elasticity_taxable_income_value2

    frac_change_net_of_cit_rate = ((1-cit_rate)-(1-cit_rate_curr_law))/(1-cit_rate_curr_law)
    frac_change_Net_tax_base = elasticity*(frac_change_net_of_cit_rate)    
    Net_tax_base_behavior = Net_tax_base*(1+frac_change_Net_tax_base)
    return Net_tax_base_behavior

DEBUG = False
DEBUG_IDX = 0

@iterate_jit(nopython=True)
def calc_mat_liability(mat_rate, Sales, MAT):
    """
    Compute tax liability given the corporate rate
    """
    # subtract TI_special_rates from TTI to get Aggregate_Income, which is
    # the portion of TTI that is taxed at normal rates
    MAT = mat_rate*max(Sales,0)
        
    return MAT

@iterate_jit(nopython=True)
def calc_cit_liability(cit_rate, MAT, Net_tax_base_behavior, citax):
    """
    Compute tax liability given the corporate rate
    """
    # subtract TI_special_rates from TTI to get Aggregate_Income, which is
    # the portion of TTI that is taxed at normal rates
    taxinc = max(Net_tax_base_behavior, 0)

    citax = cit_rate * taxinc
       
    if MAT>citax:
        citax=MAT
    
    return citax

@iterate_jit(nopython=True)
def calc_cit_net(Total_Credit, citax, citax_net):
    """
    Compute tax liability after incorporating Withholding and Other Credits Paid
    """
    # this is the tax due after Withholding and Other Credits Paid

    citax_net = max((citax - Total_Credit),0)
    
    return citax_net



