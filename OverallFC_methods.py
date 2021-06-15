# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:24:00 2021

Methods to asses differences in network metrics when groups differ in Overall FC.
Overall FC = mean correlations within Functional Connectivity matrices.
Methods were proposed in [1]

[1] van den Heuvel, M. P., de Lange, S. C., Zalesky, A., Seguin, C., Yeo, B. T., 
& Schmidt, R. (2017). Proportional thresholding in resting-state fMRI functional 
connectivity networks and consequences for patient-control connectome 
studies: Issues and recommendations. Neuroimage, 152, 437-449.


@author: Carlos Coronel
"""

import numpy as np
from scipy import stats
import statsmodels.api as sm


def linear_correction(Y,X):
    '''
    Substract the effect of Overall FC from graph metrics, starting from a linear regression
    between the Overall FC and metrics. 
    
    Parameters
    ----------
    Y : list.
        network's metric of groups 1, 2, 3...
    X : list.
        overall FC of groups 1, 2, 3...       
    Returns
    -------
    Y_res : list.
             network's metric of groups 1, 2, 3... corrected for Overall FC.
             
    '''

    ###Linear regression      
    #For the regression, groups were appended to obtain a commmon slope.
    Y_pool = Y[0]
    X_pool = X[0]    
    for i in range(1,len(Y)):
        Y_pool = np.append(Y_pool,Y[i])
        X_pool = np.append(X_pool,X[i])       
    X_pool = sm.add_constant(X_pool) #this add the intercept
    lr_model = sm.OLS(Y_pool,X_pool) 
    lr_fitt = lr_model.fit() #model fitting
    a,b = lr_fitt.params #a: intercept, b: slope
    
    #here we substract the effect of overall FC from the metrics
    Y_res = []
    for i in range(0,len(Y)):
        Y_res.append(Y[i] - b * X[i])
    
    return(Y_res)    
    

def permutation_test(Y1,Y2,permutations,parametric=True,paired=False):
    
    '''
    Simple Permutation test to analyzed if the measured difference between two groups is statistical
    significant. The permutation test randomly selects subjects from the two groups. After,
    the statistic for between groups difference was computed. When repeating this procedure X number of
    times, we can generate a cdf distribution of the statistic. Finally, we compare the real statistic with the
    cdf distribution.
    
    Parameters
    ---------
    Y1 : numpy array.
         network's metric of group 1. Matrix of n x m, with n variables and m observations.
    Y2 : numpy array.
         network's metric of group 2. Matrix of n x m, with n variables and m observations  
    permutations : integer.
                   number of permutations to perform.
    parametric : boolean.
                 True for using parametric tests (t test), False for non parametric tests
                 (wilcoxon, mann-whitney).
    paired : boolean.
             True for paired tests, False for unpaired tests.
             
    Returns
    ------
    t_real : list.
             It contains the real statistics for between-grops difference.
    mu : list.
         mean values of the surrogate statistics for each variable.
    p_val : list.
            p-values associated to the real statistic and the cdf distribution of surrogates.
    surrogates_mean : list.
                      arrays with the mean difference of metrics for each surrogate.
    surrogates_std : list.
                     arrays with the std of the difference for each surrogate.
    '''
    
    
    
    ###Permutation test
    variables = Y1.shape[0] #number of variables
    statistic = np.zeros((variables,permutations)) #matrix for saving statistics values
    surrogates_mean = np.zeros((variables,permutations)) #matrix with the mean difference of metrics for each surrogate
    surrogates_std = np.zeros((variables,permutations)) #matrix with the std of the difference for each surrogate
    observations = np.min([Y1.shape[1],Y2.shape[1]]) #number of subjects/observations
    t_real = np.zeros(variables) #real statistic for variables
    mu = np.zeros(variables) #mean value of the surrogate statistics for each variable
    p_val = np.zeros(variables) #p-values for each variable
    
    #permutations
    for i in range(0,permutations):
        np.random.seed(i)
        
        #surrogate groups for variables
        surr_Y1, surr_Y2 = np.zeros((variables,observations)), np.zeros((variables,observations))
        
        #Random selection of subjects from groups 1 and 2. Both surrogate groups have 50% of the subjects
        #of the original data. Paired == True considered that surr_X1[i] and surr_X2[j] belongs to the 
        #same subject. Paired == False completely randomize the groups.
        
        
        surr_idx = np.arange(0,observations,1)
        np.random.shuffle(surr_idx)
        even = np.arange(0,observations,2)
        odd = np.arange(1,observations,2)
                    
        #variables
        surr_Y1[:,even] = Y1[:,surr_idx[even]]
        surr_Y1[:,odd] = Y2[:,surr_idx[odd]]
        surr_Y2[:,even] = Y2[:,surr_idx[even]]
        surr_Y2[:,odd] = Y1[:,surr_idx[odd]]

    
        #Calculate the surrogate statistic for between-groups difference
        for var in range(0,variables):
        
            if (parametric == True) &  (paired == False):
                statistic[var,i] = stats.ttest_ind(surr_Y1[var,:], surr_Y2[var,:])[0]
            elif (parametric == True) &  (paired == True):
                statistic[var,i] = stats.ttest_rel(surr_Y1[var,:], surr_Y2[var,:])[0]
            elif (parametric == False) &  (paired == False):
                statistic[var,i] = stats.mannwhitneyu(surr_Y1[var,:], surr_Y2[var,:])[0]
            elif (parametric == False) &  (paired == True):
                statistic[var,i] = stats.wilcoxon(surr_Y1[var,:], surr_Y2[var,:])[0]            

            surrogates_mean[var,i] = np.mean(surr_Y1[var,:]-surr_Y2[var,:])
            surrogates_std[var,i] = np.std(surr_Y1[var,:]-surr_Y2[var,:])

    #Calculate the real statistic for between-groups difference            
    for var in range(0,variables):            
       
        if (parametric == True) &  (paired == False):    
            t_real[var] = stats.ttest_ind(Y1[var,:],Y2[var,:])[0]
        elif (parametric == True) &  (paired == True):    
              t_real[var] = stats.ttest_rel(Y1[var,:],Y2[var,:])[0]
        elif (parametric == False) &  (paired == False):    
              t_real[var] = stats.mannwhitneyu(Y1[var,:],Y2[var,:])[0]
        elif (parametric == False) &  (paired == True):    
              t_real[var] = stats.wilcoxon(Y1[var,:],Y2[var,:])[0]   
        
        ###generate the cdf
        mu[var]  = np.mean(statistic[var])
        #sort the data:
        data_sorted = np.sort(statistic[var])
        
        #calculate the proportional values of samples
        p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)    
        
        #Evaluate is the real statistic falls in the critical region of the cdf
        if t_real[var] > mu[var]:
            p_val[var] = 1 - p[np.argwhere(t_real[var] > data_sorted)[-1]]              
        else:
            p_val[var] = p[np.argwhere(t_real[var] < data_sorted)[0]]           
    
    return([t_real,mu,p_val,statistic,surrogates_mean,surrogates_std])


def overallFC_permutation_test(Y1,Y2,X1,X2,permutations,iters,tolerance,
                               parametric=True,paired=False):
    
    '''
    Permutation test to discard the effect of overall FC from between groups differences in topology.
    First, the permutation test randomly selects subjects from the two groups. Then, the code
    switch subject between groups until the original overall FC difference was achieved. After,
    the statistic for between groups difference was computed. When repeating this procedure X number of
    times, we can generate a cdf distribution of the statistic. Finally, we compare the real statistic with the
    cdf distribution.
    
    Parameters
    ---------
    Y1 : numpy array.
         network's metric of group 1. Matrix of n x m, with n variables and m observations.
    Y2 : numpy array.
         network's metric of group 2. Matrix of n x m, with n variables and m observations
    X1 : numpy array.
         overall FC of group 1.
    X2 : numpy array.
         overall FC of group 2.    
    permutations : integer.
                   number of permutations to perform.
    iters : integer.
            number of iteration to reach the tolerance interval for difference in overall FC.
    tolerance : float.
                proportion of the real difference in overall FC to use as a tolerance interval:
                delta = tolerance * difference in overall FC
                tolerance interval: delta - difference < difference < delta + difference
    parametric : boolean.
                 True for using parametric tests (t test), False for non parametric tests
                 (wilcoxon, mann-whitney).
    paired : boolean.
             True for paired tests, False for unpaired tests.
             
    Returns
    ------
    t_real : list.
             It contains the real statistics for between-grops difference.
    mu : list.
         mean values of the surrogate statistics for each variable.
    p_val : list.
            p-values associated to the real statistic and the cdf distribution of surrogates.
    surrogates_mean : list.
                      arrays with the mean difference of metrics for each surrogate.
    surrogates_std : list.
                     arrays with the std of the difference for each surrogate.
    '''
    
    
    
    ###Permutation test
    variables = Y1.shape[0] #number of variables
    real_difference = np.mean(X1-X2) #real difference in overall FC
    statistic = np.zeros((variables,permutations)) #matrix for saving statistics values
    surrogates_mean = np.zeros((variables,permutations)) #matrix with the mean difference of metrics for each surrogate
    surrogates_std = np.zeros((variables,permutations)) #matrix with the std of the difference for each surrogate
    observations = np.min([Y1.shape[1],Y2.shape[1]]) #number of subjects/observations
    t_real = np.zeros(variables) #real statistic for variables
    mu = np.zeros(variables) #mean value of the surrogate statistics for each variable
    p_val = np.zeros(variables) #p-values for each variable
    
    #permutations
    for i in range(0,permutations):
        np.random.seed(i)
        
        #surrogate groups for variables
        surr_Y1, surr_Y2 = np.zeros((variables,observations)), np.zeros((variables,observations))
        #surrogate groups for overall FC
        surr_X1, surr_X2 = np.zeros(observations), np.zeros(observations)       
        
        #Random selection of subjects from groups 1 and 2. Both surrogate groups have 50% of the subjects
        #of the original data.
        
        surr_idx = np.arange(0,observations,1)
        np.random.shuffle(surr_idx)
        even = np.arange(0,observations,2)
        odd = np.arange(1,observations,2)
        
        #overall FC
        surr_X1[even] = X1[surr_idx[even]]
        surr_X1[odd] = X2[surr_idx[odd]]
        surr_X2[even] = X2[surr_idx[even]]
        surr_X2[odd] = X1[surr_idx[odd]]
        
        #variables
        surr_Y1[:,even] = Y1[:,surr_idx[even]]
        surr_Y1[:,odd] = Y2[:,surr_idx[odd]]
        surr_Y2[:,even] = Y2[:,surr_idx[even]]
        surr_Y2[:,odd] = Y1[:,surr_idx[odd]]
        
    
        actual_difference = np.mean(surr_X1) - np.mean(surr_X2) #actual difference in overall FC (surrogates)
        delta = real_difference * tolerance #used for the tolerance interval
        for j in range(0,iters):
            
            #lower and upper bounds of the tolerance interval
            min_lim = np.min([real_difference - delta, real_difference + delta])
            max_lim = np.max([real_difference - delta, real_difference + delta])    
            if not (min_lim < actual_difference) & (max_lim > actual_difference):
                
                #selection of a random pair of subjects
                swap_index_1 = np.random.randint(0,observations,1)
                swap_index_2 = np.random.randint(0,observations,1)                   
                
                #evaluate the direction of change: from surrogate group 1 to 2, or viceversa
                if (actual_difference > real_difference) & ((surr_X1[swap_index_1] - surr_X2[swap_index_2]) > 0):
                    surr_X1[swap_index_1], surr_X2[swap_index_2] = surr_X2[swap_index_2], surr_X1[swap_index_1]
                    surr_Y1[:,swap_index_1], surr_Y2[:,swap_index_2] = surr_Y2[:,swap_index_2], surr_Y1[:,swap_index_1]
                elif (actual_difference < real_difference) & ((surr_X1[swap_index_1] - surr_X2[swap_index_2]) < 0):
                    surr_X1[swap_index_1], surr_X2[swap_index_2] = surr_X2[swap_index_2], surr_X1[swap_index_1]
                    surr_Y1[:,swap_index_1], surr_Y2[:,swap_index_2] = surr_Y2[:,swap_index_2], surr_Y1[:,swap_index_1]  
                else:
                    continue                                                                                     
            else:
                break #is the actual difference is within the tolerance interval, the cycle breaks
            actual_difference = np.mean(surr_X1-surr_X2)
        
        #Calculate the surrogate statistic for between-groups difference
        for var in range(0,variables):
        
            if (parametric == True) &  (paired == False):
                statistic[var,i] = stats.ttest_ind(surr_Y1[var,:], surr_Y2[var,:])[0]
            elif (parametric == True) &  (paired == True):
                statistic[var,i] = stats.ttest_rel(surr_Y1[var,:], surr_Y2[var,:])[0]
            elif (parametric == False) &  (paired == False):
                statistic[var,i] = stats.mannwhitneyu(surr_Y1[var,:], surr_Y2[var,:])[0]
            elif (parametric == False) &  (paired == True):
                statistic[var,i] = stats.wilcoxon(surr_Y1[var,:], surr_Y2[var,:])[0]            

            surrogates_mean[var,i] = np.mean(surr_Y1[var,:]-surr_Y2[var,:])
            surrogates_std[var,i] = np.std(surr_Y1[var,:]-surr_Y2[var,:])

    #Calculate the real statistic for between-groups difference            
    for var in range(0,variables):            
       
        if (parametric == True) &  (paired == False):    
            t_real[var] = stats.ttest_ind(Y1[var,:],Y2[var,:])[0]
        elif (parametric == True) &  (paired == True):    
              t_real[var] = stats.ttest_rel(Y1[var,:],Y2[var,:])[0]
        elif (parametric == False) &  (paired == False):    
              t_real[var] = stats.mannwhitneyu(Y1[var,:],Y2[var,:])[0]
        elif (parametric == False) &  (paired == True):    
              t_real[var] = stats.wilcoxon(Y1[var,:],Y2[var,:])[0]   
        
        ###generate the cdf
        mu[var]  = np.mean(statistic[var])
        #sort the data:
        data_sorted = np.sort(statistic[var])
        
        #calculate the proportional values of samples
        p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)    
        
        #Evaluate is the real statistic falls in the critical region of the cdf
        if t_real[var] > mu[var]:
            p_val[var] = 1 - p[np.argwhere(t_real[var] > data_sorted)[-1]]              
        else:
            p_val[var] = p[np.argwhere(t_real[var] < data_sorted)[0]]           
    
    return([t_real,mu,p_val,statistic,surrogates_mean,surrogates_std])



#%%
###Example

if __name__=='__main__':    

    #This is an example. We have two groups (1 and 2). Global efficiency is higher in group 1 (0.15 vs 0.1)
    #but there is an effect of the overall FC on global efficiency. We want to divorce this effect from the
    #integration metric.     

    observations =  65
    np.random.seed(0)
    Overall_FC_Group_1 = np.random.uniform(0.25,0.45,observations)
    Overall_FC_Group_2 = np.random.uniform(0.35,0.55,observations)
    
    GE_Group_1 = (0.15 + 1.5 * Overall_FC_Group_1) + np.random.normal(0,0.12,observations)
    GE_Group_2 = (0.1 + 1.5 * Overall_FC_Group_2) + np.random.normal(0,0.12,observations)
    GE_Group_1 = GE_Group_1.reshape((1,observations))
    GE_Group_2 = GE_Group_2.reshape((1,observations))    
    
    ###t-test
    #When performing a standard t test without correction for overall FC, global efficiency was higher in group 2.
    print(stats.ttest_ind(GE_Group_1.T,GE_Group_2.T))
    
    ###Linear correction
    GE_Group_1_res, GE_Group_2_res = linear_correction([GE_Group_1,GE_Group_2],
                                                       [Overall_FC_Group_1,Overall_FC_Group_2])
    
    #Adjusting the global efficiency of both groups, using the linear correction, reveals that global efficiency was
    #higher in group 1
    print(stats.ttest_ind(GE_Group_1_res.T,GE_Group_2_res.T))
    
    #Note: instead of using a t-test and the corrected metrics, you can also use the permutation test
    #without the correction for Overall FC (you already corrected with the linear correction). This
    #approach is recommended when groups have a small sample size.
    ###Simple Permutation test
    t_real,mu,p_val,statistic,means,surr_std = permutation_test(GE_Group_1_res,GE_Group_2_res,
                                                                50000,parametric=True,paired=False)      
    print(t_real,mu,p_val)
    

    ###Permutation test modified for discard the effect of the Overall FC
    t_real,mu,p_val,statistic,means,surr_std = overallFC_permutation_test(GE_Group_1,GE_Group_2,
                                                                          Overall_FC_Group_1,
                                                                          Overall_FC_Group_2,
                                                                          10000,1000,0.01,
                                                                          parametric=True,paired=False)  
    
    #This permutation test also suggests that global efficiency was higher in group 1.
    #The real t statistic was higher than the average statistic overall surrogates,
    #and the p-val was also significant.
    print(t_real,mu,p_val)

   


