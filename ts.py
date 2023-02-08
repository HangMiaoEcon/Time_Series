#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 22:43:38 2022

@author: Hang Miao
"""
#%%
import os
os.chdir('/Users/yeminlan/Desktop/ts')
##### Simulate ARMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_process import ArmaProcess
import arma
#%%
############################################################
## AR(1) and AR(2) process
############################################################
# simulation AR processes
# AR(1)
n = 1000
arprams = [0.7]
ar1 = arma.sim_AR(arprams,n)
# AR(2)  Unit Root non stationary
arprams = [0.7,0.3]
ar2 = arma.sim_AR(arprams,n)
# AR(2)
arprams = [0.7,0.2]
ar2 = arma.sim_AR(arprams,n)


# Model Selection
# ACF
acf_ar1= arma.ACF_arma(ar1)     # tails off: gradually diminishing
acf_ar2= arma.ACF_arma(ar2)     # tails off: gradually diminishing
# PACF
pacf_ar1= arma.PACF_arma(ar1)   # Break/cut off: stop suddenly at 1st lag
pacf_ar2= arma.PACF_arma(ar2)   # Break/cut off: stop suddenly at 2nd lag

# Estimation
res1 = arma.estimation_arma(ar1,[1,0,0] )
res2 = arma.estimation_arma(ar2,[2,0,0] )

# Model Diagnostics
res1.plot_diagnostics()
res2.plot_diagnostics()
#%%
############################################################
## MA(1) and MA(2) process
############################################################
# simulation MA processes
# MA(1)
n = 10000
maprams = [0.7]
ma1 = arma.sim_MA(maprams,n)
# MA(2)
arprams = [0.7,0.3]
ma2 = arma.sim_MA(arprams,n)

# Model Selection
# ACF
acf_ar1= arma.ACF_arma(ma1)     # Break/cut off: stop suddenly at 1st lag
acf_ar2= arma.ACF_arma(ma2)     # Break/cut off: stop suddenly at 2nd lag
# PACF
pacf_ar1= arma.PACF_arma(ma1)   # tails off: gradually diminishing
pacf_ar2= arma.PACF_arma(ma2)   # tails off: gradually diminishing

# Estimation
res1 = arma.estimation_arma(ma1,[0,0,1] )
res2 = arma.estimation_arma(ma2,[0,0,2] )

# Model Diagnostics
res1.plot_diagnostics()
res2.plot_diagnostics()

#%%
############################################################
## ARMA process
############################################################
import arma
# simulation ARMA processes
n = 10000
# ARMA(1,1)
arprams = [0.7]
maprams = [0.3]
arma1 = arma.sim_ARMA(arprams,maprams,n)
# ARMA(2,1)
arprams = [0.7,0.2]
maprams = [0.3,0.3]  # 0.3 0.1 can't be detected from AIC and BIC since the impact is too small
arma2 = arma.sim_ARMA(arprams,maprams,n)

# Model Selection
# AIC  
# stepwise_: False go over all combination
# stepwise_: True default fast algorithm
res1_aic = arma.AIC_arma(arma1, 5,5, stepwise=False)  
res2_aic = arma.AIC_arma(arma2, 5,5, stepwise=False)
# BIC
res1_bic = arma.BIC_arma(arma1, 5,5, stepwise=False)
res2_bic = arma.BIC_arma(arma2, 5,5, stepwise=False)
# ACF and PACF can no longer effectively select the p,q order
# ACF
acf_ar1= arma.ACF_arma(arma1)     # tails off: gradually diminishing
acf_ar2= arma.ACF_arma(arma2)     # tails off: gradually diminishing
# PACF
pacf_ar1= arma.PACF_arma(arma1)   # tails off: gradually diminishing
pacf_ar2= arma.PACF_arma(arma2)   # tails off: gradually diminishing

# Estimation
#res1 = arma.estimation_arma(ma1,[0,0,1] )
#res2 = arma.estimation_arma(ma2,[0,0,2] )
res1_aic.summary()
res2_aic.summary()
res1_bic.summary()
res2_bic.summary()

# Model Diagnostics
res1_aic.plot_diagnostics()
res2_aic.plot_diagnostics()
res1_bic.plot_diagnostics()
res2_bic.plot_diagnostics()













