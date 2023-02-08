# -*- coding: utf-8 -*-
"""
Created on Wed Nov 7 13:42:53 2022

@author: Hang Miao
"""
import os
os.chdir(r'C:\Users\Hahn\Dropbox\python\ts')
#os.chdir(r'/Users/yeminlan/Dropbox/python/ts')
import arma
from statsmodels.tsa.stattools import levinson_durbin
import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin,pi,exp,log,log10,sqrt
from scipy.stats import norm,uniform
from scipy.signal import fftconvolve
from scipy.fft import fft, ifft, fftfreq, fftshift
from itertools import repeat  # map function with multiple arguments ( parameter and x) where parameters needs to hold the same
from numpy.linalg import eig # Eigen decompostion
def DB (x): return 10*np.log10(x)
def DB_inverse(db): return 10**(db/10)

N = 33
tau = np.arange(0,N+1)
df = 0.0005
f = np.round( np.arange(-0.5,0.5+df,df),4)

#%%

###########
# AR(1)
###########
phi = 0.49
acvs1 = phi**np.abs(np.arange(-N,N+1)  )    # full period, works for fft, not for levinson_durbin     
acvs1 = phi**tau               # half period, works for levinson_durbin, not for fft
                                  
# levinson_durbin algorithm compute AR coefficients from autocovariance sequence
acvs1 = phi**np.arange(0,N+1)    
sigmav, arcoefs, pacf, sigma, phi_ = levinson_durbin(acvs1, nlags=N, isacov=True)
arprams = arcoefs[np.round(arcoefs,4 )>0]
ts1 = arma.sim_AR(arprams,1000)
sigmav, arcoefs, pacf, sigma, phi_ = levinson_durbin(ts1, nlags=10, isacov=False)
# generate the associated AR(1) time series
ts1 = arma.sim_AR(arprams,N+1,sigmav)

# Closed-form SDF of AR(1)
def AR_spectral(f,phi,sigma):
    return sigma**2/(1+phi**2-2*phi*cos(2*pi*f)) 
SDF1 = np.array( list( map(AR_spectral,f,repeat(phi),repeat(sqrt(1-phi**2)))) )
CSDF1 = np.cumsum(SDF1*df)
# FFT Approximation for SDF of AR(1)
acvs1 = phi**np.abs(np.arange(-N,N+1)  )    # full period, works for fft, not for levinson_durbin     
n = len(acvs1); dx = 1; # slice_n = 100 # slice_n for zoom in the center part of G(f)
f_fft = fftshift( fftfreq(n,dx)) # select center slice of fourier freq
SDF1_approx = fftshift(np.abs( fft(acvs1)))
df_f_fft = np.diff(f_fft)[0]
CSDF1_approx = np.cumsum(SDF1_approx*df_f_fft)

# plot the Closed-Form SDF and Approximated FFT of AR(1)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(f,SDF1, c= 'r',label = 'SDF of AR(1)')
axs.plot(f_fft,SDF1_approx,c= 'b',linestyle='--', label = 'FFT approximation for SDF of AR(1)')
axs.legend()
# plot the Closed-Form SDF and Approximated FFT of AR(1)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(f,CSDF1, c= 'r',label = 'CSDF of AR(1)')
axs.plot(f_fft,CSDF1_approx,c= 'b',linestyle='--', label = 'FFT approximation for CSDF of AR(1)')
axs.legend()
#%%
     
###########
# Harmonic
###########
freq =1/16
acvs2 = np.cos(2*pi*freq*tau)     
phi2 = uniform.rvs(-pi,pi,1)                     
ts2 = sqrt(2) * np.cos(2*pi*freq*tau + phi2)   
# Closed-form SDF of Harmonic 
SDF2_fun = lambda x: sqrt(2)/2 if abs(x)==freq else 0
SDF2 = np.array( list(map(SDF2_fun,f )) )
CSDF2 = np.cumsum(SDF2*df)

# FFT Approximation for SDF of Harmonic
acvs2 = np.cos(2*pi*freq*tau )     # full period, works for fft, not for levinson_durbin     
n = len(acvs2); dx = 1; 
f_fft = fftshift( fftfreq(n,dx)) # select center slice of fourier freq
SDF2_approx = fftshift(np.abs( fft(acvs2))/n)  ### /n ??
df_f_fft = np.diff(f_fft)[0]
CSDF2_approx = np.cumsum(SDF2_approx*df_f_fft)

# plot the ACVS of harmonic sequence
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(tau,acvs2, c= 'r',label = 'CSDF of Harmonic')
# plot the Closed-Form SDF and Approximated FFT of AR(1)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(f,SDF2, c= 'r',label = 'SDF of Harmonic')
axs.plot(f_fft,SDF2_approx,c= 'b',linestyle='--', label = 'FFT approximation for SDF of AR(1)')
axs.legend()


############################################
# Harmonic + White Noise
############################################
sigma=1
WN = norm.rvs(0,sigma,N+1)  # white noise
acvs_WN = np.concatenate(([sigma**2], np.zeros(N))) 
acvs3 = 0.5*acvs2 + 0.5*acvs_WN            # Harmonic+ white noise
ts3 = 0.5*ts2 + 0.5*WN

# Closed-form SDF of white noise
SDF3_fun = lambda x: sigma**2
SDF3 = np.array( list(map(SDF3_fun,f )) )
CSDF3 = np.cumsum(SDF3*df)

# FFT Approximation for SDF of Harmonic
acvs_WN = np.cos(2*pi*freq*tau )     # full period, works for fft, not for levinson_durbin     
n = len(WN); dx = 1; 
f_fft = fftshift( fftfreq(n,dx)) # select center slice of fourier freq
SDF2_approx = fftshift(np.abs( fft(acvs2))/n)  ### /n ??
df_f_fft = np.diff(f_fft)[0]
CSDF2_approx = np.cumsum(SDF2_approx*df_f_fft)

# plot the Closed-Form SDF and Approximated FFT of AR(1)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(f,SDF3, c= 'r',label = 'SDF of Harmonic')
axs.plot(f_fft,SDF2_approx,c= 'b',linestyle='--', label = 'FFT approximation for SDF of AR(1)')
axs.legend()

############################################
# Harmonic + AR(1)
############################################
acvs4 = 0.5*acvs2 + 0.5*acvs1                                       # Harmonic+ AR

#%%
############################################
# Aliasing
############################################
f_Max = 4
f_alias = np.round( np.arange(-f_Max,f_Max+df,df),4)
SDF_fun = lambda f: 1.9*np.exp(-2*f**2) + np.exp(-6*(np.abs(f) - 1.25)**2)
SDF = np.array( list(map(SDF_fun,f_alias )) )


dt = 1/4;f_Max = 1/(2*dt);k_max = 30 # dt = 1/8
SDF_fun_approx = lambda f: sum( SDF_fun(f + np.arange(-k_max,k_max+1)/dt) )
SDF_approx = np.array( list(map(SDF_fun_approx,f_alias )) )
 # plot the Closed-Form SDF and Approximated FFT of AR(1)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(f_alias,SDF, c= 'r',label = 'Aliasing')
axs.plot(f_alias,SDF_approx,c= 'b',linestyle='--', label = 'FFT approximation for SDF of AR(1)')
axs.legend()
#%%
############################################
# Spectral density of ARMA
############################################
# Closed form SDF of AR(4) as a special case
f_ = np.round( np.arange(-0.5,0.5+df,df),4)
sd_ = sqrt(0.002)
AR_coef = np.array([2.7607, -3.8106, 2.6535, -0.9238] )
SDF = np.array( list(map(spectral_ARMA,f_, repeat(AR_coef),repeat([]), repeat(sd_) )) ) 

# FFT Approximation for SDF of AR(4) only
n = 200; dx = 1; 
f_fft = fftshift( fftfreq(n,dx)) 
SDF_apprx = fftshift(sd_**2/np.abs( fft( np.r_[1,-AR_coef,np.zeros(n-len(AR_coef)-1)] ))**2)

# plot the Closed-Form SDF and Approximated FFT of AR(1)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(f_,SDF, c= 'r',label = 'SDF of AR(4)')
axs.plot(f_fft,SDF_apprx,c= 'b',linestyle='--', label = 'FFT approximation for SDF of AR(1)')
axs.legend()

##################################################
# not right uisng 3rd party package
# default sigma^2 =1 can't change. Spectral density change over N
N = 2000
from statsmodels.tsa.arima_process import arma_periodogram
ar = np.r_[1, -AR_coef]
freq,SDF = arma_periodogram(ar ,[1],N,whole=0)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
axs.plot(freq,SDF/N, c= 'r',label = 'SDF of Harmonic')




















