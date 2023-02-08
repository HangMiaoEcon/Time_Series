#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:49:34 2022

@author: Hang
"""
import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin,pi,exp,log,log10,sqrt
from scipy.stats import norm
from scipy.signal import fftconvolve
from scipy.fft import fft, ifft, fftfreq, fftshift
from itertools import repeat  # map function with multiple arguments ( parameter and x) where parameters needs to hold the same
from numpy.linalg import eig # Eigen decompostion
def DB (x): return 10*np.log10(x)
def DB_inverse(db): return 10**(db/10)
#%%
### Harmonic function
x = np.linspace(-3.5,3.5,1000)
y = 4*(np.cos(pi*x))**6 + np.sin(10*pi*x)**2


fig = plt.figure()
ax = plt.axes()
ax.plot(x, y);
ax.axhline(y = 1, color = 'r', linestyle = '-')

#%%
### Spectral density of AR(1) with variance 1-phi**2

# Original Function g(t): Related to Spectral function of AR(1). Deterministic AR(1)
def Spectral_AR1 (t,phi = 0.9):
    return (1-phi**2)/(1+phi**2-2*phi*np.cos(t))

# Fourier Approximation pf g(t) using Fourier coefficient G(n). 
def Spectral_AR1_approx (t,m=4,phi=0.9):
    ns = np.arange(1,m+1)
    return  1 + 2 * (phi**ns).dot(np.cos(np.outer(ns,t)))

x = np.linspace(-3.5,3.5,1000)
y = Spectral_AR1(x,phi = 0.9)
fig = plt.figure()
ax = plt.axes()
ax.plot(x, y)
for m in np.arange(1,5):
    y = Spectral_AR1_approx (x,2**m,phi=0.9)
    ax.plot(x, y)
#%%
# Discrete Power Spectrum S_n =  |G(n)|^2 = (phi^|n|)^2 = phi^(2|n|)
n = 32
ns = np.arange(-n,n+1)
Sn = phi**(2*abs(ns))
Sn_decibel = 10*np.log10(Sn)

# figure size
gr = 1.618 # Golden Ratio
width = 10
height = width/gr  # 6.18

# plot
fig, axs = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('Discrete Power Spectrum')
axs[0].scatter(ns, Sn)
axs[0].set_title('Arithmetic scale')
axs[1].scatter(ns, Sn_decibel)
axs[1].set_title('Decibel Scale')


#%%
# convolution

# ex1
x = np.linspace(-3.5,3.5,1000)
g_fun = lambda x: 3/4 if abs(x)<3/4 else 0
h_fun = lambda x: 1 if abs(x)<1/2 else 0
e_fun = lambda x: 3/4 if abs(x)<1/2 else 0
f_fun = lambda x: 1 if abs(x)<1/2 else 0


# method 1
g = np.array( list(map(g_fun,x )) )
h = np.array( list(map(h_fun,x )) )
e = np.array( list(map(e_fun,x )) )
f = np.array( list(map(f_fun,x )) )

# method 2 does not work in this case: for if else condition
#y_fun_vec = np.vectorize(y_fun) 
#y = y_fun_vec(x)

# convolution
convolve_gh = np.convolve(g,h,'same' )
convolve_ef = np.convolve(e,f,'same' )

# plot
fig, axs = plt.subplots(2, 2, figsize=(20,12))
fig.suptitle('Convolution')
axs[0,0].plot(x, g, label='g(t)')
axs[0,0].plot(x, h, label='h(t)')
axs[0,0].legend()
axs[1,0].plot(x, convolve_gh, label='g(t)*h(t)')

axs[0,1].plot(x, e, label='e(t)')
axs[0,1].plot(x, f, label='f(t)')
axs[0,1].legend()
axs[1,1].plot(x, convolve_ef, label='g(t)*h(t)')

#%%
# Application of convolution: Smoothing Operation
# signal
f1 = 1/6; f2=3
x = np.linspace(-4,4,1000)
dx = 8/1000
h_fun = lambda x: 5*cos(2*pi*f1*x+0.5) + cos(2*pi*f2*x +1.1 )
h = np.array( list(map(h_fun,x )) )

# Gaussian filter
sigma = 0.25
g_fun = lambda x: norm.pdf(x,0,sigma)
g = np.array( list(map(g_fun,x )) )
# Square filter
delta = 1/4  # 1/8 1/6 1/4
g_fun = lambda x: 1/(2*delta) if abs(x)<delta else 0
g = np.array( list(map(g_fun,x )) )

# Convoluted signal
# using convolution function
convolve_gh = np.convolve(g,h,'same' )* dx  
# using closed form solution: Gaussian filter
convolve_gh_fun = lambda x: 5*cos(2*pi*f1*x+0.5)*exp(  -(sigma*2*pi*f1)**2/2 ) + cos(2*pi*f2*x +1.1 )*exp(  -(sigma*2*pi*f2)**2/2 )
convolve_gh = np.array( list(map(convolve_gh_fun,x )) )
# using closed form solution: Square filter
convolve_gh_fun = lambda x: 5*cos(2*pi*f1*x+0.5)*np.sinc(delta*2*f1) + cos(2*pi*f2*x +1.1 )*np.sinc(delta*2*f2)
convolve_gh = np.array( list(map(convolve_gh_fun,x )) )

# plot
fig, axs = plt.subplots(2, 1, figsize=(20,12))
fig.suptitle('Convolution Application')
axs[0].plot(x, h, label='signal h(t)')
axs[0].plot(x, convolve_gh, label='g(t)*h(t)')
axs[0].legend()
axs[1].plot(x, g, label='Gaussian filter g(t)')
#axs[1].set_title('Arithmetic scale')
axs[1].legend()

#%%
# width: equivalent, variance, autocorrelation  
x = np.linspace(-4.5,4.5,1000)
dx = 9/1000
# Square wave
w_half = 1  #(sqrt(3)+sqrt(pi))/2
g_fun = lambda x: 1/(2*w_half) if abs(x)<w_half else 0
s = np.array( list(map(g_fun,x )) )
var = (2*w_half)**2/12
W_e = 2*w_half
W_v = 2*sqrt(3)*sqrt(var)
W_a = 1/ ((1/(2*w_half))**2*2*w_half )
# Gaussian wave
sigma = sqrt(1/3)
g_fun = lambda x: norm.pdf(x,0,sigma)
g = np.array( list(map(g_fun,x )) )

W_e_g = 1/norm.pdf(0,0,sigma)
W_v_g = 2*sqrt(3)*sigma
W_a_g = 1/ (sum(g*g)*dx) # approximation
W_a_g = 2*sigma*sqrt(pi) # closed

# plot
fig, axs = plt.subplots(2, 1, figsize=(20,12))
fig.suptitle('Width of Signal')
axs[0].plot(x, s, label='Square Wave')
axs[0].axvline(x = W_e/2, color = 'r',linestyle='--' ,label = 'Equivalent')
axs[0].axvline(x = -W_e/2, color = 'r',linestyle='--' )
axs[0].axvline(x = W_v/2, color = 'r',linestyle=':' ,label = 'Variance')
axs[0].axvline(x = -W_v/2, color = 'r',linestyle=':' )
axs[0].axvline(x = W_a/2, color = 'r',linestyle='-.' ,label = 'Autocrrelation')
axs[0].axvline(x = -W_a/2, color = 'r',linestyle='-.' )
axs[0].legend()

axs[1].plot(x, g, label='Gaussian Wave')
axs[1].axvline(x = W_e_g/2, color = 'r',linestyle='--' ,label = 'Equivalent')
axs[1].axvline(x = -W_e_g/2, color = 'r',linestyle='--' )
axs[1].axvline(x = W_v_g/2, color = 'r',linestyle=':' ,label = 'Variance')
axs[1].axvline(x = -W_v_g/2, color = 'r',linestyle=':' )
axs[1].axvline(x = W_a_g/2, color = 'r',linestyle='-.' ,label = 'Autocrrelation')
axs[1].axvline(x = -W_a_g/2, color = 'r',linestyle='-.' )
axs[1].legend()

#%%
# Discrete Sampling
x_cont = np.arange(-70,70+0.001,0.001)      # fine meshgrid of time domain
x_disc = np.arange(-70,70+1)                # coarse meshgrid of time domain
f_cont = np.arange(-0.5,0.5+0.001,0.001)    # fine meshgrid of frequency domain
#closed-form signal function g(t)
g_fun = lambda t: 2*sqrt(pi)*exp(-(pi*t)**2/10000)*(cos(pi*0.46*t) + cos(pi*0.54*t))/100
g_cont = np.array(list(map(g_fun,x_cont)))  # continuous approximation of g(t) using fine mesh grid
g_disc = np.array(list(map(g_fun,x_disc)))  # discrete approximation of g(t*dt) using coarse mesh grid

#closed-form Fourier transform function G(f)  Twin Peak
G_fun = lambda f: exp(-10000*(f-0.23)**2) + exp(-10000*(f+0.23)**2) + exp(-10000*(f-0.27)**2) + exp(-10000*(f+0.27)**2)
G_cont = np.array(list(map(G_fun,f_cont)))  # continuous approximation of G(f) using fine mesh grid
#Approximated Fourier transform function G(f) using discrete FFT
N = len(x_cont); dx = 0.001; slice_n = 100 # slice_n for zoom in the center part of G(f)
f_fft = np.concatenate((fftfreq(N,dx)[-slice_n:-1],fftfreq(N,dx)[0:slice_n])) # select center slice of fourier freq
G_fft = fft(g_cont)
G_fft = np.concatenate((G_fft[-slice_n:-1],G_fft[0:slice_n]))  # select center slice of fourier coefficient
# note G_ftt is complex number. Use np.abs() to compute the module of complex and times dx
#Approximated Fourier transform function G(f) using discrete FT, partial sum
def Gp_ft(m,f):
    t = np.arange(-m,m+1)       # fixed time interval
    dt = 1                      # fixed sampling freq
    g_t = np.array(list(map(g_fun,t)))    
    return dt*sum(g_t*np.exp( -2j*pi*f*t*dt))
m = 16 # 4, 16, 64
G_ft = np.array(list(map(Gp_ft,repeat(m),f_cont)))

# plot
fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Discrete Sampling')
axs.plot(x_cont,g_cont,linestyle ='-', color='b' )
axs.scatter(x_disc,g_disc, color='r' )

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Closed-form Fourier Transform G(f)')
axs.plot(f_cont,G_cont,linestyle ='-', color='b' )

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('DFT Approximation of G(f)')
axs.plot(f_fft,np.abs(G_fft )*dx,linestyle =':')
axs.plot(f_cont,G_cont,linestyle ='--', color='r' )

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Partial sum FT Approximation of G(f)')
axs.plot(f_cont,np.real(G_ft ),linestyle =':')  ## Stein(I) P59 Ex2, f is real and even, no complex part
axs.plot(f_cont,G_cont,linestyle ='--', color='r' )


#%%
# Dirichlet Kernel
def dirichlet_kernel(N,x):  # x in (-pi, pi)
    if x==0:
        2*N+1
    else: return sin(N*x+x/2 )/sin(x/2)
def D_cal(m,f):  # D_{2m+1}(f) f in (-0.5, 0.5)
    return dirichlet_kernel(m,2*pi*f)/(2*m+1)
# Fejer Kernel
def Fejer_kernel(N,x):  # x in (-pi, pi)
    if x==0:
        return N
    else: return 1/N*sin(N*x/2)**2/sin(x/2)**2
def Fejer_kernel_cal(m,f):  # D_{m}^2(f) /m, f in (-0.5, 0.5)
    return Fejer_kernel(m,2*pi*f)/m

x = np.linspace(-0.5,0.5,1000)
m=16  # 4, 16, 64
d = np.array( list(map(D_cal,repeat(m), x)))  # repeat from itertools
fejer = np.array( list(map(Fejer_kernel_cal,repeat(m), x)))  # repeat from itertools

fig, axs = plt.subplots(1, 1, figsize=(20,12))
# plot Dirichlet Kernel
axs.plot(x,d, c= 'r',label = 'Dirichlet Kernel and Dcal')
# plot Dirichlet Kernel cal
axs.plot(x,fejer,c= 'b', label = 'Fejer Kernel and Dcal^2')
axs.legend()

#%%
# signal g_t with square wave as fourier coefficient
        
#closed-form signal function g(t)
g_fun = lambda t: 0.5 if t==0 else [0,1,0,-1][abs(t)%4]/(pi*abs(t))
t = np.arange(-70,70+1)       
g_t = np.array(list(map(g_fun,t)))  
#closed-form Fourier transform function G(f): Square Wave
G_fun = lambda f: 1 if abs(f) <= 0.25 else 0
f = np.arange(-0.5,0.5,0.001)
G_cont = np.array(list(map(G_fun,f))) 
#Approximated Fourier transform function G(f) using discrete FT, partial sum
def Gp_ft(m,f):
    t = np.arange(-m,m+1)       # fixed time interval
    dt = 1                      # fixed sampling freq
    g_t = np.array(list(map(g_fun,t)))    
    return dt*sum(g_t*np.exp( -2j*pi*f*t*dt))
#Approximated Fourier transform function G(f) using Fejer Kernel FT, partial sum
def Gp_ft_fejer(m,f):
    t = np.arange(-m,m+1)       # fixed time interval
    dt = 1                      # fixed sampling freq
    g_t = np.array(list(map(g_fun,t)))    
    w_Fejer= 1-abs(t)/m
    return dt*sum(w_Fejer* g_t *np.exp( -2j*pi*f*t*dt))
m = 64 # 4, 16, 64
G_ft = np.array(list(map(Gp_ft,repeat(m),f)))
G_ft_fejer = np.array(list(map(Gp_ft_fejer,repeat(m),f)))

# plot
fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Signal g(t)')
axs.plot(t,g_t,linestyle ='-', color='b' )
axs.scatter(t,g_t, color='r' )

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Square wave G(f)')
axs.plot(f,G_cont,linestyle ='-', color='b' )

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Partial sum FT Approximation of G(f)')
axs.plot(f,np.real(G_ft ),linestyle =':')
axs.plot(f,G_cont,linestyle ='--', color='r' )

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Partial sum Fejer FT Approximation of G(f)')
axs.plot(f,np.real(G_ft_fejer ),linestyle =':')
axs.plot(f,G_cont,linestyle ='--', color='r' )

#%%
# relationship between Gp (discrete fourier transform periodic ) and G (continuous transform)


#%%


def convex_eliminator(x,tol=5.0,level=-120,last = False):
    N = len(x); r = np.arange(1,N-1)
    idx = [False] + list( (x[r-1]>x[r]) & (x[r+1]>x[r]) &(x[r-1]>(x[r]+tol)) )+[False]
    x[idx] = level
    if (last): x[N-1]=level
    #return x

# Slepian Sequence (DPSS, discrete prolate spheroidal sequence)  III-106
N = 64; W = 8/N ;  Nth_eigenVector = 0 #N = 32; W = 1/8 
A_matrix = np.diag( 2*W*np.ones(N) )
for i in np.arange(0,N-1):        # Loop in the index for the upper triangle
    for j in np.arange(i+1,N):
        A_matrix[i,j] = sin(2*pi*W*(j-i)) / (pi*(j-i))
        A_matrix[j,i] = A_matrix[i,j]  
from numpy.linalg import eig
e,Q = eig( A_matrix )  #eigenValue, matrix of eigen vectors as col
e =np.real(e); Q = np.real(Q)
# sort
idx = np.argsort(e)[::-1]
e = e[idx]; Q = Q[:,idx]
# |G( )|^2 of the Nth eigenVector
G_square = np.abs(fft(Q[:,Nth_eigenVector])[:(N//2)] )**2  # [:(N//2)] select only the G() associated with positive frequency
G_square_DB = DB(G_square)
convex_eliminator(G_square_DB,0.0025) # concave G_square_DB
f = fftfreq(N,1)[:(N//2)] # [:(N//2)] select only the positive frequency


# Plot for the Nth eigenVector
x = np.arange(0,N)
fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Slepian Squence v_{}({},{})'.format(Nth_eigenVector,N,W))
axs.scatter(x,Q[:,Nth_eigenVector])
# Plot for |G( )|^2 of the Nth eigenVector

fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('|G( )|^2 of the v_{}({},{})'.format(Nth_eigenVector,N,W))
axs.plot(f,G_square_DB)
axs.axvline(W,c='r',linestyle='--',label = 'W ={:.2f}'.format(W))
axs.legend()

# Plot for eigenValue
fig, axs = plt.subplots(1, 1, figsize=(20,12))
fig.suptitle('Eigen Value $\lambda({},{})$'.format(N,W))
axs.scatter(x,e, label = 'Eigenvalue')
axs.axvline(2*N*W,c='r',linestyle='--',label = '2WN')
axs.legend()








