import numpy as np
import scipy.integrate as i
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from astropy import units as u
from astropy import constants as c
import scipy.optimize as o
import scipy.interpolate as i2
import math as math
import cmath as cm
from scipy.signal.windows import general_gaussian
import warnings
from typing import Callable, Iterable
from scipy.optimize import root_scalar
from scipy.interpolate import splrep, splev
import math as mt
import pandas as pd
f = 16
from numba import jit, njit, vectorize

def resonance_res_krcomp_v8(r2, w, m, elecd, Pm, k2r, k2i,kz=np.pi/4, Bz=12.967, B0 = 0, keplerian = True, q = 1/2):
    # w = np.real(w)
    # r = np.arange(1,r2 + 24/500,24/500)
    reso = 20000
    # r = np.linspace(1,r2*3, reso)
    r = np.linspace(1E-6,r2*2, reso)
#  r = np.linspace(1E-6,r2*3, reso)
    # print(r)
    
    # Define Normalizations
    r1 = 0.1

    ## Define OMEGA0:
    bb = 10000
    v0 = bb/np.sqrt(r1)
    Omega0 = v0/r1
    elecd = elecd / (r1**2*Omega0)
    nu = elecd * Pm

    ## SET CONSTANTS
    rho = (1.0*10**19) * (2.0*1.6725*10**(-27))
    mu_0 = 1.2566*10**(-6)
        
    vAp1 = B0/(r1*Omega0*np.sqrt(rho*mu_0)*10000)
    vAz = Bz / (r1*Omega0*np.sqrt(rho*mu_0)*10000)
    vAp = vAp1/r
    vp1 = 1
    wA = (m/r)*vAp + kz*vAz
    x = r**2
    if keplerian == True:
        x = r**2
        vp = vp1/(x**(1/4))
        W = vp/np.sqrt(x)
    elif keplerian == 'khalzov':
        W = vp1/(r**(3/2))
    elif keplerian == 'rigid':
        W = vp1
    elif keplerian == False:
        vp = vp1/(x**(q/2))
        W = vp/r
        Wp = (-1*(q+1)/2)*(W/x)
    elif keplerian == 'Sh2':
        W = vp1/r**2
        Wp = -2*vp1/r**3
        Wpp = 6*vp1/r**4
    elif keplerian == 'saturated':
        x = r**2
        A0 = 1.4923889E-02
        A1 = -1.0114824E-01
        A2 = 8.5753488E-02
        A3 = -2.8021088E-02
        A4 = -4.7882189E-01
        A5 = 1.0877462E-01
        A6 = 6.7857446E+00
        A7 = -5.0846913E+00
        A8 = -4.3114342E+01
        A9 = 5.2943846E+01
        A10 = 1.2219068E+02
        A11 = -1.9185099E+02
        A12 = -1.7535573E+02
        A13 = 3.4808679E+02
        A14 = 1.1975807E+02
        A15 = -3.4500839E+02
        A16 =  -1.4844420E+01
        A17 = 1.7930272E+02
        A18 = -2.5033736E+01
        W = 0.08875 + 0.91125/np.sqrt(x)**2 + A0 + A1*(np.sqrt(x)-2) + A2*(np.sqrt(x)-2)**2+A3*(np.sqrt(x)-2)**3 + A4*(np.sqrt(x)-2)**4 + A5*(np.sqrt(x)-2)**5 + A6*(np.sqrt(x)-2)**6 + A7*(np.sqrt(x)-2)**7 + A8*(np.sqrt(x)-2)**8 + A9*(np.sqrt(x)-2)**9 + A10*(np.sqrt(x)-2)**10 + A11*(np.sqrt(x)-2)**11 + A12*(np.sqrt(x)-2)**12 + A13*(np.sqrt(x)-2)**13 + A14*(np.sqrt(x)-2)**14 + A15*(np.sqrt(x)-2)**15 + A16*(np.sqrt(x)-2)**16 + A17*(np.sqrt(x)-2)**17 + A18*(np.sqrt(x)-2)**18 - ( A1 + A3 + A5 + A7 + A9 + A11 + A13 + A15 + A17 )*(np.sqrt(x)-2)**19 - ( A0 + A2 + A4 + A6 + A8 + A10 + A12 + A14 + A16 + A18 )*(np.sqrt(x)-2)**20
        
    elif keplerian == 'saturated2':
        data = pd.read_csv('Re750_4_0.1_0_to_1.txt', header = None, delim_whitespace=True)
        r_f = data[0]
        W = data[1]
        Wr = W
        W_f = i2.CubicSpline(r_f, W)
        Wp_f = W_f.derivative()
        W = W_f(r)
    elif keplerian == 'tanh':
        a = c = b = d = 1
        d = 0.85
        W = d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv3':
        a = c = b = d = 1
        d = 0.84
        W = d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv4':
        a = c = b = d = 1
        a = 0.6
        c = a
        d = 0.9
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'mKep':
        r10 = 1.5
        R0 = 1
        W = np.zeros(len(r))
        Wp = np.zeros(len(r))
        Wpp = np.zeros(len(r))
        for j in range(0,len(r)):
            if(r[j] < r10):
                W[j] = 1
                Wp[j] = 0
                Wpp[j]= 0
            else:
                W[j] = 1/(1+((r[j]-r10)/R0)**(3/2))
                Wp[j] = -3/(2*R0)*((r[j]-r10)/R0)**(1/2)/(1+((r[j]-r10)/R0)**(3/2))**2
                u = 1+((r[j]-r10)/R0)**(3/2)
                up = 3/(2*R0)*((r[j]-r10)/R0)**(1/2)
                Wpp[j] = -( (3/(4*R0**2)*(1/R0*(r[j]-r10))**(-1/2))*u**2 - (3/(2*R0) * (1/R0*(r[j]-r10))**(1/2))*2*u*up)/(u**4)
    elif keplerian == 'mKep2':
        r10 = 1.5
        R0 = 1.5
        W = np.zeros(len(r))
        Wp = np.zeros(len(r))
        Wpp = np.zeros(len(r))
        for j in range(0,len(r)):
            if(r[j] < r10):
                W[j] = 1
                Wp[j] = 0
                Wpp[j]= 0
            else:
                W[j] = 1/(1+((r[j]-r10)/R0)**(3/2))
                Wp[j] = -3/(2*R0)*((r[j]-r10)/R0)**(1/2)/(1+((r[j]-r10)/R0)**(3/2))**2
                u = 1+((r[j]-r10)/R0)**(3/2)
                up = 3/(2*R0)*((r[j]-r10)/R0)**(1/2)
                Wpp[j] = -( (3/(4*R0**2)*(1/R0*(r[j]-r10))**(-1/2))*u**2 - (3/(2*R0) * (1/R0*(r[j]-r10))**(1/2))*2*u*up)/(u**4)
    
    elif keplerian == 'exp1':
        a = 1
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp2':
        a = 1
        W = a*r*np.exp(1-r) + (1-a)
        Wp = a*np.exp(1-r)*(1-r)
        Wpp = a*np.exp(1-r)*(r-2)
    elif keplerian == 'exp3':
        a = .75
        W = a*r*np.exp(1-r) + (1-a)
        Wp = a*np.exp(1-r)*(1-r)
        Wpp = a*np.exp(1-r)*(r-2)
    elif keplerian == 'exp4':
        a = 0.73123123
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp5':
        a = 0.8689352412721272
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp6':
        a = 0.9366210025563255
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp7':
        a = 0.9448239074457745
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp8':
        a = 0.9409132827360274
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp9':
        a = 0.9375052606522065
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'tanhv5':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.77
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv6':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.5940
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv7':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.7488
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv8':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.8463
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv9':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.8158134053621449
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv10':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.8303
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    ### Set freqs
    wb = w-m*W
    Q = k2r(r)

    # res_cond_p = wb.real - 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real
    # res_cond_m = wb.real + 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real

    res_cond_p = wb.real - 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real
    res_cond_m = wb.real + 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real


    # print('sqrt cond')
    # display(1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real)
    
    res_cond_ip = wb.imag - (1j*(elecd + nu)/2*Q).imag + 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).imag
    res_cond_im = wb.imag - (1j*(elecd + nu)/2*Q).imag - 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).imag

    # plt.figure()
    # plt.plot(r, 4*wA**2 - elecd**2*Q**2)
    # plt.ylabel(r'$4w_A^2 - \eta^2Q(r)^2$')
    # plt.xlabel('r')
    # plt.show
    ### ideal
    # plt.figure()
    # plt.plot(r, r**3*w.real-vAp1*r-1)
    # plt.xlim(0,10)
    # plt.ylim(-1,1)
    # plt.ylabel(r'res-ideal')
    # plt.xlabel('r')
    # plt.show
    # plt.figure()
    # plt.plot(r, 4*wA**2 - elecd**2*Q**2)
    # plt.ylabel(r'$4w_A^2 - \eta^2Q(r)^2$')
    # plt.xlabel('r')
    # plt.ylim(-0.1,0.1)
    # plt.xlim(0.9,r2 + 0.1)
    # plt.axhline(y = 0, linestyle = 'dashed', color = 'k')
    # plt.show
    
    # plt.figure()
    # plt.plot(r, res_cond_p, label = 'p', color = 'g')
    # plt.plot(r, res_cond_m, label = 'm', color = 'b')
    # plt.plot(r, res_cond_ip, label = 'ip', color = 'r')
    # plt.plot(r, res_cond_im, label = 'im', color = 'purple', linestyle = ':')
    # plt.axvline(x = 1, color = 'k')
    # plt.axvline(x = r2, color = 'k')
    # plt.ylim(-1,1)
    # plt.axhline(0, linestyle = 'dashed', color = 'k')
    # plt.ylabel('Resonance condition')
    # plt.legend()
    # plt.xlabel('r')
    # plt.show()

    tol = 1E-3
    ii_p = np.where(np.abs(res_cond_p) < tol)
    ii_m = np.where(np.abs(res_cond_m) < tol)
    ii_ip = np.where(np.abs(res_cond_ip) < tol)
    ii_im = np.where(np.abs(res_cond_im) < tol)

    # print(min(np.abs(res_cond_m)))
    # print(ii_m)
    # print(ii_p)
    # print(ii_m)
    # print(ii_i)

    # print(np.min(np.abs(res_cond_p)))
    # print(np.min(np.abs(res_cond_m)))

    # print(np.min(np.abs(res_cond_p)))
    if(len(ii_p[0]) != 0):
        ii_min_p = np.where(np.abs(res_cond_p) == np.min(np.abs(res_cond_p[ii_p[:]])))
        rp = r[ii_min_p[0][0]]
    else:
        rp = 0
    if(len(ii_m[0]) != 0):
        ii_min_m = np.where(np.abs(res_cond_m) == np.min(np.abs(res_cond_m[ii_m[:]])))
        rm = r[ii_min_m[0][0]]
    else:
        rm = 0
    if(len(ii_ip[0]) != 0):
        ii_min_ip = np.where(np.abs(res_cond_ip) == np.min(np.abs(res_cond_ip[ii_ip[:]])))
        rip = r[ii_min_ip[0][0]]
    else:
        rip = 0
    if(len(ii_im[0]) != 0):
        ii_min_im = np.where(np.abs(res_cond_im) == np.min(np.abs(res_cond_im[ii_im[:]])))
        rim = r[ii_min_im[0][0]]
    else:
        rim = 0
    tol = 1E-3
    try:
        rc = r[(np.where(np.abs(wb.real) < tol))[:]][0]
    except:
        rc = 0
        plt.figure()
        plt.plot(r,wb.real)
        
    # print(f'rp = {rp}')
    # print(f'rm = {rm}')
    # print(f'rip = {rip}')
    # print(f'rim = {rim}')

    # wA = kz*vAz
    # rp = ( m / (w- wA) )**(2/3)
    # rm = ( m / (w+ wA) )**(2/3)
    # r0 = ( m / w )**(2/3)

    # print(f'rp = {rp}')
    # print(f'rm = {rm}')
    
    return (rp, rm, rip, rim, rc)

def resonance_res_krcomp_v8_print(r2, w, m, elecd, Pm, k2r, k2i,kz=np.pi/4, Bz=12.967, B0 = 0, keplerian = True, q = 1/2):
    # w = np.real(w)
    # r = np.arange(1,r2 + 24/500,24/500)
    reso = 20000
    # r = np.linspace(1,r2*3, reso)
    r = np.linspace(1E-6,r2*2, reso)
#  r = np.linspace(1E-6,r2*3, reso)
    # print(r)
    
    # Define Normalizations
    r1 = 0.1

    ## Define OMEGA0:
    bb = 10000
    v0 = bb/np.sqrt(r1)
    Omega0 = v0/r1
    elecd = elecd / (r1**2*Omega0)
    nu = elecd * Pm

    ## SET CONSTANTS
    rho = (1.0*10**19) * (2.0*1.6725*10**(-27))
    mu_0 = 1.2566*10**(-6)
        
    vAp1 = B0/(r1*Omega0*np.sqrt(rho*mu_0)*10000)
    vAz = Bz / (r1*Omega0*np.sqrt(rho*mu_0)*10000)
    vAp = vAp1/r
    vp1 = 1
    wA = (m/r)*vAp + kz*vAz
    x = r**2
    if keplerian == True:
        x = r**2
        vp = vp1/(x**(1/4))
        W = vp/np.sqrt(x)
    elif keplerian == 'khalzov':
        W = vp1/(r**(3/2))
    elif keplerian == 'rigid':
        W = vp1
    elif keplerian == False:
        vp = vp1/(x**(q/2))
        W = vp/r
        Wp = (-1*(q+1)/2)*(W/x)
    elif keplerian == 'Sh2':
        W = vp1/r**2
        Wp = -2*vp1/r**3
        Wpp = 6*vp1/r**4
    elif keplerian == 'saturated':
        x = r**2
        A0 = 1.4923889E-02
        A1 = -1.0114824E-01
        A2 = 8.5753488E-02
        A3 = -2.8021088E-02
        A4 = -4.7882189E-01
        A5 = 1.0877462E-01
        A6 = 6.7857446E+00
        A7 = -5.0846913E+00
        A8 = -4.3114342E+01
        A9 = 5.2943846E+01
        A10 = 1.2219068E+02
        A11 = -1.9185099E+02
        A12 = -1.7535573E+02
        A13 = 3.4808679E+02
        A14 = 1.1975807E+02
        A15 = -3.4500839E+02
        A16 =  -1.4844420E+01
        A17 = 1.7930272E+02
        A18 = -2.5033736E+01
        W = 0.08875 + 0.91125/np.sqrt(x)**2 + A0 + A1*(np.sqrt(x)-2) + A2*(np.sqrt(x)-2)**2+A3*(np.sqrt(x)-2)**3 + A4*(np.sqrt(x)-2)**4 + A5*(np.sqrt(x)-2)**5 + A6*(np.sqrt(x)-2)**6 + A7*(np.sqrt(x)-2)**7 + A8*(np.sqrt(x)-2)**8 + A9*(np.sqrt(x)-2)**9 + A10*(np.sqrt(x)-2)**10 + A11*(np.sqrt(x)-2)**11 + A12*(np.sqrt(x)-2)**12 + A13*(np.sqrt(x)-2)**13 + A14*(np.sqrt(x)-2)**14 + A15*(np.sqrt(x)-2)**15 + A16*(np.sqrt(x)-2)**16 + A17*(np.sqrt(x)-2)**17 + A18*(np.sqrt(x)-2)**18 - ( A1 + A3 + A5 + A7 + A9 + A11 + A13 + A15 + A17 )*(np.sqrt(x)-2)**19 - ( A0 + A2 + A4 + A6 + A8 + A10 + A12 + A14 + A16 + A18 )*(np.sqrt(x)-2)**20
        
    elif keplerian == 'saturated2':
        data = pd.read_csv('Re750_4_0.1_0_to_1.txt', header = None, delim_whitespace=True)
        r_f = data[0]
        W = data[1]
        Wr = W
        W_f = i2.CubicSpline(r_f, W)
        Wp_f = W_f.derivative()
        W = W_f(r)
    elif keplerian == 'tanh':
        a = c = b = d = 1
        d = 0.85
        W = d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv3':
        a = c = b = d = 1
        d = 0.84
        W = d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv4':
        a = c = b = d = 1
        a = 0.6
        c = a
        d = 0.9
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'mKep':
        r10 = 1.5
        R0 = 1
        W = np.zeros(len(r))
        Wp = np.zeros(len(r))
        Wpp = np.zeros(len(r))
        for j in range(0,len(r)):
            if(r[j] < r10):
                W[j] = 1
                Wp[j] = 0
                Wpp[j]= 0
            else:
                W[j] = 1/(1+((r[j]-r10)/R0)**(3/2))
                Wp[j] = -3/(2*R0)*((r[j]-r10)/R0)**(1/2)/(1+((r[j]-r10)/R0)**(3/2))**2
                u = 1+((r[j]-r10)/R0)**(3/2)
                up = 3/(2*R0)*((r[j]-r10)/R0)**(1/2)
                Wpp[j] = -( (3/(4*R0**2)*(1/R0*(r[j]-r10))**(-1/2))*u**2 - (3/(2*R0) * (1/R0*(r[j]-r10))**(1/2))*2*u*up)/(u**4)
    elif keplerian == 'mKep2':
        r10 = 1.5
        R0 = 1.5
        W = np.zeros(len(r))
        Wp = np.zeros(len(r))
        Wpp = np.zeros(len(r))
        for j in range(0,len(r)):
            if(r[j] < r10):
                W[j] = 1
                Wp[j] = 0
                Wpp[j]= 0
            else:
                W[j] = 1/(1+((r[j]-r10)/R0)**(3/2))
                Wp[j] = -3/(2*R0)*((r[j]-r10)/R0)**(1/2)/(1+((r[j]-r10)/R0)**(3/2))**2
                u = 1+((r[j]-r10)/R0)**(3/2)
                up = 3/(2*R0)*((r[j]-r10)/R0)**(1/2)
                Wpp[j] = -( (3/(4*R0**2)*(1/R0*(r[j]-r10))**(-1/2))*u**2 - (3/(2*R0) * (1/R0*(r[j]-r10))**(1/2))*2*u*up)/(u**4)
    
    elif keplerian == 'exp1':
        a = 1
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp2':
        a = 1
        W = a*r*np.exp(1-r) + (1-a)
        Wp = a*np.exp(1-r)*(1-r)
        Wpp = a*np.exp(1-r)*(r-2)
    elif keplerian == 'exp3':
        a = .75
        W = a*r*np.exp(1-r) + (1-a)
        Wp = a*np.exp(1-r)*(1-r)
        Wpp = a*np.exp(1-r)*(r-2)
    elif keplerian == 'exp4':
        a = 0.73123123
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp5':
        a = 0.8689352412721272
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp6':
        a = 0.9366210025563255
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp7':
        a = 0.9448239074457745
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp8':
        a = 0.9409132827360274
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'exp9':
        a = 0.9375052606522065
        W = a*np.exp(1-r) + 1-a
        Wp = -a*np.exp(1-r)
        Wpp = a*np.exp(1-r)
    elif keplerian == 'tanhv5':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.77
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv6':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.5940
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv7':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.7488
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv8':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.8463
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv9':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.8158134053621449
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    elif keplerian == 'tanhv10':
        a = c = b = d = 1
        a = 1
        c = a
        d = 0.8303
        W= d*np.tanh(-c*r+a)+b
        Wp = -c * d * (1/np.cosh(-c * r + a))**2
        Wpp = -2 * c**2 * d * np.tanh(-c * r + a)*(1/np.cosh(-c * r + a))**2
    ### Set freqs
    wb = w-m*W
    Q = k2r(r)
    wbe = wb - 1j*elecd*Q
    wbn = wb - 1j*nu*Q
    # res_cond_p = wb.real - 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real
    # res_cond_m = wb.real + 1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real

    res_cond_p = wb.real - 1/2*np.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real
    res_cond_m = wb.real + 1/2*np.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real - (1j*(elecd + nu)/2*Q).real


    # print('sqrt cond')
    # display(1/2*np.emath.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).real)
    
    res_cond_ip = wb.imag - (1j*(elecd + nu)/2*Q).imag + 1/2*np.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).imag
    res_cond_im = wb.imag - (1j*(elecd + nu)/2*Q).imag - 1/2*np.sqrt(((4*wA**2 - (elecd-nu)**2*Q**2))).imag

    plt.figure()
    plt.title('Combined Root')
    plt.ylim(-0.1,0.1)
    plt.xlim(0.9, 5.1)
    plt.plot(r, wA**2-wbe*wbn)
    plt.axhline(0, linestyle = 'dashed', color = 'k')
    plt.show()
    
    # plt.ylabel(r'$4w_A^2 - \eta^2Q(r)^2$')
    # plt.axhline(y=0, color = 'k', linestyle = ':')
    # plt.ylim(-0.1,0.1)
    # plt.xlim(0.9, 5.1)
    # plt.axvline(x = 1, color = 'k', linestyle = ':')
    # plt.axvline(x = 5, color = 'k', linestyle = ':')
    # plt.xlabel('r')
    # plt.show
    
    plt.figure()
    plt.plot(r, 4*wA**2 - elecd**2*Q**2)
    plt.ylabel(r'$4w_A^2 - \eta^2Q(r)^2$')
    plt.axhline(y=0, color = 'k', linestyle = ':')
    plt.ylim(-0.1,0.1)
    plt.xlim(0.9, 5.1)
    plt.axvline(x = 1, color = 'k', linestyle = ':')
    plt.axvline(x = 5, color = 'k', linestyle = ':')
    plt.axhline(0, linestyle = 'dashed', color = 'k')
    plt.xlabel('r')
    plt.show
    ### ideal
    # plt.figure()
    # plt.plot(r, r**3*w.real-vAp1*r-1)
    # plt.xlim(0,10)
    # plt.ylim(-1,1)
    # plt.ylabel(r'res-ideal')
    # plt.xlabel('r')
    # plt.show
    # plt.figure()
    # plt.plot(r, 4*wA**2 - elecd**2*Q**2)
    # plt.ylabel(r'$4w_A^2 - \eta^2Q(r)^2$')
    # plt.xlabel('r')
    # plt.ylim(-0.1,0.1)
    # plt.xlim(0.9,r2 + 0.1)
    # plt.axhline(y = 0, linestyle = 'dashed', color = 'k')
    # plt.show
    
    plt.figure()
    plt.plot(r, res_cond_p, label = 'p', color = 'g')
    plt.plot(r, res_cond_m, label = 'm', color = 'b')
    plt.plot(r, res_cond_ip, label = 'ip', color = 'r')
    plt.plot(r, res_cond_im, label = 'im', color = 'purple', linestyle = ':')
    plt.axvline(x = 1, color = 'k')
    plt.axvline(x = r2, color = 'k')
    plt.ylim(-.1,.1)
    plt.xlim(0.9,5.1)
    plt.axhline(0, linestyle = 'dashed', color = 'k')
    plt.ylabel('Resonance condition')
    plt.legend()
    plt.xlabel('r')
    plt.show()

    tol = 1E-3
    ii_p = np.where(np.abs(res_cond_p) < tol)
    ii_m = np.where(np.abs(res_cond_m) < tol)
    ii_ip = np.where(np.abs(res_cond_ip) < tol)
    ii_im = np.where(np.abs(res_cond_im) < tol)

    # print(min(np.abs(res_cond_m)))
    # print(ii_m)
    # print(ii_p)
    # print(ii_m)
    # print(ii_i)

    # print(np.min(np.abs(res_cond_p)))
    # print(np.min(np.abs(res_cond_m)))

    # print(np.min(np.abs(res_cond_p)))
    if(len(ii_p[0]) != 0):
        ii_min_p = np.where(np.abs(res_cond_p) == np.min(np.abs(res_cond_p[ii_p[:]])))
        rp = r[ii_min_p[0][0]]
    else:
        rp = 0
    if(len(ii_m[0]) != 0):
        ii_min_m = np.where(np.abs(res_cond_m) == np.min(np.abs(res_cond_m[ii_m[:]])))
        rm = r[ii_min_m[0][0]]
    else:
        rm = 0
    if(len(ii_ip[0]) != 0):
        ii_min_ip = np.where(np.abs(res_cond_ip) == np.min(np.abs(res_cond_ip[ii_ip[:]])))
        rip = r[ii_min_ip[0][0]]
    else:
        rip = 0
    if(len(ii_im[0]) != 0):
        ii_min_im = np.where(np.abs(res_cond_im) == np.min(np.abs(res_cond_im[ii_im[:]])))
        rim = r[ii_min_im[0][0]]
    else:
        rim = 0
    tol = 1E-3
    try:
        rc = r[(np.where(np.abs(wb.real) < tol))[:]][0]
    except:
        rc = 0
        plt.figure()
        plt.plot(r,wb.real)
        
    # print(f'rp = {rp}')
    # print(f'rm = {rm}')
    # print(f'rip = {rip}')
    # print(f'rim = {rim}')

    # wA = kz*vAz
    # rp = ( m / (w- wA) )**(2/3)
    # rm = ( m / (w+ wA) )**(2/3)
    # r0 = ( m / w )**(2/3)

    # print(f'rp = {rp}')
    # print(f'rm = {rm}')
    
    return (rp, rm, rip, rim, rc)
