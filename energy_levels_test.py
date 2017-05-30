# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:33:33 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


def TwoQubitNonLin_wInt(EjL,EjR,CL,CR,Ej,Cj,phi=0):
    n = 10                #number of oscillator states
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    
    a = qt.tensor(qt.destroy(n),qt.qeye(n))
    b = qt.tensor(qt.qeye(n),qt.destroy(n))
    
    phi_L = np.sqrt(eps_L/2.0)*(a+a.dag())
    phi_R = np.sqrt(eps_R/2.0)*(b+b.dag())
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0 = H_L + H_R
    
    H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    
    return H0, H_LR, H_int

#H0, H_LR, H_int = TwoQubitNonLin_wInt(18.0,16.0,74e-15,67e-15,30.0,30.0e-15,-0.28762)
H0, H_LR, H_int = TwoQubitNonLin_wInt(17.0,16.0,74e-15,67e-15,30.0,30.0e-15,-0.28762)
qt.plot_energy_levels([(H0+H_LR),H_int],N=3,figsize=(6,8),show_ylabels=True)