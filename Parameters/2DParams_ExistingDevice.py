# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:25:34 2017

@author: Chris
"""

import numpy as np
import matplotlib.pyplot as plt

def SecondCondition(EjL,EJ,EjR,CL,Cj,CR):
    return (EjL/EJ + 1)*(EjR/EJ + 1) - (CL/Cj + 1)*(CR/Cj + 1)

def firstcondition(EjL,EJ,EjR,CL,Cj,CR):
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    return (EJ+EjL)*eps_L - (EJ+EjR)*eps_R

EjL = np.linspace(0.1,20.1,501)
EjR = np.linspace(0.1,20.1,501)

A = np.zeros((len(EjL),len(EjR)))
B = np.zeros((len(EjL),len(EjR)))

for i,E in enumerate(EjL):
    A[i,:] = firstcondition(E,13.28,EjR,120e-15,70e-15,150e-15)
    B[i,:] = SecondCondition(E,13.28,EjR,78e-15,62.4e-15,70.2e-15)
    
plt.pcolor(EjL,EjR,A,cmap='RdBu',vmin=-1, vmax=1)
plt.xlabel(r'$E_{J_R}$ [GHz]')
plt.ylabel(r'$E_{J_L}$ [GHz]')
plt.colorbar()
plt.grid()
plt.show()

plt.pcolor(EjL,EjR,B,cmap='RdBu',vmin=-0.5,vmax=0.5)
plt.xlabel(r'$E_{J_R}$ [GHz]')
plt.ylabel(r'$E_{J_L}$ [GHz]')
plt.colorbar()
plt.grid()
plt.show()