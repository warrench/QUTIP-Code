# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:12:09 2017

@author: Chris
"""
import numpy as np
import matplotlib.pyplot as plt


#wL = 6 GHz, wR = 6.5 GHz
def parameterspace(EjL,EJ,EjR,CL,Cj,CR):
    return (EjL/EJ + 1)*(EjR/EJ + 1) - (CL/Cj + 1)*(CR/Cj + 1)

def firstcondition(EjL,EJ,EjR,CL,Cj,CR):
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    return (EJ+EjL)*eps_L - (EJ+EjR)*eps_R
    

Ej = np.linspace(1,101,1000)
Cj = np.linspace(1e-15,101e-15,1000)

A = np.zeros((len(Ej),len(Cj)))
B = np.zeros((len(Ej),len(Cj)))

for i,E in enumerate(Ej):
        A[i,:] = parameterspace(17.0,E,16,65e-15,Cj,48.8e-15)
        B[i,:] = firstcondition(17.0,E,16,65e-15,Cj,48.8e-15)
        

        
plt.pcolor(Cj*1e15,Ej,A,cmap='RdBu',vmin=-1, vmax=1)
plt.xlabel(r'Capacitance, $C_J$ [fF]')
plt.ylabel(r'Josephson Energy $E_j$ [GHz]')
plt.colorbar()
plt.grid()
plt.show()

plt.pcolor(Cj*1e15,Ej,B,cmap='RdBu',vmin=-0.5,vmax=0.5)
plt.xlabel(r'Capacitance, $C_J$ [fF]')
plt.ylabel(r'Josephson Energy $E_j$ [GHz]')
plt.colorbar()
plt.grid()
plt.show()
        
