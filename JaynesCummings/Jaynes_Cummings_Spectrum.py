# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:20:49 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

def JC(Ej,Ec,wr,phi,n=2):
    N = 10
    a = qt.tensor(qt.destroy(N),qt.qeye(n))
    sm = qt.tensor(qt.qeye(N),qt.destroy(n))
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))*np.sqrt(1.0+(0.025*(np.tan(phi*2*np.pi)))**2.0)
    w0 = np.sqrt(8*EJ*Ec)
    wr = wr * (2*np.pi)
    
    g = 0.1 * (2*np.pi)
    if n==2:
        w01 = (w0-Ec)*(2*np.pi)
        H0 = w01*sm.dag()*sm + wr*a.dag()*a
        Hint = g*(a.dag()*sm + a*sm.dag())
        H = H0 + Hint
    elif n>2:
       H0 = (w0*(2*np.pi))*sm.dag()*sm - (Ec/12*(2*np.pi))*(sm+sm.dag())**4 + wr*a.dag()*a
       Hint = g*(a.dag()*sm + a*sm.dag())
       H = H0 + Hint
    else:
        H = np.nan
       
    return H

phi = np.linspace(-0.5,0.5,1000)

Eval_mat1 = np.zeros((len(phi),2*10))
Eval_mat2 = np.zeros((len(phi),4*10))
   
for i,Phi in enumerate(phi):
    H1 = JC(17,0.4,6,Phi)
    H2 = JC(17,0.4,6,Phi,4)
    
    evals1 = H1.eigenenergies()
    evals2 = H2.eigenenergies()
    
    Eval_mat1[i,:] = np.real(evals1)
    Eval_mat2[i,:] = np.real(evals2)

for i in range(3):
    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
plt.show()
for i in range(3):
    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
plt.show()