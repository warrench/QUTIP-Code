# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:08:23 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

def hamiltonian_cpb(Ec, Ej, N, ng):
    """
    Return the charge qubit hamiltonian as a Qobj instance.
    """
    m = np.diag(4 * Ec * (np.arange(-N,N+1)-ng)**2) + 0.5 * Ej * (np.diag(-np.ones(2*N), 1) + 
                                                               np.diag(-np.ones(2*N), -1))
    return qt.Qobj(m)

N = np.arange(2,50)

def hamiltonian_transmon(Ej,Ec,n):
    """
    Return the transmon hamiltonian approximation
    """
    
    a = qt.destroy(n)
    
    H = np.sqrt(8*Ej*Ec)*a.dag()*a - Ec/(12.0)*(a+a.dag())**4
    
    return H

def hamiltonian_trans_exp(Ej,Ec,n):
    a = qt.destroy(n)
    
    x = np.sqrt(1.0/2.0)*(a.dag()+a)
    p = 1j*np.sqrt(1.0/2.0)*(a-a.dag())
    
    H = 4*Ec*((Ej/(8*Ec))**(0.25))*(p-0.5)**2 - Ej*x.cosm()
    return H

CPB = []
Transmon = []

for n in N:
    Hcpb = hamiltonian_cpb(0.4,16,n,0.5)
    Htrans = hamiltonian_trans_exp(15,0.4,n)
    eigen1 = Hcpb.eigenenergies()
    eigen2 = Htrans.eigenenergies()
    w01_cpb = eigen1[1]-eigen1[0]
    w01_transmon = eigen2[1]-eigen1[0]
    
    CPB.append(w01_cpb)
    Transmon.append(w01_transmon)
    
plt.plot(N,CPB,N,Transmon)