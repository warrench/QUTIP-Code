# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:05:14 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

n=5

def Hamiltonian1(EjL,Ej,EjR,CL,Cj,CR,phi):
    EJL = np.abs(np.cos(phi*np.pi*2))*EjL
    EJR = np.abs(np.cos(phi*np.pi*2/0.33))*EjR
    
    
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(Ej+EJL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(Ej+EJR))
    
    a = qt.tensor(qt.destroy(n),qt.qeye(n))
    b = qt.tensor(qt.qeye(n),qt.destroy(n))
    
    H_L = ((Ej+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((Ej+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (Ej*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H = H_L + H_R + H_LR
    
    return H, H_LR

phi = np.linspace(-0.5,0.5,1001)

gg = qt.tensor(qt.basis(n,0),qt.basis(n,0))
eg = qt.tensor(qt.basis(n,1),qt.basis(n,0))
ge = qt.tensor(qt.basis(n,0),qt.basis(n,1))
ee = qt.tensor(qt.basis(n,1),qt.basis(n,1))

basislist = [gg,eg,ge,ee]

Eval_mat1 = np.zeros((len(phi),n*n))

EvecMat_gnd = np.zeros((len(phi),4))
EvecMat_first = np.zeros((len(phi),4))
EvecMat_second = np.zeros((len(phi),4))

for i,Phi in enumerate(phi):
    if (i %((len(phi)-1)/10))==0:
        print('%f Percent Completed' %(i/(len(phi)-1)*100))
    H,H_LR = Hamiltonian1(14,13.28,8.6,120e-15,70e-15,150e-15,Phi)
    H.tidyup(atol=1e-4)
    evals,evecs = H.eigenstates()
    
    Eval_mat1[i,:] = evals
    
    gnd = evecs[0]
    gnd.tidyup(atol=1e-2)
    gnd.unit()
    first = evecs[1]
    first.tidyup(atol=1e-2)
    first.unit()
    second = evecs[2]
    second.tidyup(atol=1e-2)
    second.unit()
    
    for j in range(4):
        Number_gnd = basislist[j].dag()*gnd
        Number_first = basislist[j].dag()*first
        Number_second = basislist[j].dag()*second
        EvecMat_gnd[i,j] = np.abs(Number_gnd[0,0])**2
        EvecMat_first[i,j] = np.abs(Number_first[0,0])**2
        EvecMat_second[i,j] = np.abs(Number_second[0,0])**2
        
for i in range(6):
    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
plt.ylabel(r'Freqnecy [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'Transition Energies of Nonlinear Coupling, $H_{0} + H_{int}$')
plt.grid()
plt.show()

for i in range(4):
    plt.plot(phi,EvecMat_gnd[:,i])
plt.grid()
plt.title(r'Composition of Ground State')
plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
plt.ylabel(r'Occupation Probability')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.show()

for i in range(4):
    plt.plot(phi,EvecMat_first[:,i])
plt.title(r'Composition of First Excited State')
plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.ylabel(r'Occupation Probability')
plt.grid()
plt.show()

for i in range(4):
    plt.plot(phi,EvecMat_second[:,i])
plt.title(r'Composition of Second Excited State')
plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.ylabel(r'Occupation Probability')
plt.grid()
plt.show()
    
    
    
    