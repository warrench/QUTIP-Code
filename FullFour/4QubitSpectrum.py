# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:39:41 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

def QubitHam4_MatterSitesQubits(wq1,wq2,lambda1,lambda2,EjL,EjR,CL,CR,Ej,Cj,phi=0):
    n = 10                #number of oscillator states
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    
    sm1 = qt.tensor([qt.destroy(2),qt.qeye(n),qt.qeye(n),qt.qeye(2)])
    sm2 = qt.tensor([qt.qeye(2),qt.qeye(n),qt.qeye(n),qt.destroy(2)])
    a = qt.tensor([qt.qeye(2),qt.destroy(n),qt.qeye(n),qt.qeye(2)])
    b = qt.tensor([qt.qeye(2),qt.qeye(n),qt.destroy(n),qt.qeye(2)])
    
    Hqb = (2*np.pi*wq1)*sm1.dag()*sm1 + (2*np.pi*wq2)*sm2.dag()*sm2
    H_lambda = (2*np.pi*lambda1)*(sm1.dag()*a + sm1*a.dag()) + (2*np.pi*lambda2)*(sm2.dag()*b + sm2*b.dag())
    
    phi_L = np.sqrt(eps_L/2.0)*(a+a.dag())
    phi_R = np.sqrt(eps_R/2.0)*(b+b.dag())
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0 = H_L + H_R + H_LR + Hqb
    
    H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    H = H0 + H_int + H_lambda
    
    return H

phi = np.linspace(-0.5,0.5,1000)
Eval_mat1 = np.zeros((len(phi),2*2*10*10))

for i,Phi in enumerate(phi):
    if (i %(len(phi)/10))==0:
        print('%f Percent Completed' %(i/len(phi)*100))
    H = QubitHam4_MatterSitesQubits(5.0,5.5,0.1,0.1,17.0,16.0,74e-15,67e-15,30.0,30.0e-15,Phi)
    evals = H.eigenenergies()
    
    Eval_mat1[i,:] = evals
np.savetxt('4QubitTransitionSpectrum.txt',Eval_mat1,fmt='%f',delimiter=',')
for i in range(10):
    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.ylabel(r'Frequency [GHz]')
plt.title(r'Transition Energies of 4 Qubit Hamiltonian')
plt.show()