# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:51:24 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


def TwoQubitNonLin_NoInt(EjL,EjR,CL,CR,Ej,Cj,phi=0):
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
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H = H_L + H_R + H_LR
    
    return H, H_LR

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
    
    H0 = H_L + H_R + H_LR
    
    H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    H = H0 + H_int
    
    return H
    
    
    

phi = np.linspace(-0.5,0.5,1000)

Eval_mat1 = np.zeros((len(phi),10*10))
#Eval_mat2 = np.zeros((len(phi),10*10))
coupling = np.zeros(len(phi))
coupling2 = np.zeros(len(phi))
coupling3 = np.zeros(len(phi))

state_eg = qt.tensor(qt.basis(10,1),qt.basis(10,0))
state_ge = qt.tensor(qt.basis(10,0),qt.basis(10,1))

state_ef = qt.tensor(qt.basis(10,1),qt.basis(10,2))
state_fe = qt.tensor(qt.basis(10,2),qt.basis(10,1))

state_gf = qt.tensor(qt.basis(10,0),qt.basis(10,2))
state_fg = qt.tensor(qt.basis(10,2),qt.basis(10,0))


rho = state_ge*state_eg.dag()
rho2 = state_fe*state_ef.dag()
rho3 = state_fg*state_gf.dag()


for i,Phi in enumerate(phi):
    if (i %(len(phi)/10))==0:
        print('%f Percent Completed' %(i/len(phi)*100))
    #H, H_LR = TwoQubitNonLin_NoInt(17.0,16.0,74e-15,67e-15,30.0,30.0e-15,Phi)
    H = TwoQubitNonLin_wInt(17.0,16.0,74e-15,67e-15,30.0,30.0e-15,Phi)
    evals1 = H.eigenenergies()
    #evals2 = H_LR.eigenenergies()
    
    coupling[i] = qt.expect(H,rho+rho.dag())/2
    coupling2[i] = qt.expect(H,rho2+rho2.dag())/2
    coupling3[i] = qt.expect(H,rho3+rho3.dag())/2
    
    Eval_mat1[i,:] = evals1
    #Eval_mat2[i,:] = evals2
    
for i in range(3):
    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2 \pi}$')
plt.title(r'Transition Energies of Nonlinear Coupling')
plt.show()
for i in range(6):
    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))

plt.ylabel(r'Freqnecy [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'Transition Energies of Nonlinear Coupling')
plt.show()
#for i in range(6):
#    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
#plt.show()

#for i,coup in enumerate(coupling):
#    print(i,coup,phi[i])

plt.plot(phi,coupling/(2*np.pi))
plt.plot(phi,coupling2/(2*np.pi))
plt.plot(phi,coupling3/(2*np.pi))
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.ylabel(r'Frequency [GHz]')
plt.title(r'Mode Mixing Interaction')
plt.legend([r'ge $\rightarrow$ eg',r'ef $\rightarrow$ fe',r'gf $\rightarrow$ fg' ])
plt.grid()
plt.show()