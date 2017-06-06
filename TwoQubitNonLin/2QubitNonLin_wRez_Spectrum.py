# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:56:49 2017

@author: Chris
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

n=5
def TwoQubitNonLin_wRez(EjL,EjR,CL,CR,Ej,Cj,wr,phi=0):
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    
    a = qt.tensor(qt.destroy(n),qt.qeye(n),qt.qeye(n))
    b = qt.tensor(qt.qeye(n),qt.destroy(n),qt.qeye(n))
    c = qt.tensor(qt.qeye(n),qt.qeye(n),qt.destroy(n))
    
    #phi_L = np.sqrt(eps_L/2.0)*(a+a.dag())
    #phi_R = np.sqrt(eps_R/2.0)*(b+b.dag())
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0_init = H_L + H_R + H_LR + (2*np.pi * wr)*c.dag()*c
    
    H_int1 = -(2*np.pi)*(np.exp(-eps_L/4)*EjL*eps_L**2/16.0)*((a.dag()**2)*(a**2))
    H_int2 = -(2*np.pi)*(np.exp(-eps_R/4)*EjR*eps_R**2/16.0)*((b.dag()**2)*(b**2))
    H_int3 = -(2*np.pi)*(np.exp(-np.sqrt(eps_L*eps_R)/4)/16.0)*((eps_L**2)*(a.dag()**2)*(a**2) + (eps_R**2)*(b.dag()**2)*(b**2) + 4*eps_L*eps_R*(a.dag()*a)*(b.dag()*b) )
    H0_int = H_int1 + H_int2 + H_int3
    
    #H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    H0 = H0_init + H0_int
    
    H_int = (2*np.pi * 0.1)*(a.dag()*c + a*c.dag()) + (2*np.pi * 0.15)*(b.dag()*c + b*c.dag())
    
    H = H0 + H_int
    return H,H0,H_int

phi = np.linspace(-0.5,0.5,501)

Eval_mat = np.zeros((len(phi),n*n*n))

coupling1 = np.zeros(len(phi))
coupling2 = np.zeros(len(phi))
coupling3 = np.zeros(len(phi))
coupling4 = np.zeros(len(phi))
coupling5 = np.zeros(len(phi))
coupling6 = np.zeros(len(phi))

for i, Phi in enumerate(phi):
    if (i %((len(phi)-1)/10))==0:
        print('%f Percent Completed' %(i/(len(phi)-1)*100))
    H,H0,Hint = TwoQubitNonLin_wRez(17.0,16.0,65e-15,48.8e-15,20.0,40.0e-15,10.0,Phi)
    evals = H.eigenenergies()
    eval2,evecs = H0.eigenstates()
    Eval_mat[i,:] = evals
    
    gnd = evecs[0]
    first = evecs[1]
    second = evecs[2]
    third = evecs[3]
    
    gnd_to_first = first*gnd.dag()
    gnd_to_second = second*gnd.dag()
    gnd_to_third = third*gnd.dag()
    first_to_second = second*first.dag()
    first_to_third = third*first.dag()
    second_to_third = third*second.dag()
    
    coupling1[i] = np.abs(qt.expect(Hint,gnd_to_first))
    coupling2[i] = np.abs(qt.expect(Hint,gnd_to_second))
    coupling3[i] = np.abs(qt.expect(Hint,gnd_to_third))
    coupling4[i] = np.abs(qt.expect(Hint,first_to_second))
    coupling5[i] = np.abs(qt.expect(Hint,first_to_third))
    coupling6[i] = np.abs(qt.expect(Hint,second_to_third))
    
    
    
for i in range(4):
    plt.plot(phi,(Eval_mat[:,i]-Eval_mat[:,0])/(2*np.pi))
plt.ylabel(r'Freqnecy [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'Transition Energies of Nonlinear Coupling, $H_{0} + H_{int}$')
plt.grid()
plt.show()

plt.plot(phi,coupling1/(np.pi*2))
plt.plot(phi,coupling2/(np.pi*2))
plt.plot(phi,coupling3/(np.pi*2))
plt.plot(phi,coupling4/(np.pi*2))
plt.plot(phi,coupling5/(np.pi*2))
plt.plot(phi,coupling6/(np.pi*2))
plt.legend([r'$|\psi_{g}>\rightarrow |\psi_{1}>$',
            r'$|\psi_{g}>\rightarrow |\psi_{2}>$',
            r'$|\psi_{g}>\rightarrow |\psi_{3}>$',
            r'$|\psi_{1}>\rightarrow |\psi_{2}>$',
            r'$|\psi_{1}>\rightarrow |\psi_{3}>$',
            r'$|\psi_{2}>\rightarrow |\psi_{3}>$'], loc='upper right')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.ylabel(r'Frequency [GHz]')
plt.title(r'Transition Matrix Elements w/ Resonator')
plt.show()