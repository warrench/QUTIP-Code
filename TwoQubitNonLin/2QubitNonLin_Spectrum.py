# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:51:24 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
#                         Define Two Qubit Hamiltonian
#==============================================================================

n = 5 #number of oscillator states
def TwoQubitNonLin_NoInt(EjL,EjR,CL,CR,Ej,Cj,phi=0):
               
    
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
    
    H0 = H_L + H_R
    H = H_L + H_R + H_LR
    
    return H, H0, H_LR

def TwoQubitNonLin_wInt(EjL,EjR,CL,CR,Ej,Cj,phi=0):
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    
    a = qt.tensor(qt.destroy(n),qt.qeye(n))
    b = qt.tensor(qt.qeye(n),qt.destroy(n))
    
    #phi_L = np.sqrt(eps_L/2.0)*(a+a.dag())
    #phi_R = np.sqrt(eps_R/2.0)*(b+b.dag())
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0 = H_L + H_R + H_LR
    
    H_int1 = -(2*np.pi)*(np.exp(-eps_L/4)*EjL*eps_L**2/16.0)*((a.dag()**2)*(a**2))
    H_int2 = -(2*np.pi)*(np.exp(-eps_R/4)*EjR*eps_R**2/16.0)*((b.dag()**2)*(b**2))
    H_int3 = -(2*np.pi)*(np.exp(-np.sqrt(eps_L*eps_R)/4)/16.0)*((eps_L**2)*(a.dag()**2)*(a**2) + (eps_R**2)*(b.dag()**2)*(b**2) + 4*eps_L*eps_R*(a.dag()*a)*(b.dag()*b) )
    H_int = H_int1 + H_int2 + H_int3
    
    #H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    H = H0 + H_int
    
    return H,H0,H_int
#==============================================================================
#------------------------------------------------------------------------------    
#==============================================================================
#                         Creating Basic Operators
#==============================================================================    
    
a = qt.tensor(qt.destroy(n),qt.qeye(n))
b = qt.tensor(qt.qeye(n),qt.destroy(n))
Hopping_a = a+a.dag()               #   (a + adag) X I
Hopping_b = b+b.dag()               #   I X (b + bdag)
Hopping = Hopping_a + Hopping_b

    
phi = np.linspace(-0.5,0.5,501)


gg = qt.tensor(qt.basis(n,0),qt.basis(n,0))
eg = qt.tensor(qt.basis(n,1),qt.basis(n,0))
ge = qt.tensor(qt.basis(n,0),qt.basis(n,1))
ee = qt.tensor(qt.basis(n,1),qt.basis(n,1))

basislist = [gg,eg,ge,ee]

Eval_mat1 = np.zeros((len(phi),n*n))
Eval_mat2 = np.zeros((len(phi),n*n))
coupling = np.zeros(len(phi))
coupling2 = np.zeros(len(phi))
coupling3 = np.zeros(len(phi))
coupling4 = np.zeros(len(phi))

EvecMat_gnd = np.zeros((len(phi),4))
EvecMat_first = np.zeros((len(phi),4))
EvecMat_second = np.zeros((len(phi),4))

#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================
#                         Static Hamiltonian Spectrum
#==============================================================================
for i,Phi in enumerate(phi):
    if (i %((len(phi)-1)/10))==0:
        print('%f Percent Completed' %(i/(len(phi)-1)*100))
    #H, H0, Hint = TwoQubitNonLin_NoInt(17.0,16.0,65e-15,48.8e-15,20.0,40.0e-15,Phi)
    
    H,H0,Hint = TwoQubitNonLin_wInt(17.0,16.0,65e-15,48.8e-15,20.0,40.0e-15,Phi)
#    H.tidyup(atol=1e-4)
#    Hint.tidyup(atol=1e-4)
    evals1 = H.eigenenergies()
    evals2,evecs = H0.eigenstates()    
        
    gnd = evecs[0]
#    gnd.tidyup(atol=1e-3)
#    gnd.unit()
    first = evecs[1]
#    first.tidyup(atol=1e-3)
#    first.unit()
    second = evecs[2]
#    second.tidyup(atol=1e-3)
#    second.unit()
    third = evecs[3]
#    third.tidyup(atol=1e-3)
#    third.unit()
    fourth = evecs[4]
#    fourth.tidyup(atol=1e-3)
#    fourth.unit()
    
    gnd_to_first = first*gnd.dag()
    gnd_to_second = second*gnd.dag()
    first_to_second = second*first.dag()    
    gnd_to_third = third*gnd.dag()
    first_to_third = third*first.dag()
    second_to_third = third*second.dag()
    gnd_to_fourth = fourth*gnd.dag()
    first_to_fourth = fourth*first.dag()
    second_to_fourth = fourth*second.dag()
    third_to_fourth = fourth*third.dag()
    
    for j in range(4):
        Number_gnd = basislist[j].dag()*gnd
        Number_first = basislist[j].dag()*first
        Number_second = basislist[j].dag()*second
        EvecMat_gnd[i,j] = np.abs(Number_gnd[0,0])**2
        EvecMat_first[i,j] = np.abs(Number_first[0,0])**2
        EvecMat_second[i,j] = np.abs(Number_second[0,0])**2    
    
#    coupling[i] = np.abs(qt.expect(Hint,gnd_to_fourth))
#    coupling2[i] = np.abs(qt.expect(Hint,first_to_fourth))
#    coupling3[i] = np.abs(qt.expect(Hint,second_to_fourth))
#    coupling4[i] = np.abs(qt.expect(Hint,third_to_fourth))
#    coupling[i] = np.abs(qt.expect(Hint,gnd_to_third))
#    coupling2[i] = np.abs(qt.expect(Hint,first_to_third))
#    coupling3[i] = np.abs(qt.expect(Hint,second_to_third))
    coupling[i] = np.abs(qt.expect(Hopping_b,gnd_to_first))
    coupling2[i] = np.abs(qt.expect(Hopping_b,first_to_second))
    coupling3[i] = np.abs(qt.expect(Hopping_b,gnd_to_second))
    
    Eval_mat1[i,:] = evals1
    #Eval_mat2[i,:] = evals2

#         ===========================================================
#                            Plotting Energy Levels
#         ===========================================================

#for i in range(10):
#    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
#plt.ylabel(r'Frequency [GHz]')
#plt.xlabel(r'$\frac{\Phi}{2 \pi}$')
#plt.title(r'Transition Energies of Nonlinear Coupling, $H_{0}$')
#plt.show()

#for i in range(6):
#    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
#plt.ylabel(r'Freqnecy [GHz]')
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.title(r'Transition Energies of Nonlinear Coupling, $H_{0} + H_{int}$')
#plt.grid()
#plt.show()
#for i in range(6):
#    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
#plt.show()

#         ===========================================================
#                               Plotting Coupling
#         ===========================================================

#plt.plot(phi,coupling/(2*np.pi))
#plt.plot(phi,coupling3/(2*np.pi))
#plt.plot(phi,coupling2/(2*np.pi))
#plt.plot(phi,coupling4/(2*np.pi))
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.ylabel(r'Frequency [GHz]')
#plt.legend([r'$|\psi_{g}>\rightarrow |\psi_{1}>$',
#            r'$|\psi_{1}>\rightarrow|\psi_{2}>$',
#            r'$|\psi_{g}>\rightarrow|\psi_{2}>$'], loc='upper right')
#plt.title(r'$|<\psi_{i}|\mathbb{1}\otimes (b+b^{\dagger})|\psi_{j}>|$')
#plt.grid()
#plt.show()

#         ===========================================================
#                           Plotting Eigenstate Composition
#         ===========================================================

#for i in range(4):
#    plt.plot(phi,EvecMat_gnd[:,i])
#plt.grid()
#plt.title(r'Composition of Ground State')
#plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
#plt.ylabel(r'Occupation Probability')
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.show()
#
#for i in range(4):
#    plt.plot(phi,EvecMat_first[:,i])
#plt.title(r'Composition of First Excited State')
#plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.ylabel(r'Occupation Probability')
#plt.grid()
#plt.show()
#
#for i in range(4):
#    plt.plot(phi,EvecMat_second[:,i])
#plt.title(r'Composition of Second Excited State')
#plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.ylabel(r'Occupation Probability')
#plt.grid()
#plt.show()

#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================
#                        Time Domain Measurements
#==============================================================================