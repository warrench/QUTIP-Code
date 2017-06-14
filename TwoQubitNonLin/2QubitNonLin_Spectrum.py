# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:51:24 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


#params = {'text.usetex': True,'text.latex.preamble': r'\usepackage{physics}'}
#plt.rcParams.update(params)

#==============================================================================
#                         Define Two Qubit Hamiltonian
#==============================================================================

n = 4 #number of oscillator states
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
#    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b + a*b.dag())
#            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a*b.dag()))
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


#gg = qt.tensor(qt.basis(n,0),qt.basis(n,0))
#eg = qt.tensor(qt.basis(n,1),qt.basis(n,0))
#ge = qt.tensor(qt.basis(n,0),qt.basis(n,1))
#ee = qt.tensor(qt.basis(n,1),qt.basis(n,1))

#basislist = [gg,eg,ge,ee]

Eval_mat1a = np.zeros((len(phi),n*n))
Eval_mat2a = np.zeros((len(phi),n*n))
Eval_mat3a = np.zeros((len(phi),n*n))

Eval_mat1b = np.zeros((len(phi),n*n))
Eval_mat2b = np.zeros((len(phi),n*n))
Eval_mat3b = np.zeros((len(phi),n*n))

coupling1 = np.zeros(len(phi))
coupling2 = np.zeros(len(phi))
coupling3 = np.zeros(len(phi))
coupling4 = np.zeros(len(phi))
coupling5 = np.zeros(len(phi))
coupling6 = np.zeros(len(phi))
coupling7 = np.zeros(len(phi))
coupling8 = np.zeros(len(phi))
coupling9 = np.zeros(len(phi))
coupling10 = np.zeros(len(phi))

Coupling_matA = np.zeros((len(phi),10))
Coupling_matB = np.zeros((len(phi),10))



#EvecMat_gnd = np.zeros((len(phi),4))
#EvecMat_first = np.zeros((len(phi),4))
#EvecMat_second = np.zeros((len(phi),4))

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
    H.tidyup(atol=1e-12)
#    Hint.tidyup(atol=1e-4)
    evals1 = H.eigenenergies()
    evals2,evecs = H0.eigenstates()    
        
    gnd = evecs[0]
    first = evecs[1]
    second = evecs[2]
    third = evecs[3]
    fourth = evecs[4]
    fifth = evecs[5]
    sixth = evecs[6]
    
    gnd_to_first = first*gnd.dag()
    gnd_to_second = second*gnd.dag()
    
    first_to_third = third*first.dag()
    first_to_fourth = fourth*first.dag()
    first_to_fifth = fifth*first.dag()
    first_to_sixth = sixth*first.dag()
    
    second_to_third = third*second.dag()
    second_to_fourth = fourth*second.dag()
    second_to_fifth = fifth*second.dag()
    second_to_sixth = sixth*second.dag()
    
    Transition_List = [gnd_to_first,gnd_to_second,
                       first_to_third,first_to_fourth,first_to_fifth,first_to_sixth,
                       second_to_third,second_to_fourth,second_to_fifth,second_to_sixth]
    
#    for j in range(4):
#        Number_gnd = basislist[j].dag()*gnd
#        Number_first = basislist[j].dag()*first
#        Number_second = basislist[j].dag()*second
#        EvecMat_gnd[i,j] = np.abs(Number_gnd[0,0])**2
#        EvecMat_first[i,j] = np.abs(Number_first[0,0])**2
#        EvecMat_second[i,j] = np.abs(Number_second[0,0])**2

    Eval_mat1a[i,:] = evals1 - evals1[0]
    Eval_mat2a[i,:] = evals1 - evals1[1]
    Eval_mat3a[i,:] = evals1 - evals1[2]
    
    Eval_mat1b[i,:] = evals1 - evals1[0]
    Eval_mat2b[i,:] = evals1 - evals1[1]
    Eval_mat3b[i,:] = evals1 - evals1[2]

    
#    coupling[i] = np.abs(qt.expect(Hint,gnd_to_fourth))
#    coupling2[i] = np.abs(qt.expect(Hint,first_to_fourth))
#    coupling3[i] = np.abs(qt.expect(Hint,second_to_fourth))
#    coupling4[i] = np.abs(qt.expect(Hint,third_to_fourth))
#    coupling[i] = np.abs(qt.expect(Hint,gnd_to_third))
#    coupling2[i] = np.abs(qt.expect(Hint,first_to_third))
#    coupling3[i] = np.abs(qt.expect(Hint,second_to_third))
#    coupling[i] = np.abs(qt.expect(Hopping_b,gnd_to_first))
#    coupling2[i] = np.abs(qt.expect(Hopping_b,first_to_second))
#    coupling3[i] = np.abs(qt.expect(Hopping_b,gnd_to_second))
    for j in range(10):
        Coupling_matA[i,j] = np.abs(qt.expect(Hopping_a,Transition_List[j]))
        Coupling_matB[i,j] = np.abs(qt.expect(Hopping_b,Transition_List[j]))
    
    if Coupling_matA[i,0]/(2*np.pi) < 0.1: 
        Eval_mat1a[i,1] = np.nan
    if Coupling_matA[i,1]/(2*np.pi) < 0.1:
        Eval_mat1a[i,2] = np.nan
    if Coupling_matA[i,2]/(2*np.pi) < 0.1:
        Eval_mat2a[i,3] = np.nan
    if Coupling_matA[i,3]/(2*np.pi) < 0.1:
        Eval_mat2a[i,4] = np.nan
    if Coupling_matA[i,4]/(2*np.pi) < 0.1:
        Eval_mat2a[i,5] = np.nan
    if Coupling_matA[i,5]/(2*np.pi) < 0.1:
        Eval_mat2a[i,6] = np.nan
    if Coupling_matA[i,6]/(2*np.pi) < 0.1:
        Eval_mat3a[i,3] = np.nan
    if Coupling_matA[i,7]/(2*np.pi) < 0.1:
        Eval_mat3a[i,4] = np.nan
    if Coupling_matA[i,8]/(2*np.pi) < 0.1:
        Eval_mat3a[i,5] = np.nan
    if Coupling_matA[i,9]/(2*np.pi) < 0.1:
        Eval_mat3a[i,6] = np.nan
        
    if Coupling_matB[i,0]/(2*np.pi) < 0.1: 
        Eval_mat1b[i,1] = np.nan
    if Coupling_matB[i,1]/(2*np.pi) < 0.1:
        Eval_mat1b[i,2] = np.nan
    if Coupling_matB[i,2]/(2*np.pi) < 0.1:
        Eval_mat2b[i,3] = np.nan
    if Coupling_matB[i,3]/(2*np.pi) < 0.1:
        Eval_mat2b[i,4] = np.nan
    if Coupling_matB[i,4]/(2*np.pi) < 0.1:
        Eval_mat2b[i,5] = np.nan
    if Coupling_matB[i,5]/(2*np.pi) < 0.1:
        Eval_mat2b[i,6] = np.nan
    if Coupling_matB[i,6]/(2*np.pi) < 0.1:
        Eval_mat3b[i,3] = np.nan
    if Coupling_matB[i,7]/(2*np.pi) < 0.1:
        Eval_mat3b[i,4] = np.nan
    if Coupling_matB[i,8]/(2*np.pi) < 0.1:
        Eval_mat3b[i,5] = np.nan
    if Coupling_matB[i,9]/(2*np.pi) < 0.1:
        Eval_mat3b[i,6] = np.nan

#
##         ===========================================================
##                            Plotting Energy Levels
##         ===========================================================
#
#for i in range(4):
#    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
#plt.ylabel(r'Frequency [GHz]')
#plt.xlabel(r'$\frac{\Phi}{2 \pi}$')
#plt.title(r'Transition Energies of Nonlinear Coupling, $H_{0}$')
#plt.show()
#
for i in range(1,3):
    if i==1:
        plt.plot(phi,(Eval_mat1a[:,i])/(2*np.pi),'b',label=r'$|\psi_{g}> \rightarrow 1-Manifold$')
    else:
        plt.plot(phi,(Eval_mat1a[:,i])/(2*np.pi),'b')
    plt.plot(phi,(Eval_mat1b[:,i])/(2*np.pi),'b')
for j in range(3,6):
    if j ==3:
        plt.plot(phi,(Eval_mat2a[:,j])/(2*np.pi),'r--',label=r'$|\psi_{1}> \rightarrow 2-Manifold$')
    else:
        plt.plot(phi,(Eval_mat2a[:,j])/(2*np.pi),'r--')
    plt.plot(phi,(Eval_mat2b[:,j])/(2*np.pi),'r--')
for k in range(3,6):
    if k == 3:
        plt.plot(phi,(Eval_mat3a[:,k])/(2*np.pi),'g-.',label=r'$|\psi_{2}> \rightarrow 2-Manifold$')    
    else:
        plt.plot(phi,(Eval_mat3a[:,k])/(2*np.pi),'g-.')
    plt.plot(phi,(Eval_mat3b[:,k])/(2*np.pi),'g-.')
plt.ylabel(r'Freqnecy [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'Transition Energies of Nonlinear Coupling')
#plt.legend[]
plt.grid()
plt.legend(loc='lower right')
plt.show()
    
#
#for i in range(3):
#    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
#plt.ylabel(r'Freqnecy [GHz]')
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.title(r'Transition Energies of Nonlinear Coupling, $|<\psi_{i}|\mathbb{1}\otimes(b+b^{\dagger})|\psi_{j}>|$')
#plt.grid()
#plt.show()

#for i in range(n*n-1):
#    plt.plot(phi,coupling_mat1[:,i]/(2*np.pi))
#plt.show()
#
#for i in range(n*n-1):
#    plt.plot(phi,coupling_mat2[:,i]/(2*np.pi))
#plt.show()

#for i in range(n**2*(n**2-1)):
#    plt.plot(phi,coupling_mat[]
#
##         ===========================================================
##                               Plotting Coupling
##         ===========================================================
#
for i in range(2):
    plt.plot(phi,Coupling_matA[:,i]/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'$<\psi_{i}|(a+a^{\dagger})\otimes \mathbb{1}|\psi_{j}>$')
plt.plot([-0.5,0.5],[0.1,0.1],'k')
plt.legend([r'$|\psi_{g}> \rightarrow |\psi_{1}>$',
            r'$|\psi_{g}> \rightarrow |\psi_{2}>$',
            r'Threshold'],loc='upper right')
plt.grid()
plt.show()
for i in range(2,6):
    plt.plot(phi,Coupling_matA[:,i]/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'$<\psi_{i}|(a+a^{\dagger})\otimes \mathbb{1}|\psi_{j}>$')
plt.plot([-0.5,0.5],[0.1,0.1],'k')
plt.legend([r'$|\psi_{1}> \rightarrow |\psi_{3}>$',
            r'$|\psi_{1}> \rightarrow |\psi_{4}>$',
            r'$|\psi_{1}> \rightarrow |\psi_{5}>$',
            r'$|\psi_{1}> \rightarrow |\psi_{6}>$',
            r'Threshold'],loc='upper right')
plt.grid()
plt.show()
for i in range(6,10):
    plt.plot(phi,Coupling_matA[:,i]/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'$<\psi_{i}|(a+a^{\dagger})\otimes \mathbb{1}|\psi_{j}>$')
plt.plot([-0.5,0.5],[0.1,0.1],'k')
plt.legend([r'$|\psi_{2}> \rightarrow |\psi_{3}>$',
            r'$|\psi_{2}> \rightarrow |\psi_{4}>$',
            r'$|\psi_{2}> \rightarrow |\psi_{5}>$',
            r'$|\psi_{2}> \rightarrow |\psi_{6}>$',
            r'Threshold'],loc='upper right')
plt.grid()
plt.show()

for i in range(2):
    plt.plot(phi,Coupling_matB[:,i]/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'$<\psi_{i}|\mathbb{1}\otimes (b+b^{\dagger})|\psi_{j}>$')
plt.plot([-0.5,0.5],[0.1,0.1],'k')
plt.legend([r'$|\psi_{g}> \rightarrow |\psi_{1}>$',
            r'$|\psi_{g}> \rightarrow |\psi_{2}>$',
            r'Threshold'],loc='upper right')
plt.grid()
plt.show()
for i in range(2,6):
    plt.plot(phi,Coupling_matB[:,i]/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'$<\psi_{i}|\mathbb{1}\otimes (b+b^{\dagger})|\psi_{j}>$')
plt.plot([-0.5,0.5],[0.1,0.1],'k')
plt.legend([r'$|\psi_{1}> \rightarrow |\psi_{3}>$',
            r'$|\psi_{1}> \rightarrow |\psi_{4}>$',
            r'$|\psi_{1}> \rightarrow |\psi_{5}>$',
            r'$|\psi_{1}> \rightarrow |\psi_{6}>$',
            r'Threshold'],loc='upper right')
plt.grid()
plt.show()
for i in range(6,10):
    plt.plot(phi,Coupling_matB[:,i]/(2*np.pi))
plt.ylabel(r'Frequency [GHz]')
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.title(r'$<\psi_{i}|\mathbb{1}\otimes (b+b^{\dagger})|\psi_{j}>$')
plt.plot([-0.5,0.5],[0.1,0.1],'k')
plt.legend([r'$|\psi_{2}> \rightarrow |\psi_{3}>$',
            r'$|\psi_{2}> \rightarrow |\psi_{4}>$',
            r'$|\psi_{2}> \rightarrow |\psi_{5}>$',
            r'$|\psi_{2}> \rightarrow |\psi_{6}>$',
            r'Threshold'],loc='upper right')
plt.grid()
plt.show()

#
#plt.plot(phi,coupling3/(2*np.pi))
#plt.plot(phi,coupling4/(2*np.pi))
#plt.ylabel(r'Frequency [GHz]')
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.title(r'$|<\psi_{i}|\mathbb{1}\otimes (b+b^{\dagger})|\psi_{j}>|$')
#plt.plot([-0.5,0.5],[0.04,0.04],'k')
#plt.legend([r'$|\psi_{g}> \rightarrow |\psi_{1}>$',r'$|\psi_{g}> \rightarrow |\psi_{2}>$',r'Threshold'],loc='upper right')
#plt.grid()
##plt.xlabel(r'$\frac{\Phi}{2\pi}$')
##plt.ylabel(r'Frequency [GHz]')
##plt.legend([r'$|\psi_{g}>\rightarrow |\psi_{1}>$',
##            r'$|\psi_{1}>\rightarrow|\psi_{2}>$',
##            r'$|\psi_{g}>\rightarrow|\psi_{2}>$'], loc='upper right')
##plt.title(r'$|<\psi_{i}|\mathbb{1}\otimes (b+b^{\dagger})|\psi_{j}>|$')
##plt.grid()
#plt.show()
#
##         ===========================================================
##                           Plotting Eigenstate Composition
##         ===========================================================
#
#for i in range(4):
#    plt.plot(phi,EvecMat_gnd[:,i])
#plt.grid()
#plt.title(r'Composition of Ground State')
#plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
#plt.ylabel(r'Occupation Probability')
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.show()
##
#for i in range(4):
#    plt.plot(phi,EvecMat_first[:,i])
#plt.title(r'Composition of First Excited State')
#plt.legend([r'|gg>',r'|eg>',r'|ge>',r'|ee>'])
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.ylabel(r'Occupation Probability')
#plt.grid()
#plt.show()
##
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

#H,H0,Hint = TwoQubitNonLin_wInt(17.0,16.0,65e-15,48.8e-15,20.0,40.0e-15,0) # Phi = -0.15 is where the eigenstate modes are most decoupled
#Evals,Evecs = H.eigenstates()
#
#
#tlist = np.linspace(0,400,401)
#
#psi0 = Evecs[2]
##psi0 = qt.tensor(qt.basis(n,1),qt.basis(n,0))
#
#gg_dm = qt.ket2dm(gg)
#ge_dm = qt.ket2dm(ge)
#eg_dm = qt.ket2dm(eg)
#ee_dm = qt.ket2dm(ee)
#
#Pe1 = qt.tensor(qt.fock_dm(n,1),qt.qeye(n))
#Pe2 = qt.tensor(qt.qeye(n),qt.fock_dm(n,1))
#
##swap = (Evecs[1]*Evecs[2].dag() + Evecs[2]*Evecs[1].dag())
#
#kappa1 = 0.01
#kappa2 = 0.02
#kappa3 = 0.05

##e_ops = [gg_dm,ge_dm,eg_dm,ee_dm]
#e_ops = [Evecs[1]*Evecs[1].dag(),Evecs[2]*Evecs[2].dag(),Evecs[3]*Evecs[3].dag(),Evecs[0]*Evecs[0].dag(),Pe1,Pe2]
#c_ops = [np.sqrt(kappa1)*a,np.sqrt(kappa2)*b]
##c_ops = []
#
#result = qt.mesolve(H,psi0,tlist,c_ops,e_ops,progress_bar=True)
#
#plt.plot(result.times,result.expect[0])
#plt.plot(result.times,result.expect[1])
#plt.plot(result.times,result.expect[2])
#plt.plot(result.times,result.expect[3])
##plt.plot(result.times,result.expect[4])
##plt.plot(result.times,result.expect[5])
#plt.xlabel(r'Time')
#plt.ylabel(r'Occupation Probability')
#plt.legend([r'$|\psi_{1}>$',r'$|\psi_{2}>$',r'$|\psi_3>$',r'$|\psi_g>$'])
#plt.grid()
#plt.show()
#
#plt.plot(result.times,result.expect[4])
#plt.plot(result.times,result.expect[5])
#plt.xlabel(r'Time')
#plt.ylabel(r'Stuff')
#plt.grid()
#plt.show()

