# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:24:47 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

n = 5
def QubitHam4_MatterSitesQubits_nocounter_rot(wq1,wq2,lambda1,lambda2,EjL,EjR,CL,CR,Ej,Cj,phi=0):
    #n = 5                #number of oscillator states
    
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
    
    #phi_L = np.sqrt(eps_L/2.0)*(a+a.dag())
    #phi_R = np.sqrt(eps_R/2.0)*(b+b.dag())
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0 = H_L + H_R + H_LR + Hqb
    
    H_int1 = -(2*np.pi)*(np.exp(-eps_L/4)*EjL*eps_L**2/16.0)*((a.dag()**2)*(a**2))
    H_int2 = -(2*np.pi)*(np.exp(-eps_R/4)*EjR*eps_R**2/16.0)*((b.dag()**2)*(b**2))
    H_int3 = -(2*np.pi)*(np.exp(-np.sqrt(eps_L*eps_R)/4)/16.0)*((eps_L**2)*(a.dag()**2)*(a**2) + (eps_R**2)*(b.dag()**2)*(b**2) + 4*eps_L*eps_R*(a.dag()*a)*(b.dag()*b) )
    #H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    H_int = H_int1 + H_int2 + H_int3
    
    H = H0 + H_int + H_lambda
    
    return H

H = QubitHam4_MatterSitesQubits_nocounter_rot(6,6.5,0.1,0.1,17.0,16.0,74e-15,67e-15,30.0,30.0e-15,-0.212381034483)
H.tidyup(atol=1e-1)
psi_0 = qt.tensor([qt.basis(2,1),qt.basis(n,0),qt.basis(n,1),qt.basis(2,0)])
tlist = np.linspace(0,20,300)

#qt.matrix_histogram(H)
#plt.show()

e_ops = []

# Create excitation states for qubits and oscillators
g_qb = qt.fock_dm(2,0)
e_qb = qt.fock_dm(2,1)
g_osc = qt.fock_dm(n,0)
e_osc = qt.fock_dm(n,1)
f_osc = qt.fock_dm(n,2)

P1 = qt.tensor([e_qb,g_osc,e_osc,g_qb])
P2 = qt.tensor([g_qb,e_osc,e_osc,g_qb])
P3 = qt.tensor([g_qb,e_osc,g_osc,e_qb])

#P_e1 = qt.tensor([e_qb,qt.qeye(n),qt.qeye(n),qt.qeye(2)])
#P_e2 = qt.tensor([qt.qeye(2),qt.qeye(n),qt.qeye(n),e_qb])
#P_1L = qt.tensor([qt.qeye(2),e_osc,qt.qeye(n),qt.qeye(2)])
#P_1R = qt.tensor([qt.qeye(2),qt.qeye(n),e_osc,qt.qeye(2)])
#P_2L = qt.tensor([qt.qeye(2),f_osc,qt.qeye(n),qt.qeye(2)])
#P_2R = qt.tensor([qt.qeye(2),qt.qeye(n),f_osc,qt.qeye(2)])


e_ops.append(P1)
e_ops.append(P2)
e_ops.append(P3)
#e_ops.append(P_1R)

xvec = np.linspace(-2,2,100)
#result = qt.mesolve(H,psi_0,tlist,[],e_ops,progress_bar=True)
result = qt.mesolve(H,psi_0,tlist,[],[],progress_bar=True)
#plt.plot(result.times,result.expect[0])
#plt.plot(result.times,result.expect[1])
#plt.plot(result.times,result.expect[2])
##plt.plot(result.times,result.expect[3])
#plt.xlabel(r'Time')
#plt.ylabel(r'Occupation Probability')
#plt.title(r'Initial State |egeg>')
#plt.legend([r'|egeg>',r'|geeg>',r'|gege>'])
#plt.show()

def compute_wigner(rho):
    return qt.wigner(rho,xvec,xvec)

W_list = qt.parfor(compute_wigner,result.states)

import matplotlib.animation as ani
import matplotlib.colors as clr
import types

fname = "Evolve.gif"
fig, ax = plt.subplots(1,1,figsize=(12,5))
def animate(n):
    print(n)
    ax[0].cla(); ax[0].set_aspect("equal"); ax[0].tick_params(labelsize=14)
    ax[0].set_title("Time: %.2f"%(result.times[n]),fontsize=14)
    ax[0].set_xlabel(r'$x$',fontsize=14); ax[0].set_ylabel(r'$p$',fontsize=14)
    im = ax[0].contourf(xvec,xvec,W_list[n],100,norm=clr.Normalize(-0.25,0.25),cmap=plt.get_cmap("RdBu"))
    def setvisible(self,vis):
        for c in self.collections: c.set_visible(vis)
    im.set_visible = types.MethodType(setvisible,im)
anim = ani.FuncAnimation(fig,animate,frames=len(result.times))
anim.save(fname,writer="imagemagick",fps=20)
#phi = np.linspace(-0.5,0.5,100)
#
#Eval_mat1 = np.zeros((len(phi),2*2*n*n))
#
#for i,Phi in enumerate(phi):
#    if (i %(len(phi)/10))==0:
#        print('%f Percent Completed' %(i/len(phi)*100))
#    H = QubitHam4_MatterSitesQubits_nocounter_rot(6.5,6.25,0.1,0.1,17.0,16.0,74e-15,67e-15,30.0,30.0e-15,Phi)
#    H.tidyup()
#    evals = H.eigenenergies()
#    Eval_mat1[i,:] = evals
#    
#for i in range(10):
#    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.ylabel(r'Frequency [GHz]')
#plt.title(r'Transition Energies of 4 Qubit Hamiltonian')
#plt.show()