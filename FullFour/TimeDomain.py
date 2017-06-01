# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:51:46 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

n = 4
def QubitHam4_MatterSitesQubits(wq1,wq2,lambda1,lambda2,EjL,EjR,CL,CR,Ej,Cj,phi=0):
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


H = QubitHam4_MatterSitesQubits(6,6,0.1,0.1,17.0,16.0,74e-15,67e-15,30.0,30.0e-15,-0.212381034483)
psi_0 = qt.tensor([qt.basis(2,1),qt.basis(n,0),qt.basis(n,1),qt.basis(2,0)])
tlist = np.linspace(0,50,1000)

e_ops = []

e_qb = qt.basis(2,1)
P_eqb = e_qb*e_qb.dag()
P_eqb1 = qt.tensor([P_eqb,qt.qeye(n),qt.qeye(n),qt.qeye(2)])
P_eqb2 = qt.tensor([qt.qeye(2),qt.qeye(n),qt.qeye(n),P_eqb])

e_osc = qt.basis(n,1)
P_osc = e_osc*e_osc.dag()
P_osc1 = qt.tensor([qt.qeye(2),P_osc,qt.qeye(n),qt.qeye(2)])
P_osc2 = qt.tensor([qt.qeye(2),qt.qeye(n),P_osc,qt.qeye(2)])

e_ops.append(P_eqb1*P_osc2)


result = qt.mesolve(H,psi_0,tlist,[],e_ops)

def func(t,A,f,phi,C):
    return A*np.cos(2*np.pi*f*t-phi*np.pi/180)**2.0+C
def func2(t,A,f,phi,C):
    return A*np.cos(2*np.pi*f*t-phi*np.pi/180)**4.0+C

popt, pcov = opt.curve_fit(func,result.times,result.expect[2],bounds=([0.2,0,0,0],[1.0,0.05,360,1]))
#print(popt)
popt2, pcov2 = opt.curve_fit(func2,result.times,result.expect[2],bounds=([0.2,0,0,0],[1.0,0.05,360,1]))

plt.plot(result.times,result.expect[0])
plt.plot(result.times,result.expect[1])
plt.plot(result.times,result.expect[2])
plt.plot(result.times,result.expect[3])
plt.plot(result.times,func(result.times, *popt))
plt.plot(result.times,func2(result.times, *popt2))
plt.xlabel(r'Time')
plt.ylabel(r'Occupation Probability')
plt.title(r'Initial State |egeg>')
plt.legend([r'<$P_{\uparrow 1}$>',r'<$P_{\uparrow 2}$>',r'<$P_{\uparrow 1}P_{1R}$>',r'<$P_{\uparrow 2}P_{1L}$>'])
plt.show()